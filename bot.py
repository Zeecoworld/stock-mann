"""
paper_trader/bot.py  — v3  PRODUCTION-READY
────────────────────────────────────────────────────────────────────────────
FIXES vs v2:
  1. NO-SIGNAL FIX: sentiment_threshold lowered to 0.15, confidence to 0.35
     — Llama-3 on neutral news regularly scores 0.2–0.35; v2 thresholds were
     silently blocking every single signal on slow news days.
  2. NO-SIGNAL FIX: trend_min_mentions default = 0 (filter disabled) so
     tickers are never skipped due to stale/rate-limited RSS trending feeds.
  3. NO-SIGNAL FIX: Yahoo Finance price fetcher now tries v8 → v7 → v10
     fallback chain — v8 alone has become flaky in cloud deployments.
  4. NO-SIGNAL FIX: require_market_hours defaults to False in paper mode;
     PRODUCTION=true automatically enforces market hours.
  5. PRODUCTION FIX: When PRODUCTION=true, AlpacaBroker auto-injected as
     the portfolio — real orders go to Alpaca paper/live API.
  6. ALPACA MCP: fetch_alpaca_mcp_context() pulls live account state from
     alpaca-mcp-server (github.com/tedlikeskix/alpaca-mcp-server) and
     injects it into the Llama-3 prompt so the LLM knows current positions,
     cash, and recent orders before deciding BUY/SELL.
  7. PolymarketSentimentStrategy import made optional (file may not exist).
  8. Every signal outcome is now explicitly logged — no more silent SKIP.

ENV VARS REQUIRED:
  REPLICATE_API_TOKEN   — Replicate / Llama-3 scorer
  ALPACA_API_KEY        — Alpaca paper or live key
  ALPACA_SECRET_KEY     — Alpaca secret
  ALPACA_BASE_URL       — default: https://paper-api.alpaca.markets
  PRODUCTION            — "true" to use AlpacaBroker + enforce market hours

OPTIONAL:
  ALPACA_MCP_URL        — alpaca-mcp-server URL (default http://localhost:3000)
  FINNHUB_API_KEY       — extra news headlines
  NEWSAPI_KEY           — extra news headlines
"""
from __future__ import annotations

import asyncio, logging, os, time
from typing import Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Bot")

import sys, pathlib
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

from portfolio import PaperPortfolio
from guards    import (MarketHoursGuard, DrawdownCircuitBreaker,
                       DuplicateSignalThrottle, TrendingTickerFilter)

try:
    from strategy_engine.engine          import MarketContext, Signal, SignalBus, StrategyEngine, TradeSignal
    from strategy_engine.sentiment_agent import SentimentAgent
    from strategy_engine.stock_strategy  import StockSentimentStrategy
    from strategy_engine.news_fetcher    import GoogleNewsRSSFetcher
except ImportError:
    from strategy_engine.engine          import MarketContext, Signal, SignalBus, StrategyEngine, TradeSignal
    from strategy_engine.sentiment_agent import SentimentAgent
    from strategy_engine.stock_strategy  import StockSentimentStrategy
    from strategy_engine.news_fetcher    import GoogleNewsRSSFetcher

# Optional Polymarket strategy
try:
    try:
        from strategy_engine.polymarket_strategy import PolymarketSentimentStrategy
    except ImportError:
        from strategy_engine.polymarket_strategy import PolymarketSentimentStrategy
    _HAS_POLYMARKET = True
except ImportError:
    _HAS_POLYMARKET = False
    logger.warning("[Bot] polymarket_strategy not found — Polymarket signals disabled")

PRODUCTION = os.getenv("PRODUCTION", "false").lower() == "true"
ALPACA_MCP_URL = os.getenv("ALPACA_MCP_URL", "http://localhost:3000")

DEFAULT_WATCHLIST = [
    "$NVDA","$TSLA","$AAPL","$MSFT","$AMZN",
    "$META","$GOOGL","$AMD","$PLTR","$COIN",
]


# ─────────────────────────────────────────────────────────────────────────────
# Price fetcher — v8 → v7 → v10 fallback chain
# FIX: v8 alone is flaky in cloud IPs; we now try three Yahoo endpoints
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_stock_price(ticker: str) -> Optional[Dict]:
    symbol = ticker.lstrip("$").upper()

    async def _try_v8():
        async with httpx.AsyncClient(timeout=12) as c:
            r = await c.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"interval": "1d", "range": "1mo"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()
            closes = r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            closes = [x for x in closes if x is not None]
            if not closes: return None
            price = closes[-1]
            ma_20 = sum(closes[-20:]) / len(closes[-20:]) if len(closes) >= 5 else price
            return {"price": round(price, 2), "ma_20": round(ma_20, 2)}

    async def _try_v7():
        async with httpx.AsyncClient(timeout=12) as c:
            r = await c.get(
                "https://query1.finance.yahoo.com/v7/finance/quote",
                params={"symbols": symbol},
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            )
            r.raise_for_status()
            data = r.json()["quoteResponse"]["result"]
            if not data: return None
            q = data[0]
            price = q.get("regularMarketPrice") or q.get("postMarketPrice")
            if not price: return None
            ma_20 = q.get("fiftyDayAverage") or price
            return {"price": round(float(price), 2), "ma_20": round(float(ma_20), 2)}

    async def _try_v10():
        async with httpx.AsyncClient(timeout=12) as c:
            r = await c.get(
                f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}",
                params={"modules": "price"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()
            pd = r.json()["quoteSummary"]["result"][0]["price"]
            price = pd["regularMarketPrice"]["raw"]
            ma_20 = pd.get("fiftyDayAverage", {}).get("raw", price)
            return {"price": round(float(price), 2), "ma_20": round(float(ma_20), 2)}

    for fn, label in [(_try_v8, "v8"), (_try_v7, "v7"), (_try_v10, "v10")]:
        try:
            result = await fn()
            if result:
                logger.debug("[Price/%s] %s $%.2f MA20=$%.2f",
                             label, symbol, result["price"], result["ma_20"])
                return result
        except Exception as e:
            logger.debug("[Price/%s] %s failed: %s", label, symbol, e)

    logger.warning("[Price] All sources failed for %s", symbol)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca MCP context fetcher
# Injects live account state into the LLM prompt via alpaca-mcp-server.
# Install: https://github.com/tedlikeskix/alpaca-mcp-server
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_alpaca_mcp_context(ticker: str) -> str:
    """
    Queries the local alpaca-mcp-server for live account + position context.
    Returns formatted string injected into the Llama-3 sentiment prompt so
    the LLM can reason: "We already hold NVDA, don't buy more" or
    "We have $8k cash, confident BUY is executable."
    Fails silently if MCP server not running.
    """
    symbol = ticker.lstrip("$").upper()
    try:
        async with httpx.AsyncClient(timeout=6) as c:
            r_acct = await c.post(f"{ALPACA_MCP_URL}/tools/call",
                                  json={"name": "get_account", "arguments": {}},
                                  headers={"Content-Type": "application/json"})
            acct = r_acct.json().get("content", [{}])[0].get("text", "unavailable")

            r_pos = await c.post(f"{ALPACA_MCP_URL}/tools/call",
                                 json={"name": "get_positions", "arguments": {}},
                                 headers={"Content-Type": "application/json"})
            positions = r_pos.json().get("content", [{}])[0].get("text", "unavailable")

            r_ord = await c.post(f"{ALPACA_MCP_URL}/tools/call",
                                 json={"name": "get_orders",
                                       "arguments": {"status": "all", "limit": 10}},
                                 headers={"Content-Type": "application/json"})
            orders = r_ord.json().get("content", [{}])[0].get("text", "unavailable")

        return (
            f"\n\n=== LIVE ALPACA BROKERAGE CONTEXT ===\n"
            f"Account summary: {acct}\n"
            f"Current open positions: {positions}\n"
            f"Recent orders (last 10): {orders}\n"
            f"Symbol being evaluated now: {symbol}\n"
            f"Use this context when forming your sentiment verdict.\n"
            f"======================================\n"
        )
    except Exception as e:
        logger.debug("[AlpacaMCP] Skipped (server not running?): %s", e)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Ticker-specific news fetcher
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_ticker_headlines(ticker: str, max_results: int = 15) -> List[str]:
    clean = ticker.lstrip("$").upper()
    queries = [clean, f"{clean} stock", f"{clean} earnings"]
    all_headlines: List[str] = []
    seen: set = set()

    results = await asyncio.gather(
        *[GoogleNewsRSSFetcher().fetch(q, max_results=8) for q in queries],
        return_exceptions=True
    )
    for batch in results:
        if isinstance(batch, Exception): continue
        for h in batch:
            h = h.strip()
            if h and h not in seen:
                seen.add(h); all_headlines.append(h)

    relevant = [h for h in all_headlines if clean.lower() in h.lower()]
    fallback  = [h for h in all_headlines if h not in relevant]
    combined  = (relevant + fallback)[:max_results]
    logger.info("[News] %s: %d relevant + %d fallback = %d total",
                clean, len(relevant), len(fallback), len(combined))
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Main Bot
# ─────────────────────────────────────────────────────────────────────────────

class TradingBot:
    def __init__(
        self,
        watchlist:            List[str] = None,
        interval_minutes:     int       = 15,
        sentiment_threshold:  float     = 0.15,   # FIX: was 0.25 — too high
        confidence_threshold: float     = 0.35,   # FIX: was 0.45/0.50 — too high
        require_market_hours: bool      = None,   # None = auto (False=paper, True=prod)
        daily_drawdown_limit: float     = 0.03,
        total_drawdown_limit: float     = 0.10,
        signal_cooldown_min:  int       = 30,
        trend_min_mentions:   int       = 0,      # FIX: 0 = filter disabled (fail-open)
        replicate_model:      str       = "meta/meta-llama-3-70b-instruct",
        min_position_size:    float     = 50.0,
        use_alpaca_mcp:       bool      = True,
    ):
        self.watchlist         = watchlist or DEFAULT_WATCHLIST
        self.interval          = interval_minutes * 60
        self.min_position_size = min_position_size
        self.use_alpaca_mcp    = use_alpaca_mcp
        self._trend_min        = trend_min_mentions

        # Auto-derive market hours enforcement from PRODUCTION flag
        if require_market_hours is None:
            require_market_hours = PRODUCTION

        # Portfolio — paper vs production
        if PRODUCTION:
            logger.info("[Bot] 🔴 PRODUCTION mode — using AlpacaBroker")
            try:
                from broker_alpaca import AlpacaBroker
                self.portfolio = AlpacaBroker()
            except Exception as e:
                logger.error("[Bot] AlpacaBroker init failed (%s) — falling back to paper", e)
                self.portfolio = self._make_paper_portfolio()
        else:
            logger.info("[Bot] 📄 PAPER mode — using PaperPortfolio")
            self.portfolio = self._make_paper_portfolio()

        # Sentiment agent
        self.agent = SentimentAgent(model=replicate_model)

        # Strategy engine
        self.bus    = SignalBus()
        self.engine = StrategyEngine(bus=self.bus, sentiment_agent=self.agent)
        self.engine.register(StockSentimentStrategy(
            sentiment_threshold=sentiment_threshold,
            confidence_threshold=confidence_threshold,
        ))
        if _HAS_POLYMARKET:
            self.engine.register(PolymarketSentimentStrategy(
                sentiment_threshold=sentiment_threshold))

        self.bus.subscribe(self._on_signal)

        # Guards
        self.hours_guard = MarketHoursGuard() if require_market_hours else None
        if not self.hours_guard:
            logger.info("[Bot] ⚠  Market hours guard DISABLED"
                        " (set PRODUCTION=true to enforce)")

        self.breaker      = DrawdownCircuitBreaker(
            self.portfolio, daily_limit=daily_drawdown_limit,
            total_limit=total_drawdown_limit)
        self.throttle     = DuplicateSignalThrottle(cooldown_minutes=signal_cooldown_min)
        self.trend_filter = TrendingTickerFilter()

        self._prices:  Dict[str, float] = {}
        self._signals: List[dict]       = []
        self._skipped: List[str]        = []

    @staticmethod
    def _make_paper_portfolio():
        return PaperPortfolio(
            starting_cash=10_000.0,
            risk_per_trade=0.05,
            max_positions=5,
            stop_loss_pct=0.05,
            take_profit_pct=0.12,
            min_order_size=50.0,
        )

    # ── Signal handler ────────────────────────────────────────────────────

    def _on_signal(self, sig: TradeSignal):
        self._signals.append({
            "time":       time.strftime("%H:%M:%S"),
            "ticker":     sig.ticker,
            "signal":     sig.signal.value,
            "confidence": round(sig.confidence, 3),
            "reasoning":  sig.reasoning,
            "source":     sig.source,
        })
        icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "SKIP": "⚪"}.get(
            sig.signal.value, "❓")
        print(f"  {icon} [{sig.signal.value:4s}]  {sig.ticker:<12}  "
              f"conf={sig.confidence:.0%}  |  {sig.reasoning[:100]}")

    # ── Scan cycle ────────────────────────────────────────────────────────

    async def run_once(self, tickers: List[str] = None) -> dict:
        tickers = tickers or self.watchlist

        print(f"\n{'═'*70}")
        print(f"  📡  Scan  {time.strftime('%Y-%m-%d  %H:%M:%S')}"
              f"  [{'PRODUCTION 🔴' if PRODUCTION else 'PAPER 📄'}]")
        print(f"{'═'*70}")

        # Guard 1 — market hours
        if self.hours_guard and not self.hours_guard.is_open():
            wait = self.hours_guard.seconds_until_open()
            print(f"\n  ⏸  {self.hours_guard.reason}")
            print(f"     Next open in {wait // 3600}h {(wait % 3600) // 60}m")
            return self._snapshot(blocked=self.hours_guard.reason)

        # Guard 2 — drawdown circuit breaker
        prices_for_guard = {k.lstrip("$"): v for k, v in self._prices.items()}
        if self.breaker.is_tripped(prices_for_guard):
            print(f"\n  🚨  {self.breaker.reason}\n")
            return self._snapshot(blocked=self.breaker.reason)
        print(f"  🛡  {self.breaker.reason}")

        # Guard 3 — trending feed refresh
        await self.trend_filter.refresh()
        top = self.trend_filter.top_trending(6)
        if top:
            print(f"  📈  Trending: {', '.join(f'{s}({c})' for s, c in top)}\n")
        else:
            print(f"  📈  Trending: (feed empty — scanning all tickers)\n")

        self._skipped = []

        for ticker in tickers:
            print(f"\n  ▶  {ticker}")

            # Trend gate (disabled when _trend_min == 0)
            if self._trend_min > 0 and not self.trend_filter.is_trending(ticker, self._trend_min):
                mentions = self.trend_filter.mention_count(ticker)
                print(f"     ⚪ {mentions} mention(s) < threshold {self._trend_min} → skip")
                self._skipped.append(ticker)
                continue

            # Price fetch — 3-endpoint fallback
            pd = await fetch_stock_price(ticker)
            if not pd:
                print(f"     ⚠  Price unavailable → skip")
                self._skipped.append(ticker)
                continue

            price, ma_20 = pd["price"], pd["ma_20"]
            clean_sym = ticker.lstrip("$").upper()
            self._prices[ticker]    = price
            self._prices[clean_sym] = price
            print(f"     💲 ${price:.2f}  MA20=${ma_20:.2f}  "
                  f"mentions={self.trend_filter.mention_count(ticker)}")

            # Headlines
            headlines = await fetch_ticker_headlines(ticker, max_results=15)
            print(f"     📰 {len(headlines)} headlines → Replicate…")
            if headlines:
                print(f"     📄 Top: {headlines[0][:80]}")

            # Alpaca MCP context injection
            mcp_context = ""
            if self.use_alpaca_mcp:
                mcp_context = await fetch_alpaca_mcp_context(ticker)
                if mcp_context:
                    print(f"     🔗 Alpaca MCP context injected")

            enriched_news = headlines + ([mcp_context] if mcp_context else [])

            # Strategy evaluation
            signal = await self.engine.run(MarketContext(
                ticker=ticker, source="stock",
                price=price, ma_20=ma_20,
                raw_news=enriched_news,
            ))

            if not signal:
                print(f"     ⚠  No strategy matched source='stock'")
                continue

            # Always print the signal — no silent outcomes
            print(f"     → {signal.signal.value}  conf={signal.confidence:.0%}  "
                  f"| {signal.reasoning[:100]}")

            if signal.signal in (Signal.BUY, Signal.SELL):
                if self.throttle.is_blocked(ticker, signal.signal.value):
                    print(f"     🔁 Throttled — {signal.signal.value} on {ticker} in cooldown")
                    continue
                self._execute(signal, price)
                self.throttle.record(ticker, signal.signal.value)

            # Stop/take-profit check after each ticker
            stops = self.portfolio.check_stops(self._prices)
            for s in stops:
                print(f"  🛑 STOP: {s.ticker}  PnL=${s.pnl:+.2f}  {s.reasoning}")

        print(f"\n{'─'*70}")
        print(f"  {self.portfolio.summary_str(self._prices)}")
        if self._skipped:
            print(f"  ⚪ Skipped: {', '.join(self._skipped)}")
        print(f"{'─'*70}\n")

        return self._snapshot()

    def _execute(self, signal: TradeSignal, price: float):
        ticker = signal.ticker
        if signal.signal == Signal.BUY:
            if self.portfolio.position_exists(ticker):
                print(f"     ℹ  Already holding {ticker} — skip buy")
                return
            rec = self.portfolio.execute_buy(
                ticker=ticker, price=price,
                confidence=signal.confidence,
                reasoning=signal.reasoning,
                source=signal.source,
            )
            if rec:
                print(f"     ✅ BUY  {ticker}  {rec.shares:.4f}sh @ ${price:.2f}"
                      f"  cost=${rec.shares * price:.2f}")
            else:
                print(f"     ❌ BUY FAILED {ticker} — see portfolio logs")

        elif signal.signal == Signal.SELL:
            if self.portfolio.position_exists(ticker):
                rec = self.portfolio.execute_sell(
                    ticker=ticker, price=price,
                    confidence=signal.confidence,
                    reasoning=signal.reasoning,
                )
                if rec:
                    print(f"     ✅ SELL {ticker}  PnL=${rec.pnl:+.2f}")
            else:
                print(f"     ℹ  SELL signal — no position in {ticker} (no short)")

    def _snapshot(self, blocked: str = "") -> dict:
        snap = self.portfolio.snapshot(self._prices)
        snap["signals"]  = self._signals[-50:]
        snap["skipped"]  = self._skipped
        snap["blocked"]  = blocked
        snap["trending"] = dict(self.trend_filter._counts)
        return snap

    # ── Continuous loop ───────────────────────────────────────────────────

    async def run_loop(self, interval_minutes: int = 15):
        mode = "PRODUCTION 🔴 LIVE" if PRODUCTION else "PAPER 📄"
        strat = self.engine._strategies[0] if self.engine._strategies else None
        print(f"\n🚀  TradingBot v3  [{mode}]")
        print(f"   Model         : Replicate / Llama-3-70B")
        print(f"   Watchlist     : {', '.join(self.watchlist)}")
        print(f"   Interval      : {interval_minutes} min")
        if hasattr(self.portfolio, 'starting_cash'):
            print(f"   Capital       : ${self.portfolio.starting_cash:,.2f}")
            print(f"   Risk/trade    : {self.portfolio.risk_per_trade:.0%}")
        if strat:
            print(f"   Sent threshold: {strat.sentiment_threshold}")
            print(f"   Conf threshold: {strat.confidence_threshold}")
        print(f"   Hours guard   : {'ON' if self.hours_guard else 'OFF'}")
        print(f"   Trend filter  : {self._trend_min} mentions (0=off)")
        print(f"   Alpaca MCP    : {'ON → ' + ALPACA_MCP_URL if self.use_alpaca_mcp else 'OFF'}")
        print()

        while True:
            snap = await self.run_once()
            if hasattr(self.portfolio, 'save'):
                self.portfolio.save("portfolio.json")

            blocked = snap.get("blocked", "")
            if blocked and "Market" in blocked and self.hours_guard:
                wait  = self.hours_guard.seconds_until_open()
                sleep = min(wait, interval_minutes * 60)
                print(f"  💤 Sleeping {sleep // 60} min until market opens…\n")
                await asyncio.sleep(sleep)
            else:
                print(f"  ⏱  Next scan in {interval_minutes} min…\n")
                await asyncio.sleep(interval_minutes * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    import argparse
    p = argparse.ArgumentParser(description="Trading Bot v3")
    p.add_argument("--tickers",   nargs="*", default=None)
    p.add_argument("--loop",      action="store_true")
    p.add_argument("--interval",  type=int,   default=15)
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--conf",      type=float, default=0.35)
    p.add_argument("--no-hours",  action="store_true")
    p.add_argument("--daily-dd",  type=float, default=0.03)
    p.add_argument("--total-dd",  type=float, default=0.10)
    p.add_argument("--trend-min", type=int,   default=0)
    p.add_argument("--model",     type=str,
                   default="meta/meta-llama-3-70b-instruct")
    p.add_argument("--no-mcp",    action="store_true")
    args = p.parse_args()

    bot = TradingBot(
        watchlist=args.tickers,
        interval_minutes=args.interval,
        sentiment_threshold=args.threshold,
        confidence_threshold=args.conf,
        require_market_hours=False if args.no_hours else None,
        daily_drawdown_limit=args.daily_dd,
        total_drawdown_limit=args.total_dd,
        trend_min_mentions=args.trend_min,
        replicate_model=args.model,
        use_alpaca_mcp=not args.no_mcp,
    )

    if args.loop:
        await bot.run_loop(args.interval)
    else:
        await bot.run_once(args.tickers)


if __name__ == "__main__":
    asyncio.run(main())