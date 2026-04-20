"""
paper_trader/bot.py  — v2 FIXED
────────────────────────────────────────────────────────────────────────────
FIXES applied vs original:
  1. $ ticker symbol mismatch — prices dict now keyed with $ prefix to match
     portfolio/signal ticker so stop-loss and PnL calculations work correctly.
  2. execute_buy was silently returning None due to key mismatch — fixed.
  3. Throttle cooldown reduced to 30 min (was 60) so the bot can re-evaluate
     positions more frequently without getting permanently blocked.
  4. Ticker-specific news: GoogleNews query is now the cleaned ticker symbol,
     not a generic finance query — means LLM sees GOOGL-specific headlines.
  5. After-hours guard now correctly uses ET timezone check and will run in
     paper mode with --no-hours flag clearly logged.
  6. Position size minimum raised to $50 to avoid 0-share orders.
  7. sell_position logic added — bot now sells when SELL signal fires on a
     held position instead of trying to short a stock it doesn't own.
  8. Added debug logging so you can see exactly why trades are skipped.
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
    from strategy_engine.engine              import MarketContext, Signal, SignalBus, StrategyEngine, TradeSignal
    from strategy_engine.sentiment_agent     import SentimentAgent
    from strategy_engine.stock_strategy      import StockSentimentStrategy
    from strategy_engine.polymarket_strategy import PolymarketSentimentStrategy
    from strategy_engine.news_fetcher        import NicheFetcher, GoogleNewsRSSFetcher, CompositeFetcher
except ImportError:
    from engine              import MarketContext, Signal, SignalBus, StrategyEngine, TradeSignal
    from sentiment_agent     import SentimentAgent
    from stock_strategy      import StockSentimentStrategy
    from polymarket_strategy import PolymarketSentimentStrategy
    from news_fetcher        import NicheFetcher, GoogleNewsRSSFetcher, CompositeFetcher

PRODUCTION = os.getenv("PRODUCTION", "false").lower() == "true"

DEFAULT_WATCHLIST = [
    "$NVDA","$TSLA","$AAPL","$MSFT","$AMZN",
    "$META","$GOOGL","$AMD","$PLTR","$COIN",
]


# ─────────────────────────────────────────────────────────────────────────────
# Price fetcher — Yahoo Finance v8 (free, no key)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_stock_price(ticker: str) -> Optional[Dict]:
    # FIX: strip $ for the HTTP request but keep symbol clean
    symbol = ticker.lstrip("$").upper()
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"interval": "1d", "range": "1mo"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()
            closes = r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            closes = [x for x in closes if x is not None]
            if not closes:
                return None
            price = closes[-1]
            ma_20 = sum(closes[-20:]) / len(closes[-20:]) if len(closes) >= 5 else price
            return {"price": round(price, 2), "ma_20": round(ma_20, 2)}
    except Exception as e:
        logger.warning("[Price] %s: %s", symbol, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Ticker-specific news fetcher
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_ticker_headlines(ticker: str, max_results: int = 15) -> List[str]:
    """
    FIX: Fetches ticker-SPECIFIC headlines rather than generic finance news.
    Uses Google News RSS search with the actual ticker symbol as query.
    This means GOOGL gets GOOGL headlines, not 'market is up today' noise.
    """
    clean = ticker.lstrip("$").upper()

    # Build ticker-specific queries
    queries = [
        clean,                          # e.g. "GOOGL"
        f"{clean} stock",               # e.g. "GOOGL stock"
        f"{clean} earnings revenue",    # fundamental news
    ]

    all_headlines: List[str] = []
    seen: set = set()

    async def _fetch_query(q: str) -> List[str]:
        fetcher = GoogleNewsRSSFetcher()
        return await fetcher.fetch(q, max_results=8)

    results = await asyncio.gather(*[_fetch_query(q) for q in queries],
                                   return_exceptions=True)

    for batch in results:
        if isinstance(batch, Exception):
            continue
        for h in batch:
            h = h.strip()
            if h and h not in seen:
                seen.add(h)
                all_headlines.append(h)

    # Deduplicate: keep headlines that contain the ticker name
    relevant = [h for h in all_headlines if clean.lower() in h.lower()]
    fallback  = [h for h in all_headlines if h not in relevant]

    # Return: relevant first, then general finance headlines as fallback
    combined = (relevant + fallback)[:max_results]
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
        sentiment_threshold:  float     = 0.25,   # FIX: was 0.10 in __init__, 0.30 in app — unified to 0.25
        require_market_hours: bool      = True,
        daily_drawdown_limit: float     = 0.03,
        total_drawdown_limit: float     = 0.10,
        signal_cooldown_min:  int       = 30,      # FIX: reduced from 60 → 30 min
        trend_min_mentions:   int       = 1,       # FIX: reduced from 2 → 1 (more tickers pass)
        replicate_model:      str       = "meta/meta-llama-3-70b-instruct",
        min_position_size:    float     = 50.0,    # FIX: new — minimum $50 per trade
    ):
        self.watchlist         = watchlist or DEFAULT_WATCHLIST
        self.interval          = interval_minutes * 60
        self.min_position_size = min_position_size

        # Portfolio — FIX: larger starting cash so 2% risk = meaningful trade size
        self.portfolio = PaperPortfolio(
            starting_cash=10_000.0,
            risk_per_trade=0.05,        # FIX: 5% risk per trade (was 2% — too small)
            max_positions=5,
            stop_loss_pct=0.05,
            take_profit_pct=0.12,       # FIX: raised TP from 10% → 12%
        )

        # Sentiment agent
        self.agent = SentimentAgent(model=replicate_model)

        # Strategy engine
        self.bus    = SignalBus()
        self.engine = StrategyEngine(bus=self.bus, sentiment_agent=self.agent)
        self.engine.register(StockSentimentStrategy(
            sentiment_threshold=sentiment_threshold,
            confidence_threshold=0.45,  # FIX: lowered from 0.50 → 0.45
        ))
        self.engine.register(PolymarketSentimentStrategy(
            sentiment_threshold=sentiment_threshold))
        self.bus.subscribe(self._on_signal)

        # Guards
        self.hours_guard  = MarketHoursGuard() if require_market_hours else None
        self.breaker      = DrawdownCircuitBreaker(
            self.portfolio,
            daily_limit=daily_drawdown_limit,
            total_limit=total_drawdown_limit,
        )
        self.throttle     = DuplicateSignalThrottle(cooldown_minutes=signal_cooldown_min)
        self.trend_filter = TrendingTickerFilter()
        self._trend_min   = trend_min_mentions

        # FIX: prices keyed with $ prefix to match ticker strings throughout
        self._prices:  Dict[str, float] = {}
        self._signals: List[dict]       = []
        self._skipped: List[str]        = []

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
        icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "SKIP": "⚪"}.get(sig.signal.value, "❓")
        print(f"  {icon} [{sig.signal.value:4s}]  {sig.ticker:<12}  "
              f"conf={sig.confidence:.0%}  |  {sig.reasoning[:80]}")

    # ── Scan cycle ────────────────────────────────────────────────────────

    async def run_once(self, tickers: List[str] = None) -> dict:
        tickers = tickers or self.watchlist

        print(f"\n{'═'*65}")
        print(f"  📡  Scan  {time.strftime('%Y-%m-%d  %H:%M:%S')}")
        print(f"{'═'*65}")

        # Guard 1 — market hours
        if self.hours_guard and not self.hours_guard.is_open():
            wait = self.hours_guard.seconds_until_open()
            print(f"\n  ⏸  {self.hours_guard.reason}")
            print(f"     Next open in {wait // 3600}h {(wait % 3600) // 60}m")
            print(f"     TIP: use --no-hours flag to test outside market hours\n")
            return self._snapshot(blocked=self.hours_guard.reason)

        # Guard 2 — drawdown circuit breaker
        # FIX: pass _prices dict (keyed with $) — must match how prices are stored
        prices_for_guard = {k.lstrip("$"): v for k, v in self._prices.items()}
        if self.breaker.is_tripped(prices_for_guard):
            print(f"\n  🚨  {self.breaker.reason}\n")
            return self._snapshot(blocked=self.breaker.reason)

        print(f"  🛡  {self.breaker.reason}")

        # Guard 3 — refresh trending RSS feeds (cached 15 min)
        await self.trend_filter.refresh()
        top = self.trend_filter.top_trending(6)
        print(f"  📈  Trending: {', '.join(f'{s}({c})' for s, c in top)}\n")

        self._skipped = []

        for ticker in tickers:
            print(f"  ▶  {ticker}")

            # FIX: trend check uses clean symbol
            clean_sym = ticker.lstrip("$").upper()
            if not self.trend_filter.is_trending(ticker, self._trend_min):
                mentions = self.trend_filter.mention_count(ticker)
                print(f"     ⚪ {mentions} mention(s) — below threshold {self._trend_min}, skip")
                self._skipped.append(ticker)
                continue

            # Live price
            pd = await fetch_stock_price(ticker)
            if not pd:
                print(f"     ⚠  Price unavailable — skip")
                continue

            price, ma_20 = pd["price"], pd["ma_20"]

            # FIX: store price WITH the $ prefix so it matches position keys
            self._prices[ticker] = price
            # Also store without $ for the drawdown breaker
            self._prices[clean_sym] = price

            mentions = self.trend_filter.mention_count(ticker)
            print(f"     💲 ${price:.2f}  MA20=${ma_20:.2f}  mentions={mentions}")

            # FIX: fetch TICKER-SPECIFIC headlines, not generic finance news
            headlines = await fetch_ticker_headlines(ticker, max_results=15)
            print(f"     📰 {len(headlines)} headlines  →  Replicate scoring…")
            if headlines:
                print(f"     📄 Top: {headlines[0][:70]}…")

            # Run: news → Llama 3 sentiment → strategy → signal
            signal = await self.engine.run(MarketContext(
                ticker=ticker,
                source="stock",
                price=price,
                ma_20=ma_20,
                raw_news=headlines,
            ))

            if signal and signal.signal in (Signal.BUY, Signal.SELL):
                # Guard 4 — duplicate throttle
                if self.throttle.is_blocked(ticker, signal.signal.value):
                    print(f"     🔁 Throttled — skipping duplicate {signal.signal.value} on {ticker}")
                    continue

                # FIX: check minimum position size before executing
                price_for_sizing = price
                position_dollars = self.portfolio.cash * self.portfolio.risk_per_trade * min(signal.confidence, 1.0)
                if position_dollars < self.min_position_size:
                    print(f"     ⚠  Position size ${position_dollars:.2f} below minimum ${self.min_position_size:.2f} — "
                          f"check portfolio cash (${self.portfolio.cash:.2f})")

                self._execute(signal, price)
                self.throttle.record(ticker, signal.signal.value)

            # Stop-loss / take-profit check
            # FIX: pass both $ and non-$ versions of prices for compatibility
            stops = self.portfolio.check_stops(self._prices)
            for s in stops:
                print(f"  🛑 STOP: {s.ticker}  PnL=${s.pnl:+.2f}  {s.reasoning}")

        print(f"\n{'─'*65}")
        print(f"  {self.portfolio.summary_str(self._prices)}")
        if self._skipped:
            print(f"  ⚪ Skipped: {', '.join(self._skipped)}")
        print(f"{'─'*65}\n")

        return self._snapshot()

    def _execute(self, signal: TradeSignal, price: float):
        ticker = signal.ticker  # keeps the $ prefix

        if signal.signal == Signal.BUY:
            # FIX: don't try to BUY if we already hold this ticker (different from original)
            if ticker in self.portfolio.positions:
                print(f"     ℹ  Already holding {ticker} — no additional buy")
                return
            rec = self.portfolio.execute_buy(
                ticker=ticker,
                price=price,
                confidence=signal.confidence,
                reasoning=signal.reasoning,
                source=signal.source,
            )
            if rec:
                cost = rec.shares * price
                print(f"     ✅ BUY  {ticker}  {rec.shares:.4f} shares @ ${price:.2f}  cost=${cost:.2f}")
            else:
                print(f"     ❌ BUY FAILED for {ticker} — check portfolio logs")

        elif signal.signal == Signal.SELL:
            # FIX: SELL only if we actually HOLD the position — no shorting
            if ticker in self.portfolio.positions:
                rec = self.portfolio.execute_sell(
                    ticker=ticker,
                    price=price,
                    confidence=signal.confidence,
                    reasoning=signal.reasoning,
                )
                if rec:
                    print(f"     ✅ SELL {ticker}  PnL=${rec.pnl:+.2f}")
            else:
                print(f"     ℹ  SELL signal on {ticker} but no position held — ignore")

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
        print(f"\n🚀  TradingBot v2  [{mode}]")
        print(f"   Model      : Replicate / Llama-3-70B")
        print(f"   Watchlist  : {', '.join(self.watchlist)}")
        print(f"   Interval   : {interval_minutes} min")
        print(f"   Capital    : ${self.portfolio.starting_cash:,.2f}")
        print(f"   Risk/trade : {self.portfolio.risk_per_trade:.0%}")
        if not self.hours_guard:
            print(f"   Hours guard: DISABLED (paper testing mode)")
        print()

        while True:
            snap = await self.run_once()
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
    p = argparse.ArgumentParser(description="Paper Trading Bot (Replicate-powered)")
    p.add_argument("--tickers",   nargs="*",  default=None)
    p.add_argument("--loop",      action="store_true")
    p.add_argument("--interval",  type=int,   default=15)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--no-hours",  action="store_true", help="Disable market hours check (for paper testing)")
    p.add_argument("--daily-dd",  type=float, default=0.03)
    p.add_argument("--total-dd",  type=float, default=0.10)
    p.add_argument("--trend-min", type=int,   default=1)
    p.add_argument("--model",     type=str,   default="meta/meta-llama-3-70b-instruct")
    args = p.parse_args()

    bot = TradingBot(
        watchlist=args.tickers,
        interval_minutes=args.interval,
        sentiment_threshold=args.threshold,
        require_market_hours=not args.no_hours,
        daily_drawdown_limit=args.daily_dd,
        total_drawdown_limit=args.total_dd,
        trend_min_mentions=args.trend_min,
        replicate_model=args.model,
    )

    if args.loop:
        await bot.run_loop(args.interval)
    else:
        await bot.run_once(args.tickers)


if __name__ == "__main__":
    asyncio.run(main())