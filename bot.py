"""
paper_trader/bot.py  — v4  PRODUCTION-READY
────────────────────────────────────────────────────────────────────────────
FIXES vs v3:
  1. CRITICAL BUG FIX: run_once() called non-existent `fetch_enriched_market_data`
     (a synchronous stub that was never defined). Now correctly calls
     `await fetch_alpaca_enriched_data()` — the real Alpaca-native async function.

  2. CRITICAL BUG FIX: _execute() was defined but NEVER CALLED inside run_once().
     Every scan generated signals that were logged but never turned into orders.
     Now wired up: signal → _execute() → portfolio.execute_buy/sell().

  3. CRITICAL BUG FIX: run_once() had no return statement at the end.
     The dashboard received None on every scan after guards passed.
     Added `return self._snapshot()` at the end of the ticker loop.

  4. CRITICAL BUG FIX: check_stops() was never called in the scan cycle.
     Stop-losses and take-profits were completely inactive. Now called at the
     start of each scan (after breaker check) so positions are protected.

  5. REMOVED: `import yfinance as yf` — Yahoo Finance is fully replaced by
     Alpaca Data API v2. yfinance was causing cloud IP bans and silent failures.

  6. PNL FIX: Wired self._execute() to also use the throttle guard so we
     don't re-enter a position we just sold within the cooldown window.

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
import pandas as pd
import numpy as np
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
    from engine          import MarketContext, Signal, SignalBus, StrategyEngine, TradeSignal
    from sentiment_agent import SentimentAgent
    from stock_strategy  import StockSentimentStrategy
    from news_fetcher    import GoogleNewsRSSFetcher

# Optional Polymarket strategy
try:
    try:
        from strategy_engine.polymarket_strategy import PolymarketSentimentStrategy
    except ImportError:
        from polymarket_strategy import PolymarketSentimentStrategy
    _HAS_POLYMARKET = True
except ImportError:
    _HAS_POLYMARKET = False
    logger.warning("[Bot] polymarket_strategy not found — Polymarket signals disabled")

PRODUCTION       = os.getenv("PRODUCTION",      "false").lower() == "true"
USE_ALPACA_PAPER = os.getenv("USE_ALPACA_PAPER", "false").lower() == "true"
USE_ALPACA       = PRODUCTION or USE_ALPACA_PAPER
ALPACA_MCP_URL   = os.getenv("ALPACA_MCP_URL", "http://localhost:3000")

DEFAULT_WATCHLIST = [
    "$NVDA","$TSLA","$AAPL","$MSFT","$AMZN",
    "$META","$GOOGL","$AMD","$PLTR","$COIN",
]


# ─────────────────────────────────────────────────────────────────────────────
# Price + Technical Analysis — Alpaca Data API v2 (primary, no IP bans)
# yfinance has been fully removed. Alpaca Data is free with any paper account
# and uses the same API keys already required for trading.
# ─────────────────────────────────────────────────────────────────────────────

_ALPACA_DATA_BASE = "https://data.alpaca.markets"


async def fetch_alpaca_enriched_data(ticker_symbol: str) -> dict:
    """
    Fetches 60 days of OHLCV bars from Alpaca Data API v2 and computes:
      - price  (latest close)
      - ma_20  (20-day simple moving average)
      - volume (latest volume)
      - vol_sma20 (20-day average volume)
      - atr    (14-day Average True Range)
      - adx    (14-day Average Directional Index)

    Returns {} on failure — caller should skip the ticker gracefully.
    """
    clean_symbol = ticker_symbol.lstrip("$").upper()
    key    = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")

    if not key or not secret:
        logger.error("[Alpaca Data] ALPACA_API_KEY / ALPACA_SECRET_KEY not set.")
        return {}

    headers = {
        "APCA-API-KEY-ID":     key,
        "APCA-API-SECRET-KEY": secret,
        "Accept":              "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                f"{_ALPACA_DATA_BASE}/v2/stocks/{clean_symbol}/bars",
                headers=headers,
                params={"timeframe": "1Day", "limit": 60, "adjustment": "all"},
            )
            r.raise_for_status()
            bars = r.json().get("bars", [])

            if not bars or len(bars) < 20:
                logger.warning("[Alpaca Data] %s — only %d bars returned (need ≥20)",
                               clean_symbol, len(bars))
                return {}

            df = pd.DataFrame(bars)
            df = df.rename(columns={"c": "Close", "h": "High", "l": "Low", "v": "Volume"})

            # ── Moving averages ─────────────────────────────────────────────
            df["MA20"]     = df["Close"].rolling(window=20).mean()
            df["Vol_SMA20"] = df["Volume"].rolling(window=20).mean()

            # ── ATR (14-day) ────────────────────────────────────────────────
            df["TR"] = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift(1)).abs(),
                (df["Low"]  - df["Close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            df["ATR"] = df["TR"].rolling(window=14).mean()

            # ── ADX (14-day) ────────────────────────────────────────────────
            up   = df["High"] - df["High"].shift(1)
            down = df["Low"].shift(1) - df["Low"]
            df["+DM"]   = np.where((up > down) & (up > 0),     up,   0.0)
            df["-DM"]   = np.where((down > up) & (down > 0),   down, 0.0)
            df["TR14"]  = df["TR"].ewm(span=14, adjust=False).mean()
            df["+DI14"] = 100 * (df["+DM"].ewm(span=14, adjust=False).mean() / df["TR14"])
            df["-DI14"] = 100 * (df["-DM"].ewm(span=14, adjust=False).mean() / df["TR14"])
            di_sum      = df["+DI14"] + df["-DI14"]
            df["DX"]    = 100 * ((df["+DI14"] - df["-DI14"]).abs() / di_sum.replace(0, 1))
            df["ADX"]   = df["DX"].ewm(span=14, adjust=False).mean()

            latest = df.iloc[-1]
            result = {
                "price":     float(latest["Close"]),
                "ma_20":     float(latest["MA20"]),
                "volume":    float(latest["Volume"]),
                "vol_sma20": float(latest["Vol_SMA20"]),
                "atr":       float(latest["ATR"]),
                "adx":       float(latest["ADX"]),
            }
            logger.debug("[Alpaca Data] %s — $%.2f  MA20=$%.2f  ADX=%.1f  ATR=%.2f",
                         clean_symbol, result["price"], result["ma_20"],
                         result["adx"], result["atr"])
            return result

    except httpx.HTTPStatusError as e:
        logger.error("[Alpaca Data] HTTP %d for %s: %s",
                     e.response.status_code, clean_symbol, e)
    except Exception as e:
        logger.error("[Alpaca Data] Error fetching TA data for %s: %s", clean_symbol, e)
    return {}


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
        sentiment_threshold:  float     = 0.15,
        confidence_threshold: float     = 0.35,
        require_market_hours: bool      = None,   # None = auto (False=paper, True=prod)
        daily_drawdown_limit: float     = 0.03,
        total_drawdown_limit: float     = 0.10,
        signal_cooldown_min:  int       = 30,
        trend_min_mentions:   int       = 0,      # 0 = filter disabled
        replicate_model:      str       = "meta/meta-llama-3-70b-instruct",
        min_position_size:    float     = 50.0,
        use_alpaca_mcp:       bool      = True,
        portfolio_path:       str       = "portfolio.json",
    ):
        self.watchlist         = watchlist or DEFAULT_WATCHLIST
        self.interval          = interval_minutes * 60
        self.min_position_size = min_position_size
        self.use_alpaca_mcp    = use_alpaca_mcp
        self._trend_min        = trend_min_mentions
        self._portfolio_path   = portfolio_path

        # Auto-derive market hours enforcement from PRODUCTION flag only
        if require_market_hours is None:
            require_market_hours = USE_ALPACA

        # Portfolio — use AlpacaBroker when either PRODUCTION or USE_ALPACA_PAPER is set
        if USE_ALPACA:
            logger.info("[Bot] %s — using AlpacaBroker (%s)",
                        "🔴 PRODUCTION" if PRODUCTION else "📊 ALPACA PAPER",
                        os.getenv("ALPACA_BASE_URL", "paper"))
            try:
                from broker_alpaca import AlpacaBroker
                self.portfolio = AlpacaBroker()
            except Exception as e:
                logger.error("[Bot] AlpacaBroker init failed (%s) — falling back to paper", e)
                self.portfolio = PaperPortfolio.load(portfolio_path)
        else:
            logger.info("[Bot] 📄 PAPER mode — loading portfolio from %s", portfolio_path)
            self.portfolio = PaperPortfolio.load(portfolio_path)

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

    # ── Signal handler ─────────────────────────────────────────────────────

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

    # ── Scan cycle ─────────────────────────────────────────────────────────

    async def run_once(self, tickers: List[str] = None) -> dict:
        tickers = tickers or self.watchlist

        print(f"\n{'═'*70}")
        print(f"  📡  Scan  {time.strftime('%Y-%m-%d  %H:%M:%S')}"
              f"  [{'PRODUCTION 🔴' if PRODUCTION else 'ALPACA PAPER 📊' if USE_ALPACA_PAPER else 'PAPER 📄'}]")
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

        # FIX #4 — Check stops/take-profits on existing positions before scanning
        # Previously check_stops() was never called — stop-losses were dead code.
        if self._prices:
            stops_closed = self.portfolio.check_stops(self._prices)
            for rec in stops_closed:
                pnl_icon = "🟢" if rec.pnl >= 0 else "🔴"
                print(f"  {pnl_icon} STOP/TP  {rec.ticker}  PnL=${rec.pnl:+.2f}")

        # Guard 3 — trending feed refresh
        await self.trend_filter.refresh()
        top = self.trend_filter.top_trending(6)
        if top:
            print(f"  📈  Trending: {', '.join(f'{s}({c})' for s, c in top)}\n")
        else:
            print(f"  📈  Trending: (feed empty — scanning all tickers)\n")

        self._skipped  = []
        self._signals  = []   # reset per scan so dashboard shows only latest cycle

        for ticker in tickers:
            print(f"\n  ▶  {ticker}")

            # Trend gate (disabled when _trend_min == 0)
            if self._trend_min > 0 and not self.trend_filter.is_trending(ticker, self._trend_min):
                mentions = self.trend_filter.mention_count(ticker)
                print(f"     ⚪ {mentions} mention(s) < threshold {self._trend_min} → skip")
                self._skipped.append(ticker)
                continue

            # FIX #1 — was calling non-existent `fetch_enriched_market_data(ticker)`
            # (a synchronous stub that was never defined anywhere in the codebase).
            # Now correctly calls the real async Alpaca function.
            ta_data = await fetch_alpaca_enriched_data(ticker)
            if not ta_data or ta_data.get("price") is None:
                print(f"     ⚠  Price/TA data unavailable → skip")
                self._skipped.append(ticker)
                continue

            price = ta_data["price"]
            ma_20 = ta_data["ma_20"]
            clean_sym = ticker.lstrip("$").upper()
            self._prices[ticker]    = price
            self._prices[clean_sym] = price
            print(f"     💲 ${price:.2f}  MA20=${ma_20:.2f}  "
                  f"ADX={ta_data['adx']:.1f}  ATR={ta_data['atr']:.2f}  "
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

            source_type = "polymarket" if not ticker.startswith("$") else "stock"

            # Strategy evaluation
            signal = await self.engine.run(MarketContext(
                ticker=ticker,
                source=source_type,
                price=price,
                ma_20=ma_20,
                adx=ta_data["adx"],
                atr=ta_data["atr"],
                volume=ta_data["volume"],
                vol_sma20=ta_data["vol_sma20"],
                raw_news=enriched_news,
            ))

            # FIX #2 — _execute() was defined but NEVER CALLED.
            # Every scan produced signals that were logged to self._signals
            # but never forwarded to the portfolio/broker. No orders were sent.
            if signal and signal.signal not in (Signal.HOLD, Signal.SKIP):
                # Check throttle before executing to avoid rapid re-entries
                if self.throttle.is_blocked(ticker, signal.signal.value):
                    print(f"     ⏳ Throttled — cooldown active for {ticker} {signal.signal.value}")
                else:
                    self._execute(signal, price)
                    self.throttle.record(ticker, signal.signal.value)

        # FIX #3 — run_once() had no return statement; callers received None.
        # The Flask dashboard, run_bot.py, and the loop all depend on this dict.
        return self._snapshot()

    # ── Order execution ────────────────────────────────────────────────────

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
                shares = getattr(rec, "shares", "?")
                print(f"     ✅ BUY  {ticker}  {shares} sh @ ${price:.2f}")
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
                    pnl = getattr(rec, "pnl", 0)
                    icon = "🟢" if pnl >= 0 else "🔴"
                    print(f"     {icon} SELL {ticker}  PnL=${pnl:+.2f}")
            else:
                print(f"     ℹ  SELL signal — no position in {ticker} (no short selling)")

    # ── Snapshot ───────────────────────────────────────────────────────────

    def _snapshot(self, blocked: str = "") -> dict:
        snap = self.portfolio.snapshot(self._prices)
        snap["signals"]  = self._signals[-50:]
        snap["skipped"]  = self._skipped
        snap["blocked"]  = blocked
        snap["trending"] = dict(self.trend_filter._counts)
        return snap

    # ── Continuous loop ────────────────────────────────────────────────────

    async def run_loop(self, interval_minutes: int = 15):
        mode  = "PRODUCTION 🔴 LIVE" if PRODUCTION else ("ALPACA PAPER 📊" if USE_ALPACA_PAPER else "PAPER 📄")
        strat = self.engine._strategies[0] if self.engine._strategies else None
        print(f"\n🚀  TradingBot v4  [{mode}]")
        print(f"   Model         : Replicate / Llama-3-70B")
        print(f"   Watchlist     : {', '.join(self.watchlist)}")
        print(f"   Interval      : {interval_minutes} min")
        if hasattr(self.portfolio, "starting_cash"):
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
            if hasattr(self.portfolio, "save"):
                self.portfolio.save(self._portfolio_path)

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
    p = argparse.ArgumentParser(description="Trading Bot v4")
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