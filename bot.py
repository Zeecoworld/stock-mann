"""
bot.py  — v5  PRODUCTION-READY
─────────────────────────────────────────────────────────────────────────────
CHANGES vs v4:
  1. 15Min bars  -- intraday TA instead of yesterday's daily close
  2. SPY regime  -- BUY suppressed when SPY < its 15-min MA20
  3. Parallel sentiment -- enrich_batch() cuts 10-ticker scan from ~50s to ~8s
  4. Startup reconciliation -- existing Alpaca positions synced on __init__
  5. Smart loop errors -- network=30s retry, logic=5m back-off
  6. ATR forwarded to broker for accurate bracket sizing
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

try:
    try:
        from strategy_engine.polymarket_strategy import PolymarketSentimentStrategy
    except ImportError:
        from polymarket_strategy import PolymarketSentimentStrategy
    _HAS_POLYMARKET = True
except ImportError:
    _HAS_POLYMARKET = False

PRODUCTION       = os.getenv("PRODUCTION",      "false").lower() == "true"
USE_ALPACA_PAPER = os.getenv("USE_ALPACA_PAPER", "false").lower() == "true"
USE_ALPACA       = PRODUCTION or USE_ALPACA_PAPER
ALPACA_MCP_URL   = os.getenv("ALPACA_MCP_URL", "http://localhost:3000")

DEFAULT_WATCHLIST = [
    "$NVDA","$TSLA","$AAPL","$MSFT","$AMZN",
    "$META","$GOOGL","$AMD","$PLTR","$COIN",
]

_ALPACA_DATA_BASE = "https://data.alpaca.markets"


# -- FIX 1: 15-minute bars ----------------------------------------------------

async def fetch_alpaca_enriched_data(ticker_symbol: str) -> dict:
    clean  = ticker_symbol.lstrip("$").upper()
    key    = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        logger.error("[Alpaca Data] Credentials not set"); return {}

    headers = {
        "APCA-API-KEY-ID":     key,
        "APCA-API-SECRET-KEY": secret,
        "Accept":              "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                f"{_ALPACA_DATA_BASE}/v2/stocks/{clean}/bars",
                headers=headers,
                params={"timeframe": "15Min", "limit": 120, "adjustment": "all"},
            )
            r.raise_for_status()
            bars = r.json().get("bars", [])

        if not bars or len(bars) < 20:
            logger.warning("[Alpaca Data] %s -- %d bars (need >=20)", clean, len(bars))
            return {}

        df = pd.DataFrame(bars)
        df = df.rename(columns={"c": "Close", "h": "High", "l": "Low", "v": "Volume"})
        df["MA20"]      = df["Close"].rolling(20).mean()
        df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
        df["TR"] = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"]  - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["ATR"] = df["TR"].rolling(14).mean()
        up   = df["High"] - df["High"].shift(1)
        down = df["Low"].shift(1) - df["Low"]
        df["+DM"]   = np.where((up > down) & (up > 0),   up,   0.0)
        df["-DM"]   = np.where((down > up) & (down > 0), down, 0.0)
        df["TR14"]  = df["TR"].ewm(span=14, adjust=False).mean()
        df["+DI14"] = 100 * (df["+DM"].ewm(span=14, adjust=False).mean() / df["TR14"])
        df["-DI14"] = 100 * (df["-DM"].ewm(span=14, adjust=False).mean() / df["TR14"])
        di_sum      = df["+DI14"] + df["-DI14"]
        df["DX"]    = 100 * ((df["+DI14"] - df["-DI14"]).abs() / di_sum.replace(0, 1))
        df["ADX"]   = df["DX"].ewm(span=14, adjust=False).mean()
        latest = df.iloc[-1]
        return {
            "price":     float(latest["Close"]),
            "ma_20":     float(latest["MA20"]),
            "volume":    float(latest["Volume"]),
            "vol_sma20": float(latest["Vol_SMA20"]),
            "atr":       float(latest["ATR"]),
            "adx":       float(latest["ADX"]),
        }
    except httpx.HTTPStatusError as e:
        logger.error("[Alpaca Data] HTTP %d %s: %s", e.response.status_code, clean, e)
    except Exception as e:
        logger.error("[Alpaca Data] %s: %s", clean, e)
    return {}


# -- FIX 2: SPY regime filter -------------------------------------------------

async def fetch_market_regime() -> bool:
    """True = SPY above 15-min MA20 -> BUY allowed. False -> BUY suppressed."""
    spy = await fetch_alpaca_enriched_data("SPY")
    if not spy:
        logger.warning("[Regime] SPY unavailable -- defaulting bullish"); return True
    bullish = spy["price"] > spy["ma_20"]
    logger.info("[Regime] SPY $%.2f vs MA20 $%.2f -> %s",
                spy["price"], spy["ma_20"],
                "BULLISH" if bullish else "BEARISH -- BUY suppressed")
    return bullish


async def fetch_alpaca_mcp_context(ticker: str) -> str:
    symbol = ticker.lstrip("$").upper()
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r_acct = await c.post(f"{ALPACA_MCP_URL}/tools/call",
                                  json={"name": "get_account", "arguments": {}},
                                  headers={"Content-Type": "application/json"})
            acct = r_acct.json().get("content", [{}])[0].get("text", "n/a")
            r_pos = await c.post(f"{ALPACA_MCP_URL}/tools/call",
                                 json={"name": "get_positions", "arguments": {}},
                                 headers={"Content-Type": "application/json"})
            pos = r_pos.json().get("content", [{}])[0].get("text", "n/a")
        return (f"\n\n=== LIVE ALPACA BROKERAGE CONTEXT ===\n"
                f"Account: {acct}\nPositions: {pos}\nTicker: {symbol}\n"
                f"======================================\n")
    except Exception as e:
        logger.debug("[MCP] Skipped: %s", e); return ""


async def fetch_ticker_headlines(ticker: str, max_results: int = 15) -> List[str]:
    clean   = ticker.lstrip("$").upper()
    queries = [clean, f"{clean} stock", f"{clean} earnings"]
    seen: set        = set()
    all_h: List[str] = []
    results = await asyncio.gather(
        *[GoogleNewsRSSFetcher().fetch(q, max_results=8) for q in queries],
        return_exceptions=True)
    for batch in results:
        if isinstance(batch, Exception): continue
        for h in batch:
            h = h.strip()
            if h and h not in seen:
                seen.add(h); all_h.append(h)
    relevant = [h for h in all_h if clean.lower() in h.lower()]
    fallback  = [h for h in all_h if h not in relevant]
    return (relevant + fallback)[:max_results]


# -- TradingBot ----------------------------------------------------------------

class TradingBot:
    def __init__(
        self,
        watchlist:            List[str] = None,
        interval_minutes:     int       = 15,
        sentiment_threshold:  float     = 0.15,
        confidence_threshold: float     = 0.35,
        require_market_hours: bool      = None,
        daily_drawdown_limit: float     = 0.03,
        total_drawdown_limit: float     = 0.10,
        signal_cooldown_min:  int       = 30,
        trend_min_mentions:   int       = 0,
        replicate_model:      str       = "meta/llama-3.3-70b-instruct",
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

        if require_market_hours is None:
            require_market_hours = USE_ALPACA

        if USE_ALPACA:
            logger.info("[Bot] %s", "PRODUCTION" if PRODUCTION else "ALPACA PAPER")
            try:
                from broker_alpaca import AlpacaBroker
                self.portfolio = AlpacaBroker()
            except Exception as e:
                logger.error("[Bot] AlpacaBroker failed (%s) -- fallback paper", e)
                self.portfolio = PaperPortfolio.load(portfolio_path)
        else:
            self.portfolio = PaperPortfolio.load(portfolio_path)

        self.agent  = SentimentAgent(model=replicate_model)
        self.bus    = SignalBus()
        self.engine = StrategyEngine(bus=self.bus, sentiment_agent=None)
        self.engine.register(StockSentimentStrategy(
            sentiment_threshold=sentiment_threshold,
            confidence_threshold=confidence_threshold))
        if _HAS_POLYMARKET:
            try:
                self.engine.register(PolymarketSentimentStrategy(
                    sentiment_threshold=sentiment_threshold))
            except Exception: pass

        self.bus.subscribe(self._on_signal)
        self.hours_guard  = MarketHoursGuard() if require_market_hours else None
        self.breaker      = DrawdownCircuitBreaker(
            self.portfolio,
            daily_limit=daily_drawdown_limit,
            total_limit=total_drawdown_limit)
        self.throttle     = DuplicateSignalThrottle(cooldown_minutes=signal_cooldown_min)
        self.trend_filter = TrendingTickerFilter()

        self._prices:  Dict[str, float] = {}
        self._signals: List[dict]       = []
        self._skipped: List[str]        = []

        # FIX 4 -- reconcile existing positions on startup
        self._reconcile_positions()

    def _reconcile_positions(self):
        if not USE_ALPACA or not hasattr(self.portfolio, "client"):
            return
        try:
            positions = self.portfolio.client.get_all_positions()
            for p in positions:
                price = float(p.current_price or p.avg_entry_price or 0)
                if price > 0:
                    self._prices[p.symbol]       = price
                    self._prices[f"${p.symbol}"] = price
            if positions:
                logger.info("[Bot] Reconciled %d positions: %s", len(positions),
                            ", ".join(f"{p.symbol}=${float(p.current_price or 0):.2f}"
                                      for p in positions))
        except Exception as e:
            logger.warning("[Bot] Reconciliation failed: %s", e)

    def _on_signal(self, sig: TradeSignal):
        self._signals.append({
            "time":       time.strftime("%H:%M:%S"),
            "ticker":     sig.ticker,
            "signal":     sig.signal.value,
            "confidence": round(sig.confidence, 3),
            "reasoning":  sig.reasoning,
            "source":     sig.source,
        })
        icon = {"BUY": "BUY+", "SELL": "SELL", "HOLD": "HOLD", "SKIP": "SKIP"}.get(
            sig.signal.value, "?")
        print(f"  [{icon}]  {sig.ticker:<12}  conf={sig.confidence:.0%}  "
              f"|  {sig.reasoning[:100]}")

    async def run_once(self, tickers: List[str] = None) -> dict:
        tickers = tickers or self.watchlist
        print(f"\n{'='*70}")
        print(f"  Scan  {time.strftime('%Y-%m-%d  %H:%M:%S')}  "
              f"[{'PROD' if PRODUCTION else 'PAPER' if USE_ALPACA_PAPER else 'LOCAL'}]")
        print(f"{'='*70}")

        if self.hours_guard and not self.hours_guard.is_open():
            wait = self.hours_guard.seconds_until_open()
            print(f"  PAUSED  {self.hours_guard.reason}  "
                  f"(opens in {wait//3600}h {(wait%3600)//60}m)")
            return self._snapshot(blocked=self.hours_guard.reason)

        prices_for_guard = {k.lstrip("$"): v for k, v in self._prices.items()}
        if self.breaker.is_tripped(prices_for_guard):
            print(f"  {self.breaker.reason}")
            return self._snapshot(blocked=self.breaker.reason)
        print(f"  GUARD  {self.breaker.reason}")

        if self._prices:
            for rec in self.portfolio.check_stops(self._prices):
                icon = "+" if rec.pnl >= 0 else "-"
                print(f"  [{icon}] STOP/TP  {rec.ticker}  PnL=${rec.pnl:+.2f}")

        # FIX 2 + trending refresh -- run concurrently
        bullish_regime, _ = await asyncio.gather(
            fetch_market_regime(), self.trend_filter.refresh())
        top = self.trend_filter.top_trending(6)
        print(f"  REGIME  {'Bullish' if bullish_regime else 'Bearish -- BUY suppressed'}")
        if top:
            print(f"  TREND   {', '.join(f'{s}({c})' for s,c in top)}\n")

        self._skipped = []
        self._signals = []

        # Phase 1 -- fetch all TA + news in parallel
        ta_results, news_results = await asyncio.gather(
            asyncio.gather(*[fetch_alpaca_enriched_data(t) for t in tickers]),
            asyncio.gather(*[fetch_ticker_headlines(t, 15) for t in tickers]),
        )

        # Phase 2 -- build contexts
        contexts: List[MarketContext] = []
        valid_tickers: List[str]      = []
        for ticker, ta, headlines in zip(tickers, ta_results, news_results):
            if self._trend_min > 0 and not self.trend_filter.is_trending(ticker, self._trend_min):
                self._skipped.append(ticker); continue
            if not ta or ta.get("price") is None:
                print(f"  SKIP  {ticker} -- no data"); self._skipped.append(ticker); continue

            price = ta["price"]
            self._prices[ticker]                     = price
            self._prices[ticker.lstrip("$").upper()] = price

            mcp = ""
            if self.use_alpaca_mcp:
                try:
                    mcp = await asyncio.wait_for(
                        fetch_alpaca_mcp_context(ticker), timeout=5)
                except asyncio.TimeoutError: pass

            print(f"  >> {ticker:<10} ${price:.2f}  MA20=${ta['ma_20']:.2f}  "
                  f"ADX={ta['adx']:.1f}  ATR={ta['atr']:.2f}  "
                  f"news={len(headlines)}{'  MCP' if mcp else ''}")

            src = "polymarket" if not ticker.startswith("$") else "stock"
            contexts.append(MarketContext(
                ticker=ticker, source=src, price=price, ma_20=ta["ma_20"],
                adx=ta["adx"], atr=ta["atr"], volume=ta["volume"],
                vol_sma20=ta["vol_sma20"],
                raw_news=headlines + ([mcp] if mcp else []),
            ))
            valid_tickers.append(ticker)

        if not contexts:
            return self._snapshot()

        # FIX 3 -- Phase 3: parallel sentiment scoring
        print(f"\n  Scoring {len(contexts)} tickers concurrently via Replicate...")
        enriched = await self.agent.enrich_batch(contexts)

        # Phase 4 -- strategy + execution
        print()
        for ctx, ticker in zip(enriched, valid_tickers):
            signal = await self.engine.run(ctx)
            if not signal or signal.signal in (Signal.HOLD, Signal.SKIP):
                continue
            if signal.signal == Signal.BUY and not bullish_regime:
                print(f"  SKIP BUY {ticker} -- bearish regime"); continue
            if self.throttle.is_blocked(ticker, signal.signal.value):
                print(f"  THROTTLED {ticker}"); continue
            self._execute(signal, self._prices[ticker], atr=ctx.atr or 0.0)
            self.throttle.record(ticker, signal.signal.value)

        return self._snapshot()

    def _execute(self, signal: TradeSignal, price: float, atr: float = 0.0):
        ticker = signal.ticker
        if signal.signal == Signal.BUY:
            if self.portfolio.position_exists(ticker):
                print(f"     INFO  Already holding {ticker}"); return
            kwargs = {"atr": atr} if hasattr(self.portfolio, "_calc_notional") else {}
            rec = self.portfolio.execute_buy(
                ticker=ticker, price=price, conf=signal.confidence,
                reason=signal.reasoning, source=signal.source, **kwargs)
            if rec: print(f"     BUY  {ticker} @ ${price:.2f}")
            else:   print(f"     BUY FAILED  {ticker}")
        elif signal.signal == Signal.SELL:
            if self.portfolio.position_exists(ticker):
                rec = self.portfolio.execute_sell(
                    ticker=ticker, price=price, confidence=signal.confidence,
                    reasoning=signal.reasoning)
                if rec:
                    pnl = getattr(rec, "pnl", 0)
                    print(f"     SELL {ticker}  PnL=${pnl:+.2f}")
            else:
                print(f"     INFO  SELL {ticker} -- no position")

    def _snapshot(self, blocked: str = "") -> dict:
        snap = self.portfolio.snapshot(self._prices)
        snap["signals"]  = self._signals[-50:]
        snap["skipped"]  = self._skipped
        snap["blocked"]  = blocked
        snap["trending"] = dict(self.trend_filter._counts)
        return snap

    async def run_loop(self, interval_minutes: int = 15):
        mode = "PROD" if PRODUCTION else "ALPACA PAPER" if USE_ALPACA_PAPER else "LOCAL PAPER"
        print(f"\nTradingBot v5  [{mode}]  bars=15Min  regime=SPY_MA20\n")
        while True:
            try:
                snap = await self.run_once()
                if hasattr(self.portfolio, "save"):
                    self.portfolio.save(self._portfolio_path)
                blocked = snap.get("blocked", "")
                if blocked and "Market" in blocked and self.hours_guard:
                    sleep = min(self.hours_guard.seconds_until_open(),
                                interval_minutes * 60)
                    print(f"  Sleeping {sleep//60}m until market opens...\n")
                else:
                    sleep = interval_minutes * 60
                    print(f"  Next scan in {interval_minutes}m...\n")
                for _ in range(int(sleep)):
                    await asyncio.sleep(1)

            # FIX 5 -- smart error handling
            except (httpx.ConnectError, httpx.TimeoutException, ConnectionError) as e:
                logger.warning("[Bot] Network error -- retry in 30s: %s", e)
                await asyncio.sleep(30)
            except Exception as e:
                logger.error("[Bot] Scan error -- backing off 5m: %s", e, exc_info=True)
                await asyncio.sleep(300)


async def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tickers",  nargs="*",  default=None)
    p.add_argument("--loop",     action="store_true")
    p.add_argument("--interval", type=int,   default=15)
    p.add_argument("--no-hours", action="store_true")
    p.add_argument("--no-mcp",   action="store_true")
    args = p.parse_args()
    bot = TradingBot(
        watchlist=args.tickers,
        interval_minutes=args.interval,
        require_market_hours=False if args.no_hours else None,
        use_alpaca_mcp=not args.no_mcp,
    )
    if args.loop: await bot.run_loop(args.interval)
    else:         await bot.run_once(args.tickers)

if __name__ == "__main__":
    asyncio.run(main())