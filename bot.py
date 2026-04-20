"""
paper_trader/bot.py
────────────────────────────────────────────────────────────────────────────
PRODUCTION-READY trading bot — powered by Replicate (Llama 3 70B).

Guards active on every scan:
  1. MarketHoursGuard        — NYSE calendar, no weekend/holiday trades
  2. DrawdownCircuitBreaker  — halts at 3% daily / 10% total loss
  3. TrendingTickerFilter    — 13 free RSS feeds, only trade trending tickers
  4. DuplicateSignalThrottle — 60 min cooldown per ticker per signal

Deployment:
  Paper  →  python bot.py --loop
  Live   →  PRODUCTION=true  (swap PaperPortfolio → AlpacaBroker in __init__)

.env required:
  REPLICATE_API_TOKEN=r8_xxxx          ← required
  FINNHUB_API_KEY=                     ← optional (better stock headlines)
  NEWSAPI_KEY=                         ← optional (100 req/day extra coverage)
  PRODUCTION=false                     ← set true + add Alpaca keys for live
  ALPACA_API_KEY=
  ALPACA_SECRET_KEY=
  ALPACA_BASE_URL=https://paper-api.alpaca.markets
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

from strategy_engine.engine              import MarketContext, Signal, SignalBus, StrategyEngine, TradeSignal
from strategy_engine.sentiment_agent     import SentimentAgent
from strategy_engine.stock_strategy      import StockSentimentStrategy
from strategy_engine.polymarket_strategy import PolymarketSentimentStrategy
from strategy_engine.news_fetcher        import NicheFetcher

PRODUCTION = os.getenv("PRODUCTION", "false").lower() == "true"

DEFAULT_WATCHLIST = [
    "$NVDA","$TSLA","$AAPL","$MSFT","$AMZN",
    "$META","$GOOGL","$AMD","$PLTR","$COIN",
]


# ─────────────────────────────────────────────────────────────────────────────
# Price fetcher — Yahoo Finance v8 (free, no key)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_stock_price(ticker: str) -> Optional[Dict]:
    symbol = ticker.lstrip("$").upper()
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"interval":"1d","range":"1mo"},
                headers={"User-Agent":"Mozilla/5.0"},
            )
            r.raise_for_status()
            closes = r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            closes = [x for x in closes if x is not None]
            if not closes: return None
            price = closes[-1]
            ma_20 = sum(closes[-20:]) / len(closes[-20:]) if len(closes) >= 5 else price
            return {"price": round(price,2), "ma_20": round(ma_20,2)}
    except Exception as e:
        logger.warning("[Price] %s: %s", symbol, e); return None


# ─────────────────────────────────────────────────────────────────────────────
# Main Bot
# ─────────────────────────────────────────────────────────────────────────────

class TradingBot:
    """
    Full trading bot — paper or live.

    Sentiment engine:  Replicate  (meta/meta-llama-3-70b-instruct)
    News sources:      13 free RSS feeds + optional Finnhub/NewsAPI
    Guards:            Market hours, drawdown breaker, throttle, trending filter

    To go live:
        1. Set PRODUCTION=true in .env
        2. Add ALPACA_API_KEY + ALPACA_SECRET_KEY
        3. Replace  self.portfolio = PaperPortfolio()
           with     from broker_alpaca import AlpacaBroker
                    self.portfolio = AlpacaBroker()
    """

    def __init__(
        self,
        watchlist:            List[str] = None,
        interval_minutes:     int       = 15,
        sentiment_threshold: float = 0.10,
        require_market_hours: bool      = True,
        daily_drawdown_limit: float     = 0.03,
        total_drawdown_limit: float     = 0.10,
        signal_cooldown_min:  int       = 60,
        trend_min_mentions:   int       = 2,
        replicate_model:      str       = "meta/meta-llama-3-70b-instruct",
    ):
        self.watchlist = watchlist or DEFAULT_WATCHLIST
        self.interval  = interval_minutes * 60

        # ── Portfolio (swap for AlpacaBroker to go live) ──────────────────
        self.portfolio = PaperPortfolio()

        # ── Replicate sentiment agent ─────────────────────────────────────
        self.agent = SentimentAgent(model=replicate_model)

        # ── Strategy engine ───────────────────────────────────────────────
        self.bus    = SignalBus()
        self.engine = StrategyEngine(bus=self.bus, sentiment_agent=self.agent)
        self.engine.register(StockSentimentStrategy(
            sentiment_threshold=sentiment_threshold))
        self.engine.register(PolymarketSentimentStrategy(
            sentiment_threshold=sentiment_threshold))
        self.bus.subscribe(self._on_signal)

        # ── Guards ────────────────────────────────────────────────────────
        self.hours_guard  = MarketHoursGuard() if require_market_hours else None
        self.breaker      = DrawdownCircuitBreaker(
            self.portfolio,
            daily_limit=daily_drawdown_limit,
            total_limit=total_drawdown_limit,
        )
        self.throttle     = DuplicateSignalThrottle(cooldown_minutes=signal_cooldown_min)
        self.trend_filter = TrendingTickerFilter()
        self._trend_min   = trend_min_mentions

        self._prices:  Dict[str,float] = {}
        self._signals: List[dict]      = []
        self._skipped: List[str]       = []

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
        icon = {"BUY":"🟢","SELL":"🔴","HOLD":"🟡","SKIP":"⚪"}.get(sig.signal.value,"❓")
        print(f"  {icon} [{sig.signal.value:4s}]  {sig.ticker:<10}  "
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
            print(f"     Next open in {wait//3600}h {(wait%3600)//60}m\n")
            return self._snapshot(blocked=self.hours_guard.reason)

        # Guard 2 — drawdown circuit breaker
        if self.breaker.is_tripped(self._prices):
            print(f"\n  🚨  {self.breaker.reason}\n")
            return self._snapshot(blocked=self.breaker.reason)

        print(f"  🛡  {self.breaker.reason}")

        # Guard 3 — refresh trending RSS feeds (cached 15 min)
        await self.trend_filter.refresh()
        top = self.trend_filter.top_trending(6)
        print(f"  📈  Trending: {', '.join(f'{s}({c})' for s,c in top)}\n")

        self._skipped = []

        for ticker in tickers:
            print(f"  ▶  {ticker}")

            # Trending check
            if not self.trend_filter.is_trending(ticker, self._trend_min):
                m = self.trend_filter.mention_count(ticker)
                print(f"     ⚪ Only {m} mention(s) — not trending, skip")
                self._skipped.append(ticker); continue

            # Live price
            pd = await fetch_stock_price(ticker)
            if not pd:
                print(f"     ⚠  Price unavailable — skip"); continue

            price, ma_20 = pd["price"], pd["ma_20"]
            sym = ticker.lstrip("$").upper()
            self._prices[sym] = price
            m = self.trend_filter.mention_count(ticker)
            print(f"     💲 ${price:.2f}  MA20=${ma_20:.2f}  mentions={m}")

            # Fetch headlines (composite of all free sources)
            fetcher   = NicheFetcher(niche="finance")
            headlines = await fetcher.fetch(ticker, max_results=15)
            print(f"     📰 {len(headlines)} headlines  →  Replicate scoring…")

            # Run: news → Llama 3 sentiment → strategy → signal
            signal = await self.engine.run(MarketContext(
                ticker=ticker, source="stock",
                price=price, ma_20=ma_20,
                raw_news=headlines,
            ))

            if signal and signal.signal in (Signal.BUY, Signal.SELL):
                # Guard 4 — duplicate throttle
                if self.throttle.is_blocked(ticker, signal.signal.value):
                    print(f"     🔁 Throttled — skipping duplicate {signal.signal.value}")
                    continue
                self._execute(signal, price)
                self.throttle.record(ticker, signal.signal.value)

            # Stop-loss / take-profit check
            for s in self.portfolio.check_stops(self._prices):
                print(f"  🛑 STOP: {s.ticker}  PnL=${s.pnl:+.2f}  {s.reasoning}")

        print(f"\n{'─'*65}")
        print(f"  {self.portfolio.summary_str(self._prices)}")
        if self._skipped:
            print(f"  ⚪ Skipped (not trending): {', '.join(self._skipped)}")
        print(f"{'─'*65}\n")

        return self._snapshot()

    def _execute(self, signal: TradeSignal, price: float):
        if signal.signal == Signal.BUY:
            self.portfolio.execute_buy(
                ticker=signal.ticker, price=price,
                confidence=signal.confidence,
                reasoning=signal.reasoning, source=signal.source)
        elif signal.signal == Signal.SELL:
            self.portfolio.execute_sell(
                ticker=signal.ticker, price=price,
                confidence=signal.confidence, reasoning=signal.reasoning)

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
        print(f"\n🚀  TradingBot  [{mode}]")
        print(f"   Model      : Replicate / Llama-3-70B")
        print(f"   Watchlist  : {', '.join(self.watchlist)}")
        print(f"   Interval   : {interval_minutes} min")
        print(f"   Capital    : ${self.portfolio.starting_cash:,.2f}\n")

        while True:
            snap = await self.run_once()
            self.portfolio.save("portfolio.json")

            blocked = snap.get("blocked","")
            if blocked and "Market" in blocked and self.hours_guard:
                wait  = self.hours_guard.seconds_until_open()
                sleep = min(wait, interval_minutes * 60)
                print(f"  💤 Sleeping {sleep//60} min until market opens…\n")
                await asyncio.sleep(sleep)
            else:
                print(f"  ⏱  Next scan in {interval_minutes} min…\n")
                await asyncio.sleep(interval_minutes * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    import argparse
    p = argparse.ArgumentParser(description="Production Trading Bot (Replicate-powered)")
    p.add_argument("--tickers",    nargs="*",  default=None,
                   help="e.g. --tickers '$NVDA' '$TSLA'")
    p.add_argument("--loop",       action="store_true",   help="Run continuously")
    p.add_argument("--interval",   type=int,   default=15, help="Minutes between scans")
    p.add_argument("--threshold",  type=float, default=0.30)
    p.add_argument("--no-hours",   action="store_true",   help="Disable market hours guard")
    p.add_argument("--daily-dd",   type=float, default=0.03)
    p.add_argument("--total-dd",   type=float, default=0.10)
    p.add_argument("--trend-min",  type=int,   default=2)
    p.add_argument("--model",      type=str,
                   default="meta/meta-llama-3-70b-instruct",
                   help="Replicate model ID")
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
