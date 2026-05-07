"""
paper_trader_flask/bot_core.py
──────────────────────────────
Re-exports TradingBot + helpers from bot.py so Flask (app.py) can import
them without touching __main__ guards.

FIX: The except block previously only caught ImportError. If bot.py imports
fine but a *dependency* (pandas, numpy, httpx) is missing, Python raises
ModuleNotFoundError (a subclass of ImportError — caught) but also other
errors like AttributeError or ImportError from transitive deps. Widened to
catch Exception so the Flask app always boots and the real error is logged.

FIX: The stub TradingBot was missing attributes that app.py accesses
directly on the bot object:
  - _portfolio_path  → caused AttributeError in app.py line 143
  - breaker          → caused AttributeError in /api/status
  - trend_filter     → caused AttributeError in /api/trending
  - throttle         → caused AttributeError in loop teardown
These are now all present in the stub so the dashboard renders something
useful even when the real bot fails to load, instead of crashing the loop.
"""

import logging
_log = logging.getLogger(__name__)

try:
    from bot import TradingBot, DEFAULT_WATCHLIST  # type: ignore

    # fetch_stock_price was removed in bot.py v4 (yfinance gone).
    # Provide a no-op shim so any old code that imports it doesn't break.
    try:
        from bot import fetch_stock_price  # type: ignore
    except ImportError:
        async def fetch_stock_price(ticker):  # type: ignore
            return None

    _log.info("[BotCore] bot.py imported successfully → real TradingBot active")

except Exception as exc:
    # Log the REAL reason so it shows up in Render logs.
    # Previously this was silently swallowed — developers had no idea why
    # the stub was active. Now you'll see exactly which dependency is missing.
    import warnings, traceback
    warnings.warn(
        f"bot.py failed to import ({type(exc).__name__}: {exc}) "
        f"— TradingBot is running as a stub. "
        f"Check Render logs for the full traceback.",
        stacklevel=1,
    )
    _log.error("[BotCore] bot.py import FAILED — stub active. Full error:\n%s",
               traceback.format_exc())

    class _FakeBreaker:
        reason = "stub mode — bot.py failed to import"
        def is_tripped(self, prices): return False

    class _FakeTrendFilter:
        _counts = {}
        async def refresh(self, force=False): pass
        def top_trending(self, n=10): return []
        def is_trending(self, ticker, min_mentions=None): return True
        def mention_count(self, ticker): return 0

    class _FakeThrottle:
        def is_blocked(self, ticker, signal): return False
        def record(self, ticker, signal): pass

    class _FakePortfolio:
        starting_cash = 100_000.0
        risk_per_trade = 0.05
        def snapshot(self, prices):
            return {
                "cash": 100_000.0, "positions": {}, "pnl": 0,
                "total_value": 100_000.0, "total_pnl": 0,
                "recent_trades": [], "win_rate": 0.0, "realised_pnl": 0.0,
            }
        def save(self, path): pass
        def check_stops(self, prices): return []
        def position_exists(self, ticker): return False
        def total_value(self, prices): return self.starting_cash

    class TradingBot:   # type: ignore[no-redef]
        """
        Stub — active only when bot.py fails to import.
        Contains all attributes that app.py accesses so the Flask
        dashboard loads and shows a clear error instead of crashing.
        """
        def __init__(self, **kwargs):
            self._prices         = {}
            self._signals        = []
            self._skipped        = []
            self.watchlist       = kwargs.get("watchlist", DEFAULT_WATCHLIST)
            self.hours_guard     = None
            self.portfolio       = _FakePortfolio()
            self.breaker         = _FakeBreaker()
            self.trend_filter    = _FakeTrendFilter()
            self.throttle        = _FakeThrottle()
            # FIX: _portfolio_path was missing → AttributeError in app.py:143
            self._portfolio_path = kwargs.get("portfolio_path", "portfolio.json")

        def _snapshot(self, blocked=""):
            snap = self.portfolio.snapshot(self._prices)
            snap["signals"]  = []
            snap["skipped"]  = []
            snap["blocked"]  = blocked or "bot.py failed to import — check Render logs"
            snap["trending"] = {}
            return snap

        async def run_once(self, tickers=None):
            return self._snapshot()

    DEFAULT_WATCHLIST = [
        "$NVDA","$TSLA","$AAPL","$MSFT","$AMZN",
        "$META","$GOOGL","$AMD","$PLTR","$COIN",
    ]

    async def fetch_stock_price(ticker):  # type: ignore
        return None


__all__ = ["TradingBot", "fetch_stock_price", "DEFAULT_WATCHLIST"]