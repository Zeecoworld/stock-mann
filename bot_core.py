"""
paper_trader_flask/bot_core.py
──────────────────────────────
Re-exports TradingBot + helpers from the original bot.py so Flask can
import them without touching __main__ guards.

Drop your original bot.py alongside this file and rename it to
_bot_impl.py  (or just paste the class here directly).
"""

# If you kept the original file as bot.py in this same folder:
try:
    from bot import TradingBot, fetch_stock_price, DEFAULT_WATCHLIST  # type: ignore
except ImportError:
    # Stub so the Flask app boots even without the real bot installed.
    # Replace with your actual imports once the full project is wired up.
    import warnings
    warnings.warn(
        "bot.py not found — TradingBot is a stub. "
        "Place your original bot.py in this directory.",
        stacklevel=1,
    )

    class TradingBot:   # type: ignore[no-redef]
        """Placeholder — replace by importing the real TradingBot."""
        def __init__(self, **kwargs):
            self._prices  = {}
            self.watchlist = kwargs.get("watchlist", [])
            self.hours_guard = None

            class _FakePortfolio:
                starting_cash = 100_000.0
                def snapshot(self, prices): return {"cash": 100_000.0, "positions": {}, "pnl": 0}
                def save(self, path): pass
                def check_stops(self, prices): return []

            self.portfolio = _FakePortfolio()

        async def run_once(self, tickers=None):
            return {
                "signals":  [],
                "skipped":  [],
                "blocked":  "bot.py not installed",
                "trending": {},
            }

    DEFAULT_WATCHLIST = [
        "$NVDA","$TSLA","$AAPL","$MSFT","$AMZN",
        "$META","$GOOGL","$AMD","$PLTR","$COIN",
    ]

    async def fetch_stock_price(ticker):
        return None


__all__ = ["TradingBot", "fetch_stock_price", "DEFAULT_WATCHLIST"]