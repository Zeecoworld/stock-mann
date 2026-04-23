
from __future__ import annotations
import logging, os
logger = logging.getLogger(__name__)

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
_DEFAULT_NOTIONAL = float(os.getenv("ALPACA_ORDER_NOTIONAL", "200"))   # $200 per trade


class AlpacaBroker:
    """
    Wraps Alpaca TradingClient to match PaperPortfolio's interface
    so bot.py can swap between paper and live trading with zero code changes.
    """

    def __init__(self):
        try:
            from alpaca.trading.client   import TradingClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums    import OrderSide, TimeInForce
            self._TC  = TradingClient
            self._MOR = MarketOrderRequest
            self._OS  = OrderSide
            self._TIF = TimeInForce
        except ImportError:
            raise ImportError(
                "alpaca-py not installed. Run: pip install alpaca-py"
            )

        key    = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise EnvironmentError(
                "Alpaca credentials missing.\n"
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file.\n"
                "Paper trading keys: https://app.alpaca.markets/paper/dashboard/overview\n"
                "Live trading keys:  https://app.alpaca.markets/live/dashboard/overview"
            )

        is_paper = "paper-api" in ALPACA_BASE_URL
        try:
            self.client = self._TC(key, secret, paper=is_paper)
            acct = self.client.get_account()
        except Exception as e:
            raise ConnectionError(
                f"Alpaca connection failed: {e}\n"
                f"  URL:    {ALPACA_BASE_URL}\n"
                f"  Paper:  {is_paper}\n"
                "Check your API keys and ALPACA_BASE_URL env var."
            )

        # Mirror PaperPortfolio attributes used by bot.py
        self.starting_cash  = float(acct.cash)
        self.risk_per_trade = 0.05        # informational only — Alpaca uses notional
        self.max_positions  = 10          # Alpaca doesn't enforce this — advisory

        logger.info("[Alpaca] Connected (%s)  cash=$%.2f  equity=$%.2f",
                    "PAPER" if is_paper else "🔴 LIVE",
                    float(acct.cash), float(acct.equity))

    # ── PaperPortfolio interface ──────────────────────────────────────────

    def position_exists(self, ticker: str) -> bool:
        """Check whether we currently hold this ticker (any key format)."""
        symbol = ticker.lstrip("$").upper()
        try:
            positions = self.client.get_all_positions()
            held = {p.symbol.upper() for p in positions}
            return symbol in held
        except Exception as e:
            logger.warning("[Alpaca] position_exists check failed: %s", e)
            return False

    def execute_buy(self, ticker, price, confidence, reasoning,
                    source="stock", notional=None):
        symbol   = ticker.lstrip("$").upper()
        notional = notional or _DEFAULT_NOTIONAL
        try:
            order = self.client.submit_order(self._MOR(
                symbol=symbol,
                notional=notional,
                side=self._OS.BUY,
                time_in_force=self._TIF.DAY,
            ))
            logger.info("[Alpaca] ✅ BUY  %s  $%.0f  id=%s", symbol, notional, order.id)
            # Return a duck-typed record so bot.py printing logic works
            return _FakeRecord(ticker=ticker, shares=notional / price,
                               price=price, pnl=0.0)
        except Exception as e:
            logger.error("[Alpaca] BUY FAILED %s: %s", symbol, e)
            return None

    def execute_sell(self, ticker, price, confidence, reasoning):
        symbol = ticker.lstrip("$").upper()
        try:
            self.client.close_position(symbol)
            logger.info("[Alpaca] ✅ SELL %s", symbol)
            return _FakeRecord(ticker=ticker, shares=0, price=price, pnl=0.0)
        except Exception as e:
            logger.error("[Alpaca] SELL FAILED %s: %s", symbol, e)
            return None

    def check_stops(self, prices) -> list:
        """Alpaca handles stops natively — we don't trigger them manually."""
        return []

    def snapshot(self, prices=None) -> dict:
        try:
            acct = self.client.get_account()
            positions = self.client.get_all_positions()
            pos_out = {}
            for p in positions:
                pos_out[p.symbol] = {
                    "shares":         float(p.qty),
                    "avg_price":      float(p.avg_entry_price),
                    "current_price":  float(p.current_price or 0),
                    "unrealised_pnl": float(p.unrealized_pl or 0),
                    "pnl_pct":        float(p.unrealized_plpc or 0) * 100,
                }
            return {
                "cash":          float(acct.cash),
                "starting_cash": self.starting_cash,
                "total_value":   float(acct.equity),
                "total_pnl":     float(acct.equity) - self.starting_cash,
                "total_pnl_pct": ((float(acct.equity) - self.starting_cash)
                                  / self.starting_cash * 100),
                "realised_pnl":  0.0,
                "win_rate":      0.0,
                "num_positions": len(positions),
                "positions":     pos_out,
                "recent_trades": [],
            }
        except Exception as e:
            logger.error("[Alpaca] snapshot error: %s", e)
            return {"cash": 0, "total_value": 0, "positions": {}}

    def save(self, path=""): pass   # Alpaca is cloud-persisted

    def summary_str(self, prices=None) -> str:
        try:
            acct = self.client.get_account()
            return (f"Cash=${float(acct.cash):,.2f}  "
                    f"Equity=${float(acct.equity):,.2f}  "
                    f"P&L=${float(acct.equity) - self.starting_cash:+,.2f}  "
                    f"[Alpaca {'PAPER' if 'paper' in ALPACA_BASE_URL else '🔴 LIVE'}]")
        except Exception as e:
            return f"[Alpaca summary error: {e}]"

    def total_value(self, prices) -> float:
        try:
            return float(self.client.get_account().equity)
        except Exception:
            return self.starting_cash


class _FakeRecord:
    """Minimal duck-type for TradeRecord so bot.py printing works."""
    def __init__(self, ticker, shares, price, pnl):
        self.ticker = ticker
        self.shares = shares
        self.price  = price
        self.pnl    = pnl

    def to_dict(self):
        return vars(self)