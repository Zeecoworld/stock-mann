from __future__ import annotations
import logging, os
logger = logging.getLogger(__name__)

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
_DEFAULT_NOTIONAL = float(os.getenv("ALPACA_ORDER_NOTIONAL", "200"))   # $200 per trade


class AlpacaBroker:
    """
    Wraps Alpaca TradingClient to match PaperPortfolio's interface
    so bot.py can swap between paper and live trading with zero code changes.

    Verified against alpaca-py 0.43.3 docs:
      - submit_order() takes order_data= keyword arg (not positional)
      - close_position() takes symbol_or_asset_id= keyword arg
      - get_all_positions() returns Position objects with .qty (not .shares)
        and .avg_entry_price (not .avg_cost)
      - position_exists() uses get_open_position(symbol) which raises
        APIError if no position — cleaner than fetching all positions
    """

    def __init__(self):
        try:
            from alpaca.trading.client   import TradingClient
            from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
            from alpaca.trading.enums    import OrderSide, TimeInForce
            self._TC   = TradingClient
            self._MOR  = MarketOrderRequest
            self._CPR  = ClosePositionRequest
            self._OS   = OrderSide
            self._TIF  = TimeInForce
        except ImportError:
            raise ImportError(
                "alpaca-py not installed. Run: pip install alpaca-py==0.43.3"
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
            self.client  = self._TC(key, secret, paper=is_paper)
            self._is_paper = is_paper
            acct = self.client.get_account()
        except Exception as e:
            raise ConnectionError(
                f"Alpaca connection failed: {e}\n"
                f"  URL:    {ALPACA_BASE_URL}\n"
                f"  Paper:  {is_paper}\n"
                "Check your API keys and ALPACA_BASE_URL env var."
            )

        # Mirror PaperPortfolio attributes used by bot.py
        self.starting_cash  = float(acct.equity)
        self.risk_per_trade = 0.05        # informational only — Alpaca uses notional
        self.max_positions  = 10          # Alpaca doesn't enforce this — advisory

        logger.info("[Alpaca] Connected (%s)  cash=$%.2f  equity=$%.2f",
                    "PAPER" if is_paper else "🔴 LIVE",
                    float(acct.cash), float(acct.equity))

    # ── PaperPortfolio interface ──────────────────────────────────────────

    def position_exists(self, ticker: str) -> bool:
        """
        FIX (alpaca-py 0.43.3): use get_open_position(symbol) which raises
        an APIError / exception when no position exists — faster than
        fetching all positions and scanning the list.
        """
        symbol = ticker.lstrip("$").upper()
        try:
            self.client.get_open_position(symbol)
            return True
        except Exception:
            # Any exception (APIError 404, connection error) → treat as no position
            return False

    def execute_buy(self, ticker: str, price: float, conf: float, reason: str, source: str = "stock"):
        try:
           
            bp = self.get_safe_buying_power()
            
         
            trade_size = min(_DEFAULT_NOTIONAL, bp * 0.95) # 5% buffer
            
            if bp < trade_size or bp <= 0:
                logger.warning(f"[Alpaca] Insufficient buying power (${bp:,.2f}). Skipping {ticker}.")
                return None

            clean_ticker = ticker.lstrip("$").upper()
            qty = max(1, int(trade_size / price))

            order_data = self._MOR(
                symbol=clean_ticker,
                qty=qty,
                side=self._OS.BUY,
                time_in_force=self._TIF.DAY
            )
            
            order = self.client.submit_order(order_data=order_data)
            logger.info(f"[Alpaca] BUY Executed: {qty} shares of {clean_ticker} | Order ID: {order.id}")
            return order

        except Exception as e:
            logger.error(f"[Alpaca] Failed to execute BUY for {ticker}: {e}")
            return None

    def execute_sell(self, ticker, price, confidence, reasoning):
        symbol = ticker.lstrip("$").upper()
        try:
            # FIX (alpaca-py 0.43.3): close_position() requires
            # symbol_or_asset_id= keyword arg, not positional
            self.client.close_position(symbol_or_asset_id=symbol)
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
            acct      = self.client.get_account()
            positions = self.client.get_all_positions()
            pos_out   = {}
            for p in positions:
                # FIX (alpaca-py 0.43.3): Position model fields are:
                #   p.qty            (not p.shares)
                #   p.avg_entry_price (not p.avg_cost)
                #   p.current_price  (may be None pre-market)
                #   p.unrealized_pl  (not p.unrealized_pnl)
                #   p.unrealized_plpc (as a decimal, e.g. 0.05 = 5%)
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

    def save(self, path=""): pass   # Alpaca is cloud-persisted — no local state needed

    @classmethod
    def load(cls, path="portfolio.json") -> "AlpacaBroker":
        """
        Drop-in equivalent of PaperPortfolio.load() so bot.py can call
        PaperPortfolio.load() or AlpacaBroker.load() interchangeably.
        Alpaca state lives in the cloud — we just instantiate a fresh client.
        The path argument is accepted but ignored.
        """
        import logging as _log
        _log.getLogger(__name__).info(
            "[Alpaca] load() called — state is cloud-persisted, connecting fresh"
        )
        return cls()

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
        
    def get_safe_buying_power(self) -> float:
        try:
            acct = self.client.get_account()
            if acct.trading_blocked or acct.account_blocked:
                logger.warning("[Alpaca] Account is restricted. Buying power = $0")
                return 0.0
            return float(acct.buying_power)
        except Exception as e:
            logger.error(f"[Alpaca] Error fetching buying power: {e}")
            return 0.0


class _FakeRecord:
    """Minimal duck-type for TradeRecord so bot.py printing works."""
    def __init__(self, ticker, shares, price, pnl):
        self.ticker = ticker
        self.shares = shares
        self.price  = price
        self.pnl    = pnl

    def to_dict(self):
        return vars(self)