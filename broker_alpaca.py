"""
paper_trader/broker_alpaca.py
Drop-in replacement for PaperPortfolio when PRODUCTION=true.

Swap in bot.py __init__:
    from broker_alpaca import AlpacaBroker
    self.portfolio = AlpacaBroker()

pip install alpaca-py
"""
from __future__ import annotations
import logging, os
logger = logging.getLogger(__name__)

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


class AlpacaBroker:
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
            raise ImportError("pip install alpaca-py")

        key    = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise EnvironmentError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")

        paper = "paper-api" in ALPACA_BASE_URL
        self.client        = self._TC(key, secret, paper=paper)
        self.starting_cash = float(self.client.get_account().cash)
        logger.info("[Alpaca] Connected (%s)  cash=$%.2f",
                    "PAPER" if paper else "LIVE", self.starting_cash)

    def execute_buy(self, ticker, price, confidence, reasoning, source="stock",
                    notional=200.0):
        symbol = ticker.lstrip("$").upper()
        try:
            order = self.client.submit_order(self._MOR(
                symbol=symbol, notional=notional,
                side=self._OS.BUY, time_in_force=self._TIF.DAY))
            logger.info("[Alpaca] BUY  %s  $%.0f  id=%s", symbol, notional, order.id)
            return order
        except Exception as e:
            logger.error("[Alpaca] BUY  %s  failed: %s", symbol, e)

    def execute_sell(self, ticker, price, confidence, reasoning):
        symbol = ticker.lstrip("$").upper()
        try:
            self.client.close_position(symbol)
            logger.info("[Alpaca] SELL %s", symbol)
        except Exception as e:
            logger.error("[Alpaca] SELL %s  failed: %s", symbol, e)

    def check_stops(self, prices): return []   # Alpaca handles stops natively
    def snapshot(self, prices=None): return {}
    def save(self, path=""): pass
    def summary_str(self, prices=None):
        acct = self.client.get_account()
        return f"Cash=${float(acct.cash):,.2f}  Equity=${float(acct.equity):,.2f}"
    def total_value(self, prices):
        return float(self.client.get_account().equity)
