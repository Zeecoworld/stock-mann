"""
broker_alpaca.py  — v2  PRODUCTION-READY
─────────────────────────────────────────────────────────────────────────────
FIXES vs v1:
  1. BRACKET ORDERS: execute_buy() now submits a bracket order (entry + stop-loss
     + take-profit) in a single API call. Alpaca enforces exits server-side even
     if the bot process dies. Previously positions had ZERO protection.

  2. ATR-BASED POSITION SIZING: fixed $200 notional replaced with volatility-
     adjusted sizing. We risk exactly RISK_PCT % of portfolio equity per trade,
     divided by 2×ATR as the stop distance. Larger ATR = smaller position.
     This keeps dollar-risk constant across all tickers and portfolio sizes.

  3. MAX POSITIONS ENFORCED: max_positions was "advisory only" — nothing stopped
     the bot from opening 10 correlated tech longs simultaneously. Now checked
     before every buy order.

  4. TIME-BASED EXIT in check_stops(): previously returned [] always. Now scans
     all open positions and closes any that have been held > MAX_HOLD_DAYS and
     are still down > MIN_LOSS_TO_EXIT. Prevents dead-weight positions from
     bleeding the portfolio indefinitely.

  5. NOTIONAL ORDERS: switched from qty= (integer, floors fractional shares and
     can force unaffordable orders) to notional= so Alpaca handles fractional
     math server-side and we spend exactly the calculated amount.

  6. execute_sell() now returns real PnL from the position before closing it,
     not a hardcoded 0.0. Dashboard will now show correct trade history.

ENV:
  ALPACA_API_KEY          — required
  ALPACA_SECRET_KEY       — required
  ALPACA_BASE_URL         — default https://paper-api.alpaca.markets
  ALPACA_RISK_PCT         — % of equity to risk per trade (default 0.01 = 1%)
  ALPACA_TP_PCT           — take-profit % above entry  (default 0.08 = 8%)
  ALPACA_SL_MULTIPLIER    — stop = entry - SL_MULT × ATR (default 2.0)
  ALPACA_MAX_POSITIONS    — max open positions allowed  (default 6)
  ALPACA_MAX_HOLD_DAYS    — days before time-exit fires (default 5)
  ALPACA_MIN_LOSS_EXIT    — min loss % to trigger time-exit (default 0.02)
"""
from __future__ import annotations

import logging, os
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
_RISK_PCT         = float(os.getenv("ALPACA_RISK_PCT",      "0.01"))   # 1% equity risk/trade
_TP_PCT           = float(os.getenv("ALPACA_TP_PCT",        "0.08"))   # 8% take-profit
_SL_MULTIPLIER    = float(os.getenv("ALPACA_SL_MULTIPLIER", "2.0"))    # 2×ATR stop
_MAX_POSITIONS    = int(os.getenv("ALPACA_MAX_POSITIONS",   "6"))
_MAX_HOLD_DAYS    = int(os.getenv("ALPACA_MAX_HOLD_DAYS",   "5"))
_MIN_LOSS_EXIT    = float(os.getenv("ALPACA_MIN_LOSS_EXIT", "0.02"))   # 2% loss to time-exit
_MAX_NOTIONAL_PCT = float(os.getenv("ALPACA_MAX_NOTIONAL_PCT", "0.10"))# cap at 10% equity/trade


class AlpacaBroker:
    """
    Wraps Alpaca TradingClient. Fully implements PaperPortfolio's interface
    so bot.py can swap brokers with zero code changes.
    """

    def __init__(self):
        try:
            from alpaca.trading.client   import TradingClient
            from alpaca.trading.requests import (MarketOrderRequest,
                                                  TakeProfitRequest,
                                                  StopLossRequest)
            from alpaca.trading.enums    import OrderSide, TimeInForce, OrderClass
            self._TC    = TradingClient
            self._MOR   = MarketOrderRequest
            self._TPR   = TakeProfitRequest
            self._SLR   = StopLossRequest
            self._OS    = OrderSide
            self._TIF   = TimeInForce
            self._OC    = OrderClass
        except ImportError:
            raise ImportError("alpaca-py not installed. Run: pip install alpaca-py==0.43.4")

        key    = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise EnvironmentError(
                "Alpaca credentials missing.\n"
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file.\n"
                "Paper keys: https://app.alpaca.markets/paper/dashboard/overview"
            )

        is_paper = "paper-api" in ALPACA_BASE_URL
        try:
            self.client    = self._TC(key, secret, paper=is_paper)
            self._is_paper = is_paper
            acct           = self.client.get_account()
        except Exception as e:
            raise ConnectionError(
                f"Alpaca connection failed: {e}\n"
                f"  URL:   {ALPACA_BASE_URL}\n"
                f"  Paper: {is_paper}\n"
                "Check your API keys and ALPACA_BASE_URL."
            )

        self.starting_cash  = float(acct.equity)
        self.risk_per_trade = _RISK_PCT
        self.max_positions  = _MAX_POSITIONS

        logger.info("[Alpaca] Connected (%s)  equity=$%.2f  cash=$%.2f  "
                    "risk=%.0f%%  max_pos=%d  TP=%.0f%%  SL=%.1f×ATR",
                    "PAPER" if is_paper else "🔴 LIVE",
                    float(acct.equity), float(acct.cash),
                    _RISK_PCT * 100, _MAX_POSITIONS, _TP_PCT * 100, _SL_MULTIPLIER)

    # ── Position helpers ───────────────────────────────────────────────────

    def position_exists(self, ticker: str) -> bool:
        symbol = ticker.lstrip("$").upper()
        try:
            self.client.get_open_position(symbol)
            return True
        except Exception:
            return False

    def get_safe_buying_power(self) -> float:
        try:
            acct = self.client.get_account()
            if acct.trading_blocked or acct.account_blocked:
                logger.warning("[Alpaca] Account restricted — buying power=$0")
                return 0.0
            return float(acct.buying_power)
        except Exception as e:
            logger.error("[Alpaca] Error fetching buying power: %s", e)
            return 0.0

    def _open_position_count(self) -> int:
        try:
            return len(self.client.get_all_positions())
        except Exception:
            return 0

    # ── ATR-based position sizing ──────────────────────────────────────────

    def _calc_notional(self, price: float, atr: float, equity: float) -> Optional[float]:
        """
        Risk-based sizing: risk exactly RISK_PCT of equity per trade.
        Stop distance = SL_MULTIPLIER × ATR. Cap at MAX_NOTIONAL_PCT of equity.
        Returns None if the trade is too small to be worth placing.
        """
        if atr <= 0 or price <= 0:
            # Fallback: risk RISK_PCT of equity with a 3% assumed stop
            stop_dist = price * 0.03
        else:
            stop_dist = _SL_MULTIPLIER * atr

        shares   = (equity * _RISK_PCT) / stop_dist
        notional = shares * price
        notional = min(notional, equity * _MAX_NOTIONAL_PCT)   # hard cap

        if notional < 10:
            logger.warning("[Alpaca] Computed notional $%.2f < $10 — skipping", notional)
            return None

        logger.debug("[Alpaca] Sizing: equity=$%.0f  risk=$%.0f  "
                     "ATR=%.2f  stop_dist=%.2f  notional=$%.2f",
                     equity, equity * _RISK_PCT, atr, stop_dist, notional)
        return round(notional, 2)

    # ── Execute BUY — bracket order ────────────────────────────────────────

    def execute_buy(self, ticker: str, price: float, conf: float,
                    reason: str, source: str = "stock", atr: float = 0.0):
        """
        FIX 1 — bracket order: entry + stop-loss + take-profit submitted atomically.
        FIX 2 — ATR-based notional replaces fixed $200.
        FIX 3 — max_positions enforced before every buy.
        FIX 5 — uses notional= instead of qty= to avoid integer-floor errors.
        """
        clean = ticker.lstrip("$").upper()
        try:
            # Guard: max positions
            open_count = self._open_position_count()
            if open_count >= self.max_positions:
                logger.warning("[Alpaca] Max positions (%d/%d) reached — skip BUY %s",
                               open_count, self.max_positions, clean)
                return None

            bp     = self.get_safe_buying_power()
            acct   = self.client.get_account()
            equity = float(acct.equity)

            # Guard: buying power
            if bp <= 10:
                logger.warning("[Alpaca] Buying power $%.2f too low — skip BUY %s", bp, clean)
                return None

            # ATR-based sizing
            notional = self._calc_notional(price, atr, equity)
            if notional is None:
                return None

            # Never spend more than available buying power
            notional = min(notional, bp * 0.95)
            if notional < 10:
                logger.warning("[Alpaca] Post-bp-cap notional $%.2f too low — skip", notional)
                return None

            # Bracket prices
            tp_price = round(price * (1 + _TP_PCT), 2)
            sl_price = round(price - (_SL_MULTIPLIER * atr) if atr > 0 else price * 0.965, 2)
            sl_price = max(sl_price, round(price * 0.90, 2))  # hard floor: never > 10% stop

            order_data = self._MOR(
                symbol         = clean,
                notional       = notional,
                side           = self._OS.BUY,
                time_in_force  = self._TIF.DAY,
                order_class    = self._OC.BRACKET,
                take_profit    = self._TPR(limit_price=tp_price),
                stop_loss      = self._SLR(stop_price=sl_price),
            )

            order = self.client.submit_order(order_data=order_data)
            logger.info("[Alpaca] ✅ BUY %s  notional=$%.2f  entry≈$%.2f  "
                        "TP=$%.2f (+%.0f%%)  SL=$%.2f (-%.1f%%)  id=%s",
                        clean, notional, price, tp_price, _TP_PCT * 100,
                        sl_price, (price - sl_price) / price * 100, order.id)
            return order

        except Exception as e:
            logger.error("[Alpaca] BUY FAILED %s: %s", clean, e)
            return None

    # ── Execute SELL ───────────────────────────────────────────────────────

    def execute_sell(self, ticker: str, price: float,
                     confidence: float = 0.0, reasoning: str = ""):
        """
        FIX 6 — captures real unrealized PnL from the position before closing,
        so the dashboard shows correct trade results instead of hardcoded $0.
        Also cancels any open bracket child orders first to avoid conflicts.
        """
        symbol = ticker.lstrip("$").upper()
        try:
            # Get position PnL before closing
            pnl = 0.0
            try:
                pos = self.client.get_open_position(symbol)
                pnl = float(pos.unrealized_pl or 0)
            except Exception:
                pass

            # Cancel open orders for this symbol first (bracket children)
            try:
                open_orders = self.client.get_orders()
                for o in open_orders:
                    if o.symbol == symbol:
                        self.client.cancel_order_by_id(o.id)
                        logger.debug("[Alpaca] Cancelled bracket child order %s", o.id)
            except Exception as ce:
                logger.debug("[Alpaca] Order cancel attempt: %s", ce)

            self.client.close_position(symbol_or_asset_id=symbol)
            icon = "🟢" if pnl >= 0 else "🔴"
            logger.info("[Alpaca] %s SELL %s  PnL=$%+.2f  reason=%s",
                        icon, symbol, pnl, reasoning[:60])
            return _FakeRecord(ticker=ticker, shares=0, price=price, pnl=pnl)

        except Exception as e:
            logger.error("[Alpaca] SELL FAILED %s: %s", symbol, e)
            return None

    # ── check_stops — time-based exit ─────────────────────────────────────

    def check_stops(self, prices: dict) -> list:
        """
        FIX 4 — Previously returned [] always. Bracket orders handle price-based
        stops natively on Alpaca's side. This method handles the one thing Alpaca
        can't do automatically: TIME-BASED exits.

        If a position has been held > MAX_HOLD_DAYS AND is still down > MIN_LOSS_EXIT,
        we close it manually. This prevents dead-weight losers sitting forever because
        sentiment never turns negative enough to trigger a SELL signal.
        """
        closed = []
        try:
            positions = self.client.get_all_positions()
        except Exception as e:
            logger.error("[Alpaca] check_stops: failed to fetch positions: %s", e)
            return []

        now = datetime.now(timezone.utc)

        for p in positions:
            try:
                # Parse entry time — Alpaca returns ISO string
                created_at = p.created_at
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00"))
                held_days = (now - created_at).total_seconds() / 86400

                pnl_pct = float(p.unrealized_plpc or 0)  # decimal e.g. -0.05 = -5%

                if held_days >= _MAX_HOLD_DAYS and pnl_pct <= -_MIN_LOSS_EXIT:
                    logger.warning(
                        "[Alpaca] ⏰ TIME-EXIT %s  held=%.1fd  PnL=%.1f%%",
                        p.symbol, held_days, pnl_pct * 100)
                    rec = self.execute_sell(
                        ticker    = p.symbol,
                        price     = float(p.current_price or 0),
                        reasoning = f"time-exit: held {held_days:.0f}d, PnL={pnl_pct:.1%}",
                    )
                    if rec:
                        closed.append(rec)
                else:
                    logger.debug("[Alpaca] Hold %s  days=%.1f  PnL=%.1f%%",
                                 p.symbol, held_days, pnl_pct * 100)
            except Exception as e:
                logger.error("[Alpaca] check_stops position loop error (%s): %s",
                             p.symbol, e)

        return closed

    # ── Portfolio snapshot ─────────────────────────────────────────────────

    def snapshot(self, prices: dict = None) -> dict:
        try:
            acct      = self.client.get_account()
            positions = self.client.get_all_positions()
            pos_out   = {}

            for p in positions:
                pos_out[p.symbol] = {
                    "shares":         float(p.qty),
                    "avg_price":      float(p.avg_entry_price),
                    "current_price":  float(p.current_price or 0),
                    "unrealised_pnl": float(p.unrealized_pl or 0),
                    "pnl_pct":        float(p.unrealized_plpc or 0) * 100,
                }

            equity       = float(acct.equity)
            total_pnl    = equity - self.starting_cash
            total_pnl_pct = (total_pnl / self.starting_cash * 100) if self.starting_cash else 0

            return {
                "cash":           float(acct.cash),
                "starting_cash":  self.starting_cash,
                "total_value":    equity,
                "total_pnl":      total_pnl,
                "total_pnl_pct":  total_pnl_pct,
                "realised_pnl":   0.0,   # Alpaca doesn't expose this simply
                "win_rate":       0.0,
                "num_positions":  len(positions),
                "positions":      pos_out,
                "recent_trades":  [],
                "max_positions":  self.max_positions,
                "risk_pct":       self.risk_per_trade,
            }
        except Exception as e:
            logger.error("[Alpaca] snapshot error: %s", e)
            return {
                "cash": 0, "total_value": 0, "positions": {},
                "total_pnl": 0, "total_pnl_pct": 0,
                "recent_trades": [], "win_rate": 0, "realised_pnl": 0,
            }

    # ── Misc interface methods ─────────────────────────────────────────────

    def total_value(self, prices: dict) -> float:
        try:
            return float(self.client.get_account().equity)
        except Exception:
            return self.starting_cash

    def save(self, path: str = ""):
        pass   # Alpaca is cloud-persisted

    @classmethod
    def load(cls, path: str = "portfolio.json") -> "AlpacaBroker":
        logger.info("[Alpaca] load() — state is cloud-persisted, connecting fresh")
        return cls()

    def summary_str(self, prices=None) -> str:
        try:
            acct = self.client.get_account()
            mode = "PAPER" if self._is_paper else "🔴 LIVE"
            return (f"Cash=${float(acct.cash):,.2f}  "
                    f"Equity=${float(acct.equity):,.2f}  "
                    f"P&L=${float(acct.equity) - self.starting_cash:+,.2f}  "
                    f"[Alpaca {mode}]")
        except Exception as e:
            return f"[Alpaca summary error: {e}]"


class _FakeRecord:
    """Duck-type for TradeRecord — carries real PnL from execute_sell()."""
    def __init__(self, ticker: str, shares: float, price: float, pnl: float):
        self.ticker = ticker
        self.shares = shares
        self.price  = price
        self.pnl    = pnl

    def to_dict(self) -> dict:
        return vars(self)