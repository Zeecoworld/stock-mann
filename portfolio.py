"""
paper_trader/portfolio.py  — v3 FIXED
─────────────────────────────────────────────────────────────────────────────
FIXES vs v2:
  1. total_value() / check_stops() now handles both "$GOOGL" and "GOOGL" price
     keys — previously broke silently when prices were stored with $ prefix.
  2. execute_buy() now logs WHY it failed (max positions, insufficient cash,
     size too small) instead of returning None silently.
  3. _size_order() has a minimum floor so tiny portfolios still generate
     meaningful orders (default $50 minimum).
  4. Positions dict is keyed exactly as passed — no $ stripping — so lookups
     from the signal (which uses original $-prefixed ticker) always hit.
  5. Added position_exists() helper used by bot.py to detect duplicates.
  6. summary_str() now shows win rate cleanly.

NEW in v3 (persistence fix):
  7. save() now writes a full state file (cash + positions + trade_log +
     config) — not just the display snapshot — so nothing is lost on restart.
  8. load() classmethod reconstructs a PaperPortfolio from that state file,
     including all open positions and the full trade history.
  9. TradingBot calls load() automatically on startup when portfolio.json
     exists, so a process restart never creates phantom duplicate buys.
"""
from __future__ import annotations

import json, logging, time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    ticker:    str
    shares:    float
    avg_price: float
    source:    str   = "stock"
    opened_at: float = field(default_factory=time.time)

    @property
    def cost_basis(self):
        return self.shares * self.avg_price

    def current_value(self, price):
        return self.shares * price

    def unrealised_pnl(self, price):
        return self.current_value(price) - self.cost_basis

    def pnl_pct(self, price):
        return self.unrealised_pnl(price) / self.cost_basis if self.cost_basis else 0.0


@dataclass
class TradeRecord:
    timestamp:  float
    ticker:     str
    action:     str
    shares:     float
    price:      float
    pnl:        float = 0.0
    reasoning:  str   = ""
    confidence: float = 0.0
    source:     str   = "stock"

    def to_dict(self):
        return asdict(self)


class PaperPortfolio:
    def __init__(
        self,
        starting_cash:   float = 10_000.0,
        risk_per_trade:  float = 0.05,     # FIX: 5% default (was 2%)
        max_positions:   int   = 5,
        stop_loss_pct:   float = 0.05,
        take_profit_pct: float = 0.12,
        min_order_size:  float = 50.0,     # FIX: new — minimum $ per trade
    ):
        self.cash            = starting_cash
        self.starting_cash   = starting_cash
        self.risk_per_trade  = risk_per_trade
        self.max_positions   = max_positions
        self.stop_loss_pct   = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_order_size  = min_order_size
        self.positions: Dict[str, Position] = {}
        self.trade_log: List[TradeRecord]   = []

    # ── Price lookup helper ───────────────────────────────────────────────

    def _get_price(self, ticker: str, prices: Dict[str, float]) -> Optional[float]:
        """
        FIX: Try both $TICKER and TICKER formats so price lookup never fails
        due to key format mismatch between bot.py and portfolio.py.
        """
        if ticker in prices:
            return prices[ticker]
        # Try stripping $
        stripped = ticker.lstrip("$")
        if stripped in prices:
            return prices[stripped]
        # Try adding $
        dollarised = f"${stripped}"
        if dollarised in prices:
            return prices[dollarised]
        return None

    # ── Portfolio value ───────────────────────────────────────────────────

    def total_value(self, prices: Dict[str, float]) -> float:
        pos_value = sum(
            p.current_value(self._get_price(p.ticker, prices) or p.avg_price)
            for p in self.positions.values()
        )
        return self.cash + pos_value

    def total_pnl(self, prices):
        return self.total_value(prices) - self.starting_cash

    def total_pnl_pct(self, prices):
        return self.total_pnl(prices) / self.starting_cash

    def realised_pnl(self):
        return sum(t.pnl for t in self.trade_log if t.action in ("SELL", "CLOSE"))

    def win_rate(self):
        closed = [t for t in self.trade_log if t.action in ("SELL", "CLOSE")]
        return sum(1 for t in closed if t.pnl > 0) / len(closed) if closed else 0.0

    def position_exists(self, ticker: str) -> bool:
        """FIX: Check both $ and non-$ formats."""
        return ticker in self.positions or ticker.lstrip("$") in self.positions

    # ── Order sizing ──────────────────────────────────────────────────────

    def _size_order(self, price: float, confidence: float) -> float:
        portfolio_value = self.cash + sum(p.cost_basis for p in self.positions.values())
        target_dollars  = portfolio_value * self.risk_per_trade * min(confidence, 1.0)

        target_dollars  = max(target_dollars, self.min_order_size)

        # Don't exceed available cash (95% of cash to leave buffer)
        max_dollars     = self.cash * 0.95
        actual_dollars  = min(target_dollars, max_dollars)

        shares = actual_dollars / price if price > 0 else 0
        logger.info("[Sizing] ticker target=$%.2f  actual=$%.2f  shares=%.4f @ $%.2f",
                    target_dollars, actual_dollars, shares, price)
        return round(shares, 4)


    def execute_buy(self, ticker, price, confidence, reasoning, source="stock"):
        if len(self.positions) >= self.max_positions:
            logger.warning("[Portfolio] BUY FAILED %s — max positions (%d) reached",
                           ticker, self.max_positions)
            return None

        # FIX: check both key formats before rejecting as duplicate
        if self.position_exists(ticker):
            logger.info("[Portfolio] BUY SKIPPED %s — already holding this ticker", ticker)
            return None

        shares = self._size_order(price, confidence)
        cost   = shares * price

        if cost < self.min_order_size:
            logger.warning("[Portfolio] BUY FAILED %s — order size $%.2f below minimum $%.2f "
                           "(portfolio cash=$%.2f, risk=%.0f%%, conf=%.2f)",
                           ticker, cost, self.min_order_size, self.cash,
                           self.risk_per_trade * 100, confidence)
            return None

        if cost > self.cash:
            logger.warning("[Portfolio] BUY FAILED %s — insufficient cash ($%.2f < $%.2f)",
                           ticker, self.cash, cost)
            return None

        self.cash -= cost
        # FIX: store with whatever ticker key was passed — keep consistent with signals
        self.positions[ticker] = Position(
            ticker=ticker, shares=shares, avg_price=price, source=source
        )
        rec = TradeRecord(
            timestamp=time.time(), ticker=ticker, action="BUY",
            shares=shares, price=price, reasoning=reasoning,
            confidence=confidence, source=source,
        )
        self.trade_log.append(rec)
        logger.info("[Portfolio] ✅ BUY  %s  %.4f shares @ $%.2f  cost=$%.2f  cash_remaining=$%.2f",
                    ticker, shares, price, cost, self.cash)
        return rec

    def execute_sell(self, ticker, price, confidence, reasoning):
        # FIX: try both key formats
        pos = self.positions.get(ticker)
        if pos is None:
            stripped = ticker.lstrip("$")
            pos = self.positions.get(stripped)
            if pos:
                ticker = stripped  # use the key that actually exists

        if not pos:
            logger.warning("[Portfolio] SELL FAILED %s — no position found", ticker)
            return None

        proceeds = pos.shares * price
        pnl      = proceeds - pos.cost_basis
        self.cash += proceeds
        del self.positions[ticker]

        rec = TradeRecord(
            timestamp=time.time(), ticker=ticker, action="SELL",
            shares=pos.shares, price=price, pnl=pnl,
            reasoning=reasoning, confidence=confidence, source=pos.source,
        )
        self.trade_log.append(rec)
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        logger.info("[Portfolio] %s SELL %s  PnL=$%+.2f  proceeds=$%.2f",
                    pnl_emoji, ticker, pnl, proceeds)
        return rec

    def check_stops(self, prices: Dict[str, float]) -> List[TradeRecord]:
        """FIX: uses _get_price() so $ vs non-$ mismatch can't cause silent skips."""
        closed = []
        for ticker, pos in list(self.positions.items()):
            price = self._get_price(ticker, prices)
            if price is None:
                logger.debug("[Portfolio] No price for %s — skip stop check", ticker)
                continue
            pct = pos.pnl_pct(price)
            if pct <= -self.stop_loss_pct:
                logger.info("[Portfolio] 🛑 STOP-LOSS %s at %.1f%%", ticker, pct * 100)
                r = self.execute_sell(ticker, price, 1.0, f"Stop-loss triggered at {pct:.1%}")
                if r:
                    closed.append(r)
            elif pct >= self.take_profit_pct:
                logger.info("[Portfolio] 🎯 TAKE-PROFIT %s at %.1f%%", ticker, pct * 100)
                r = self.execute_sell(ticker, price, 1.0, f"Take-profit triggered at {pct:.1%}")
                if r:
                    closed.append(r)
        return closed


    def snapshot(self, prices: Dict[str, float] = None) -> dict:
        prices = prices or {}
        positions_out = {}
        for t, p in self.positions.items():
            curr_price = self._get_price(t, prices) or p.avg_price
            positions_out[t] = {
                "shares":         p.shares,
                "avg_price":      p.avg_price,
                "current_price":  curr_price,
                "unrealised_pnl": round(p.unrealised_pnl(curr_price), 2),
                "pnl_pct":        round(p.pnl_pct(curr_price) * 100, 2),
                "source":         p.source,
                "cost_basis":     round(p.cost_basis, 2),
            }
        total_val = self.total_value(prices)
        total_pnl = self.total_pnl(prices)
        return {
            "cash":            round(self.cash, 2),
            "starting_cash":   self.starting_cash,
            "total_value":     round(total_val, 2),
            "total_pnl":       round(total_pnl, 2),
            "total_pnl_pct":   round(self.total_pnl_pct(prices) * 100, 2),
            "realised_pnl":    round(self.realised_pnl(), 2),
            "unrealised_pnl":  round(total_val - self.cash - self.starting_cash + self.cash, 2),
            "win_rate":        round(self.win_rate() * 100, 2),
            "num_positions":   len(self.positions),
            "num_trades":      len(self.trade_log),
            "max_positions":   self.max_positions,
            "positions":       positions_out,
            "recent_trades":   [t.to_dict() for t in self.trade_log[-20:]],
        }

    def save(self, path="portfolio.json"):
        """
        Persist full portfolio state — cash, open positions, trade log, and
        config — so load() can reconstruct it exactly after a restart.
        This replaces the old snapshot()-only save that lost positions on restart.
        """
        state = {
            "_version":       3,
            "cash":           self.cash,
            "starting_cash":  self.starting_cash,
            "risk_per_trade": self.risk_per_trade,
            "max_positions":  self.max_positions,
            "stop_loss_pct":  self.stop_loss_pct,
            "take_profit_pct":self.take_profit_pct,
            "min_order_size": self.min_order_size,
            "positions": {
                ticker: {
                    "ticker":    p.ticker,
                    "shares":    p.shares,
                    "avg_price": p.avg_price,
                    "source":    p.source,
                    "opened_at": p.opened_at,
                }
                for ticker, p in self.positions.items()
            },
            "trade_log": [asdict(t) for t in self.trade_log],
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("[Portfolio] Saved to %s  (positions=%d  trades=%d)",
                    path, len(self.positions), len(self.trade_log))

    @classmethod
    def load(cls, path="portfolio.json") -> "PaperPortfolio":
        """
        Reconstruct a PaperPortfolio from a state file written by save().
        Restores cash, all open positions, and the full trade log.
        Returns a fresh PaperPortfolio (with defaults) if the file is missing
        or corrupt — never raises.
        """
        import os
        if not os.path.exists(path):
            logger.info("[Portfolio] No state file at %s — starting fresh", path)
            return cls()
        try:
            with open(path) as f:
                state = json.load(f)

            # Support old snapshot-only saves (v1/v2): they have no "_version"
            # key and no raw "positions" with avg_price. Fall back to fresh.
            if state.get("_version", 1) < 3:
                logger.warning(
                    "[Portfolio] State file %s is old format (pre-v3) — "
                    "cannot restore positions. Starting fresh. "
                    "Run the bot once to create an up-to-date state file.", path
                )
                return cls(starting_cash=state.get("starting_cash", 10_000.0))

            p = cls(
                starting_cash   = state["starting_cash"],
                risk_per_trade  = state.get("risk_per_trade",  0.05),
                max_positions   = state.get("max_positions",   5),
                stop_loss_pct   = state.get("stop_loss_pct",   0.05),
                take_profit_pct = state.get("take_profit_pct", 0.12),
                min_order_size  = state.get("min_order_size",  50.0),
            )
            p.cash = state["cash"]

            # Restore open positions
            for ticker, pos_data in state.get("positions", {}).items():
                p.positions[ticker] = Position(
                    ticker    = pos_data["ticker"],
                    shares    = pos_data["shares"],
                    avg_price = pos_data["avg_price"],
                    source    = pos_data.get("source", "stock"),
                    opened_at = pos_data.get("opened_at", time.time()),
                )

            # Restore full trade log
            for rec in state.get("trade_log", []):
                p.trade_log.append(TradeRecord(
                    timestamp  = rec["timestamp"],
                    ticker     = rec["ticker"],
                    action     = rec["action"],
                    shares     = rec["shares"],
                    price      = rec["price"],
                    pnl        = rec.get("pnl", 0.0),
                    reasoning  = rec.get("reasoning", ""),
                    confidence = rec.get("confidence", 0.0),
                    source     = rec.get("source", "stock"),
                ))

            logger.info(
                "[Portfolio] Loaded from %s — cash=$%.2f  positions=%d  trades=%d",
                path, p.cash, len(p.positions), len(p.trade_log),
            )
            if p.positions:
                for t, pos in p.positions.items():
                    logger.info("  [Portfolio]   holding %s  %.4f sh @ $%.2f",
                                t, pos.shares, pos.avg_price)
            return p

        except Exception as exc:
            logger.error(
                "[Portfolio] Failed to load %s (%s) — starting fresh", path, exc
            )
            return cls()

    def summary_str(self, prices=None):
        s = self.snapshot(prices or {})
        wr = f"{s['win_rate']:.0f}%" if s['num_trades'] > 0 else "—"
        return (
            f"Cash=${s['cash']:,.2f}  "
            f"Portfolio=${s['total_value']:,.2f}  "
            f"PnL=${s['total_pnl']:+,.2f} ({s['total_pnl_pct']:+.1f}%)  "
            f"WinRate={wr}  "
            f"Positions={s['num_positions']}/{self.max_positions}  "
            f"Trades={s['num_trades']}"
        )