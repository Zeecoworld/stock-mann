"""
paper_trader/portfolio.py
Paper trading portfolio — cash, positions, P&L, stop-loss, take-profit, trade log.
Swap execute_buy/sell for AlpacaBroker methods to go live.
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
    def cost_basis(self): return self.shares * self.avg_price
    def current_value(self, price): return self.shares * price
    def unrealised_pnl(self, price): return self.current_value(price) - self.cost_basis
    def pnl_pct(self, price): return self.unrealised_pnl(price) / self.cost_basis if self.cost_basis else 0.0


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
    def to_dict(self): return asdict(self)


class PaperPortfolio:
    def __init__(
        self,
        starting_cash:   float = 10_000.0,
        risk_per_trade:  float = 0.02,
        max_positions:   int   = 10,
        stop_loss_pct:   float = 0.05,
        take_profit_pct: float = 0.10,
    ):
        self.cash            = starting_cash
        self.starting_cash   = starting_cash
        self.risk_per_trade  = risk_per_trade
        self.max_positions   = max_positions
        self.stop_loss_pct   = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.positions: Dict[str, Position]  = {}
        self.trade_log: List[TradeRecord]    = []

    def total_value(self, prices: Dict[str,float]) -> float:
        return self.cash + sum(
            p.current_value(prices.get(p.ticker, p.avg_price))
            for p in self.positions.values())

    def total_pnl(self, prices): return self.total_value(prices) - self.starting_cash
    def total_pnl_pct(self, prices): return self.total_pnl(prices) / self.starting_cash
    def realised_pnl(self): return sum(t.pnl for t in self.trade_log if t.action in ("SELL","CLOSE"))

    def win_rate(self):
        c = [t for t in self.trade_log if t.action in ("SELL","CLOSE")]
        return sum(1 for t in c if t.pnl > 0) / len(c) if c else 0.0

    def _size_order(self, price: float, confidence: float) -> float:
        portfolio = self.cash + sum(p.cost_basis for p in self.positions.values())
        dollars   = portfolio * self.risk_per_trade * min(confidence, 1.0)
        shares    = dollars / price if price > 0 else 0
        max_sh    = (self.cash / price) * 0.95 if price > 0 else 0
        return round(min(shares, max_sh), 4)

    def execute_buy(self, ticker, price, confidence, reasoning, source="stock"):
        if len(self.positions) >= self.max_positions:
            logger.warning("[Portfolio] Max positions reached"); return None
        if ticker in self.positions:
            logger.info("[Portfolio] Already holding %s", ticker); return None
        shares = self._size_order(price, confidence)
        cost   = shares * price
        if cost > self.cash:
            logger.warning("[Portfolio] Insufficient cash"); return None
        self.cash -= cost
        self.positions[ticker] = Position(ticker=ticker, shares=shares,
                                          avg_price=price, source=source)
        rec = TradeRecord(timestamp=time.time(), ticker=ticker, action="BUY",
                          shares=shares, price=price, reasoning=reasoning,
                          confidence=confidence, source=source)
        self.trade_log.append(rec)
        logger.info("[Portfolio] BUY  %s  %.4f @ $%.2f  cost=$%.2f", ticker, shares, price, cost)
        return rec

    def execute_sell(self, ticker, price, confidence, reasoning):
        pos = self.positions.get(ticker)
        if not pos: return None
        proceeds = pos.shares * price
        pnl      = proceeds - pos.cost_basis
        self.cash += proceeds
        del self.positions[ticker]
        rec = TradeRecord(timestamp=time.time(), ticker=ticker, action="SELL",
                          shares=pos.shares, price=price, pnl=pnl,
                          reasoning=reasoning, confidence=confidence, source=pos.source)
        self.trade_log.append(rec)
        logger.info("[Portfolio] SELL %s  PnL=$%+.2f", ticker, pnl)
        return rec

    def check_stops(self, prices: Dict[str,float]):
        closed = []
        for ticker, pos in list(self.positions.items()):
            price = prices.get(ticker)
            if price is None: continue
            pct = pos.pnl_pct(price)
            if pct <= -self.stop_loss_pct:
                r = self.execute_sell(ticker, price, 1.0, f"Stop-loss at {pct:.1%}")
                if r: closed.append(r)
            elif pct >= self.take_profit_pct:
                r = self.execute_sell(ticker, price, 1.0, f"Take-profit at {pct:.1%}")
                if r: closed.append(r)
        return closed

    def snapshot(self, prices: Dict[str,float] = None) -> dict:
        prices = prices or {}
        return {
            "cash":          round(self.cash, 2),
            "starting_cash": self.starting_cash,
            "total_value":   round(self.total_value(prices), 2),
            "total_pnl":     round(self.total_pnl(prices), 2),
            "total_pnl_pct": round(self.total_pnl_pct(prices) * 100, 2),
            "realised_pnl":  round(self.realised_pnl(), 2),
            "win_rate":      round(self.win_rate() * 100, 2),
            "num_positions": len(self.positions),
            "num_trades":    len(self.trade_log),
            "positions": {
                t: {"shares": p.shares, "avg_price": p.avg_price,
                    "current_price": prices.get(t, p.avg_price),
                    "unrealised_pnl": round(p.unrealised_pnl(prices.get(t, p.avg_price)), 2),
                    "pnl_pct": round(p.pnl_pct(prices.get(t, p.avg_price)) * 100, 2),
                    "source": p.source}
                for t, p in self.positions.items()
            },
            "recent_trades": [t.to_dict() for t in self.trade_log[-20:]],
        }

    def save(self, path="portfolio.json"):
        with open(path, "w") as f: json.dump(self.snapshot(), f, indent=2)

    def summary_str(self, prices=None):
        s = self.snapshot(prices or {})
        return (f"Cash=${s['cash']:,.2f}  Portfolio=${s['total_value']:,.2f}  "
                f"PnL=${s['total_pnl']:+,.2f} ({s['total_pnl_pct']:+.1f}%)  "
                f"WinRate={s['win_rate']:.0f}%  Positions={s['num_positions']}/{self.max_positions}")
