"""strategy_engine/engine.py — core data structures and orchestrator."""
from __future__ import annotations
import asyncio, logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class Signal(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    SKIP = "SKIP"


@dataclass
class MarketContext:
    ticker: str
    source: str                          
    price:  Optional[float] = None
    ma_20:  Optional[float] = None
    adx:    Optional[float] = None          
    atr:    Optional[float] = None          
    volume: Optional[float] = None         
    vol_sma20: Optional[float] = None 
    raw_news: List[str]     = field(default_factory=list)
    sentiment_score:      Optional[float] = None
    sentiment_confidence: Optional[float] = None


@dataclass
class TradeSignal:
    ticker:     str
    source:     str
    signal:     Signal
    confidence: float
    reasoning:  str


class SignalBus:
    def __init__(self):
        self._subs: List[Callable[[TradeSignal], None]] = []

    def subscribe(self, fn):
        self._subs.append(fn)

    def publish(self, sig: TradeSignal):
        for fn in self._subs:
            try: fn(sig)
            except Exception as e: logger.warning("[Bus] %s", e)


class BaseStrategy:
    name: str = "base"

    async def evaluate(self, ctx: MarketContext) -> TradeSignal:
        raise NotImplementedError

    def _signal(self, sig, ctx, conf, reason) -> TradeSignal:
        return TradeSignal(ticker=ctx.ticker, source=ctx.source,
                           signal=sig, confidence=conf, reasoning=reason)


class StrategyEngine:
    def __init__(self, bus: SignalBus, sentiment_agent=None):
        self.bus   = bus
        self.agent = sentiment_agent
        self._strategies: List[BaseStrategy] = []

    def register(self, s: BaseStrategy):
        self._strategies.append(s)
        logger.info("[Engine] registered: %s", s.name)

    async def run(self, ctx: MarketContext) -> Optional[TradeSignal]:
        if self.agent:
            ctx = await self.agent.enrich(ctx)
        for s in self._strategies:
            if s.name != ctx.source:
                continue
            try:
                sig = await s.evaluate(ctx)
                self.bus.publish(sig)
                return sig
            except Exception as e:
                logger.error("[Engine] %s failed: %s", s.name, e)
        return None