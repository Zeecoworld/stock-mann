"""
strategy_engine/strategies/stock_strategy.py
Strategy for traditional stock/equity bots.

Logic:
  BUY  if sentiment >= sentiment_threshold AND price > ma_20
  SELL if sentiment <= -sentiment_threshold AND price < ma_20
  HOLD in all other cases (including bearish-but-above-MA and bullish-but-below-MA)

FIX (2025-04):
  - Bearish sentiment + price still above MA20 now returns an explicit HOLD with
    a descriptive reason instead of silently falling through to the catch-all HOLD
    with the misleading "Sentiment neutral" message.
  - Same symmetric fix for bullish sentiment + price below MA20.
"""
from __future__ import annotations
try:
    from strategy_engine.engine import BaseStrategy, MarketContext, Signal, TradeSignal
except ImportError:
    from engine import BaseStrategy, MarketContext, Signal, TradeSignal


class StockSentimentStrategy(BaseStrategy):
    """
    Combines LLM sentiment + 20-day MA crossover.

    Args:
        sentiment_threshold:  minimum |score| to act (default 0.3)
        confidence_threshold: minimum agent confidence to act (default 0.5)
    """

    name = "stock"

    def __init__(
        self,
        sentiment_threshold: float = 0.30,
        confidence_threshold: float = 0.50,
    ):
        self.sentiment_threshold = sentiment_threshold
        self.confidence_threshold = confidence_threshold

    async def evaluate(self, ctx: MarketContext) -> TradeSignal:
        score = ctx.sentiment_score or 0.0
        conf  = ctx.sentiment_confidence or 0.0
        price = ctx.price
        ma_20 = ctx.ma_20

        # ── guard: not enough confidence ─────────────────────────────────
        if conf < self.confidence_threshold:
            return self._signal(
                Signal.SKIP, ctx, conf,
                f"Low agent confidence ({conf:.2f} < {self.confidence_threshold})"
            )

        has_price_data = price is not None and ma_20 is not None

        # ── BUY ───────────────────────────────────────────────────────────
        if score >= self.sentiment_threshold:
            if has_price_data and price > ma_20:
                return self._signal(
                    Signal.BUY, ctx, conf,
                    f"Bullish sentiment ({score:.2f}) + price {price:.2f} above MA20 {ma_20:.2f}"
                )
            if has_price_data and price <= ma_20:
                # FIX: bullish sentiment but price hasn't broken above MA yet — no entry
                return self._signal(
                    Signal.HOLD, ctx, conf * 0.9,
                    f"Bullish sentiment ({score:.2f}) but price {price:.2f} "
                    f"not yet above MA20 {ma_20:.2f} — waiting for confirmation"
                )
            if not has_price_data:
                return self._signal(
                    Signal.BUY, ctx, conf * 0.8,
                    f"Bullish sentiment ({score:.2f}); no price data for MA check"
                )

        # ── SELL ──────────────────────────────────────────────────────────
        if score <= -self.sentiment_threshold:
            if has_price_data and price < ma_20:
                return self._signal(
                    Signal.SELL, ctx, conf,
                    f"Bearish sentiment ({score:.2f}) + price {price:.2f} below MA20 {ma_20:.2f}"
                )
            if has_price_data and price >= ma_20:
                # FIX: bearish sentiment but price hasn't broken down yet — no entry
                return self._signal(
                    Signal.HOLD, ctx, conf * 0.9,
                    f"Bearish sentiment ({score:.2f}) but price {price:.2f} "
                    f"still above MA20 {ma_20:.2f} — waiting for confirmation"
                )
            if not has_price_data:
                return self._signal(
                    Signal.SELL, ctx, conf * 0.8,
                    f"Bearish sentiment ({score:.2f}); no price data for MA check"
                )

        # ── HOLD (neutral sentiment) ──────────────────────────────────────
        return self._signal(
            Signal.HOLD, ctx, conf,
            f"Neutral sentiment ({score:.2f}) — no actionable edge"
        )