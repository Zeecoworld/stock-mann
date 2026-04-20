
from __future__ import annotations
try:
    from strategy_engine.engine import BaseStrategy, MarketContext, Signal, TradeSignal
except ImportError:
    from engine import BaseStrategy, MarketContext, Signal, TradeSignal


class PolymarketSentimentStrategy(BaseStrategy):
    """
    Sentiment-driven strategy for Polymarket binary markets.

    Args:
        sentiment_threshold:    min |score| to act (default 0.30)
        confidence_threshold:   min agent confidence (default 0.50)
        fair_value_sensitivity: how aggressively score maps to fair value (default 0.4)
    """

    name = "polymarket"

    def __init__(
        self,
        sentiment_threshold: float = 0.30,
        confidence_threshold: float = 0.50,
        fair_value_sensitivity: float = 0.40,
    ):
        self.sentiment_threshold = sentiment_threshold
        self.confidence_threshold = confidence_threshold
        self.fair_value_sensitivity = fair_value_sensitivity

    async def evaluate(self, ctx: MarketContext) -> TradeSignal:
        score    = ctx.sentiment_score or 0.0
        conf     = ctx.sentiment_confidence or 0.0
        imp_prob = ctx.price           # 0–1 market-implied probability

        # FIX: safe display string for imp_prob — avoids printing "None" in reasoning
        market_str = f"{imp_prob:.3f}" if imp_prob is not None else "unknown"

        # ── guard ─────────────────────────────────────────────────────────
        if conf < self.confidence_threshold:
            return self._signal(
                Signal.SKIP, ctx, conf,
                f"Low agent confidence ({conf:.2f})"
            )

        # Estimated fair value given sentiment
        fair_value = 0.5 + (score * self.fair_value_sensitivity)
        fair_value = max(0.01, min(0.99, fair_value))

        # ── BUY YES ───────────────────────────────────────────────────────
        if score >= self.sentiment_threshold:
            if imp_prob is None or imp_prob < fair_value:
                edge = (fair_value - imp_prob) if imp_prob is not None else None
                edge_str = f", edge={edge:.3f}" if edge is not None else ""
                return self._signal(
                    Signal.BUY, ctx, conf,
                    f"Positive sentiment ({score:.2f}); fair={fair_value:.3f} "
                    f"vs market={market_str}{edge_str} → BUY YES"
                )

        # ── BUY NO ────────────────────────────────────────────────────────
        if score <= -self.sentiment_threshold:
            no_fair = 1.0 - fair_value
            if imp_prob is None or imp_prob > no_fair:
                edge = (imp_prob - no_fair) if imp_prob is not None else None
                edge_str = f", edge={edge:.3f}" if edge is not None else ""
                # Signal.SELL == "take NO side" in Polymarket context
                return self._signal(
                    Signal.SELL, ctx, conf,
                    f"Negative sentiment ({score:.2f}); fair_NO={no_fair:.3f} "
                    f"vs market={market_str}{edge_str} → BUY NO"
                )

        return self._signal(
            Signal.HOLD, ctx, conf,
            f"No edge: sentiment={score:.2f}, implied={market_str}, fair={fair_value:.3f}"
        )