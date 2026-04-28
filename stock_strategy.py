
from __future__ import annotations
try:
    from .engine import BaseStrategy, MarketContext, Signal, TradeSignal
except ImportError:
    from engine import BaseStrategy, MarketContext, Signal, TradeSignal


class StockSentimentStrategy(BaseStrategy):

    name = "stock"

    def __init__(
        self,
        sentiment_threshold:  float = 0.15,   # FIX: was 0.30
        confidence_threshold: float = 0.35,   # FIX: was 0.50
    ):
        self.sentiment_threshold  = sentiment_threshold
        self.confidence_threshold = confidence_threshold

    async def evaluate(self, ctx: MarketContext) -> TradeSignal:
        score = ctx.sentiment_score      or 0.0
        conf  = ctx.sentiment_confidence or 0.0
        price = ctx.price
        ma_20 = ctx.ma_20

        # Detect if Alpaca MCP context was injected
        has_mcp = any(
            h.startswith("=== LIVE ALPACA")
            for h in (ctx.raw_news or [])
        )
        mcp_tag = " [MCP-enriched]" if has_mcp else ""

        # ── guard: confidence too low ─────────────────────────────────────
        if conf < self.confidence_threshold:
            return self._signal(
                Signal.SKIP, ctx, conf,
                f"Low confidence ({conf:.2f} < {self.confidence_threshold}){mcp_tag}"
                f" — score={score:+.2f}"
            )

        has_price_data = price is not None and ma_20 is not None

        # ── BUY ───────────────────────────────────────────────────────────
        if score >= self.sentiment_threshold:
            if has_price_data and price > ma_20:
                return self._signal(
                    Signal.BUY, ctx, conf,
                    f"Bullish ({score:+.2f}) + price ${price:.2f} > MA20 ${ma_20:.2f}{mcp_tag}"
                )
            if has_price_data and price <= ma_20:
                return self._signal(
                    Signal.HOLD, ctx, conf * 0.9,
                    f"Bullish ({score:+.2f}) but ${price:.2f} <= MA20 ${ma_20:.2f}"
                    f" — waiting for breakout{mcp_tag}"
                )
            # No price data — still act on sentiment alone
            return self._signal(
                Signal.BUY, ctx, conf * 0.8,
                f"Bullish ({score:+.2f}) — no MA data, sentiment-only entry{mcp_tag}"
            )

        # ── SELL ──────────────────────────────────────────────────────────
        if score <= -self.sentiment_threshold:
            if has_price_data and price < ma_20:
                return self._signal(
                    Signal.SELL, ctx, conf,
                    f"Bearish ({score:+.2f}) + price ${price:.2f} < MA20 ${ma_20:.2f}{mcp_tag}"
                )
            if has_price_data and price >= ma_20:
                return self._signal(
                    Signal.HOLD, ctx, conf * 0.9,
                    f"Bearish ({score:+.2f}) but ${price:.2f} >= MA20 ${ma_20:.2f}"
                    f" — waiting for breakdown{mcp_tag}"
                )
            return self._signal(
                Signal.SELL, ctx, conf * 0.8,
                f"Bearish ({score:+.2f}) — no MA data, sentiment-only exit{mcp_tag}"
            )

        return self._signal(
            Signal.HOLD, ctx, conf,
            f"Neutral score ({score:+.2f}) within ±{self.sentiment_threshold}"
            f" threshold{mcp_tag}"
        )