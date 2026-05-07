from __future__ import annotations
try:
    from .engine import BaseStrategy, MarketContext, Signal, TradeSignal
except ImportError:
    from engine import BaseStrategy, MarketContext, Signal, TradeSignal

class StockSentimentStrategy(BaseStrategy):
    name = "stock"

    def __init__(
        self,
        sentiment_threshold:  float = 0.15,
        confidence_threshold: float = 0.35,
        min_adx: float = 25.0,         # Minimum trend strength
        vol_multiplier: float = 1.2    # Require 1.2x average volume to confirm breakout
    ):
        self.sentiment_threshold  = sentiment_threshold
        self.confidence_threshold = confidence_threshold
        self.min_adx = min_adx
        self.vol_multiplier = vol_multiplier

    async def evaluate(self, ctx: MarketContext) -> TradeSignal:
        score = ctx.sentiment_score      or 0.0
        conf  = ctx.sentiment_confidence or 0.0
        price = ctx.price
        ma_20 = ctx.ma_20

        # NEW: Technical Indicators
        adx = ctx.adx or 0.0
        atr = ctx.atr or 0.0
        volume = ctx.volume or 0.0
        vol_sma = ctx.vol_sma20 or 1.0

        has_mcp = any(h.startswith("=== LIVE ALPACA") for h in (ctx.raw_news or []))
        mcp_tag = " [MCP-enriched]" if has_mcp else ""

       
        if conf < self.confidence_threshold:
            return self._signal(Signal.SKIP, ctx, conf, f"Low confidence ({conf:.2f} < {self.confidence_threshold}){mcp_tag}")

        has_price_data = price is not None and ma_20 is not None

    
        if score >= self.sentiment_threshold:
            if has_price_data:
                # 1. Base Price Condition
                if price <= ma_20:
                    return self._signal(Signal.HOLD, ctx, conf * 0.9,
                        f"Bullish but ${price:.2f} <= MA20 ${ma_20:.2f} — waiting for breakout{mcp_tag}")

                # 2. Chop Filter (ADX)
                if adx < self.min_adx:
                    return self._signal(Signal.HOLD, ctx, conf * 0.8,
                        f"Bullish but ADX is low ({adx:.1f} < {self.min_adx}). Market is choppy{mcp_tag}")

                # 3. Volume Confirmation
                if volume < (vol_sma * self.vol_multiplier):
                    return self._signal(Signal.HOLD, ctx, conf * 0.8,
                        f"Bullish but weak volume. Needs {self.vol_multiplier}x average{mcp_tag}")

                # Passed all filters! Calculate ATR Stop Loss
                stop_loss = price - (atr * 2.0)
                
                return self._signal(
                    Signal.BUY, ctx, conf,
                    f"Bullish ({score:+.2f}) + Breakout confirmed! ADX={adx:.1f}, Vol Surge. Dynamic StopLoss: ${stop_loss:.2f}{mcp_tag}"
                )

            # Fallback if Yahoo Finance fails
            return self._signal(Signal.BUY, ctx, conf * 0.8, f"Bullish ({score:+.2f}) — no MA data, sentiment-only entry{mcp_tag}")


        
        if score <= -self.sentiment_threshold:
            if has_price_data and price < ma_20:
                return self._signal(Signal.SELL, ctx, conf, f"Bearish ({score:+.2f}) + price ${price:.2f} < MA20 ${ma_20:.2f}{mcp_tag}")
            if has_price_data and price >= ma_20:
                return self._signal(Signal.HOLD, ctx, conf * 0.9, f"Bearish ({score:+.2f}) but ${price:.2f} >= MA20 ${ma_20:.2f}{mcp_tag}")
            
            return self._signal(Signal.SELL, ctx, conf * 0.8, f"Bearish ({score:+.2f}) — no MA data{mcp_tag}")

        return self._signal(Signal.HOLD, ctx, conf, f"Neutral score ({score:+.2f}){mcp_tag}")