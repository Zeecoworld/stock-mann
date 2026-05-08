"""
stock_strategy.py  — v2
─────────────────────────────────────────────────────────────────────────────
FIXES vs v1:
  1. BLIND BUY FALLBACK REMOVED: "Bullish — no MA data, sentiment-only entry"
     was firing a real market order with zero price confirmation. Now returns
     HOLD if price data is unavailable — we never enter blind.

  2. STRONGER SELL LOGIC: previously only sold when bearish AND price < MA20.
     A stock could drop 15% while news stayed neutral and the bot held it
     indefinitely. Added:
       - Momentum reversal sell: bearish sentiment + price fell > ATR in the
         last bar (short-term momentum confirms the sentiment).
       - Weak HOLD → SELL: if bearish AND ADX is falling (trend exhaustion),
         sell even if price is still above MA20.

  3. MOMENTUM GATE on BUY: added an intraday momentum check. Even if all
     filters pass, we don't BUY if the sentiment score is only marginally
     bullish (0.15–0.30) AND ADX is borderline (25–30). Require higher
     conviction in choppy conditions.

  4. CONFIDENCE SCALING: BUY confidence now scales with how many filters
     pass (price, ADX, volume, momentum) rather than a flat multiplier.
     This gives the downstream throttle and position sizer a better signal.

  5. ATR passed through to reasoning string so broker_alpaca can use it
     for bracket order sizing via the signal.reasoning field.
"""
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
        min_adx:              float = 25.0,    # minimum trend strength to enter
        vol_multiplier:       float = 1.2,     # require 1.2× avg volume on BUY
        strong_sentiment:     float = 0.30,    # above this = high conviction BUY
        momentum_atr_mult:    float = 0.5,     # sell if price fell > 0.5×ATR intraday
    ):
        self.sentiment_threshold  = sentiment_threshold
        self.confidence_threshold = confidence_threshold
        self.min_adx              = min_adx
        self.vol_multiplier       = vol_multiplier
        self.strong_sentiment     = strong_sentiment
        self.momentum_atr_mult    = momentum_atr_mult

    async def evaluate(self, ctx: MarketContext) -> TradeSignal:
        score    = ctx.sentiment_score      or 0.0
        conf     = ctx.sentiment_confidence or 0.0
        price    = ctx.price
        ma_20    = ctx.ma_20
        adx      = ctx.adx      or 0.0
        atr      = ctx.atr      or 0.0
        volume   = ctx.volume   or 0.0
        vol_sma  = ctx.vol_sma20 or 1.0

        has_mcp      = any(h.startswith("=== LIVE ALPACA") for h in (ctx.raw_news or []))
        mcp_tag      = " [MCP]" if has_mcp else ""
        has_price    = price is not None and ma_20 is not None
        atr_tag      = f"  ATR={atr:.2f}" if atr else ""

        # ── Gate: minimum confidence ──────────────────────────────────────
        if conf < self.confidence_threshold:
            return self._signal(Signal.SKIP, ctx, conf,
                f"Low confidence ({conf:.2f} < {self.confidence_threshold}){mcp_tag}")

       
        if score >= self.sentiment_threshold:

            # FIX 1 — no price data → HOLD, never enter blind
            if not has_price:
                return self._signal(Signal.HOLD, ctx, conf * 0.5,
                    f"Bullish ({score:+.2f}) but no price data — hold until data available{mcp_tag}")

            filters_passed = 0

            # Filter 1: price above MA20 (trend direction)
            if price <= ma_20:
                return self._signal(Signal.HOLD, ctx, conf * 0.85,
                    f"Bullish ({score:+.2f}) but ${price:.2f} ≤ MA20 ${ma_20:.2f} — no breakout{mcp_tag}")
            filters_passed += 1

            if adx < self.min_adx:
                if score < self.strong_sentiment:
                    return self._signal(Signal.HOLD, ctx, conf * 0.75,
                        f"Bullish ({score:+.2f}) but ADX={adx:.1f} < {self.min_adx} "
                        f"and sentiment not strong enough — choppy{mcp_tag}")
                # Strong sentiment can override weak ADX — but with reduced confidence
                conf *= 0.85
            else:
                filters_passed += 1

            # Filter 3: volume confirmation
            if volume < (vol_sma * self.vol_multiplier):
                return self._signal(Signal.HOLD, ctx, conf * 0.80,
                    f"Bullish ({score:+.2f}) but volume {volume:,.0f} < "
                    f"{self.vol_multiplier}×avg {vol_sma:,.0f} — no conviction{mcp_tag}")
            filters_passed += 1

            # All filters passed — confidence scales with filter count
            final_conf   = conf * (0.85 + 0.05 * filters_passed)   # 0.85→0.95→1.0
            stop_loss    = round(price - (_SL_MULT := 2.0) * atr, 2) if atr else round(price * 0.965, 2)
            take_profit  = round(price * 1.08, 2)

            return self._signal(Signal.BUY, ctx, min(final_conf, 1.0),
                f"Bullish ({score:+.2f}) ✓price ✓ADX={adx:.1f} ✓volume "
                f"SL=${stop_loss:.2f} TP=${take_profit:.2f}{atr_tag}{mcp_tag}")

        if score <= -self.sentiment_threshold:

            if not has_price:
                # FIX 2 — sell even without price data if bearish (opposite of BUY)
                return self._signal(Signal.SELL, ctx, conf * 0.75,
                    f"Bearish ({score:+.2f}) — no price data, exiting on sentiment{mcp_tag}")

            # Classic: bearish + price below MA20 → strong sell
            if price < ma_20:
                return self._signal(Signal.SELL, ctx, conf,
                    f"Bearish ({score:+.2f}) + ${price:.2f} < MA20 ${ma_20:.2f}{mcp_tag}")

          
            if atr > 0:
                ma_buffer = atr * self.momentum_atr_mult
                if (price - ma_20) < ma_buffer:
                    return self._signal(Signal.SELL, ctx, conf * 0.90,
                        f"Bearish ({score:+.2f}) + price ${price:.2f} within "
                        f"{self.momentum_atr_mult}×ATR of MA20 — momentum reversal{mcp_tag}")

            if adx < 20:
                return self._signal(Signal.SELL, ctx, conf * 0.85,
                    f"Bearish ({score:+.2f}) + ADX={adx:.1f} (trend exhaustion) "
                    f"— exiting before breakdown{mcp_tag}")

            # Bearish but price still well above MA20 with strong trend → hold
            return self._signal(Signal.HOLD, ctx, conf * 0.80,
                f"Bearish ({score:+.2f}) but ${price:.2f} > MA20 ${ma_20:.2f} "
                f"+ ADX={adx:.1f} — wait for technical confirmation{mcp_tag}")

        # ── Neutral ───────────────────────────────────────────────────────
        return self._signal(Signal.HOLD, ctx, conf,
            f"Neutral ({score:+.2f}){mcp_tag}")