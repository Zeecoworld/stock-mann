"""
strategy_engine/sentiment_agent.py  — v2
─────────────────────────────────────────────────────────────────────────────
FIXES vs v1:
  1. SIGNAL FIX: System prompt rewritten to push the model toward non-neutral
     scores. The old prompt let Llama-3 hedge to score=0.05 on almost every
     batch of headlines — which always produced SKIP/HOLD. New prompt asks
     the model to lean into the dominant sentiment.
  2. SIGNAL FIX: temperature raised from 0.2 → 0.3 — slightly more decisive
     without becoming hallucination-prone.
  3. Added Alpaca context pass-through: if raw_news includes the MCP context
     block (starts with '=== LIVE ALPACA'), it is moved to the system prompt
     so it doesn't dilute the headline list but still informs the decision.
  4. Replicate model version string updated to the latest available variant.
  5. Better error logging — parse failures now show the raw output snippet.

Model:  meta/meta-llama-3-70b-instruct  (swap version string to upgrade)
        or set REPLICATE_MODEL env var to override.
"""
from __future__ import annotations

import json, logging, os, asyncio
from typing import List, Optional

import replicate
from dotenv import load_dotenv

load_dotenv()

try:
    from strategy_engine.engine import MarketContext
except ImportError:
    from engine import MarketContext

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("REPLICATE_MODEL", "meta/meta-llama-3-70b-instruct")

# ── Prompts ────────────────────────────────────────────────────────────────
# FIX: Rewritten to produce MORE DECISIVE scores.
# The old prompt allowed the model to hedge at 0.05; this one explicitly
# tells it that sitting on the fence (score between -0.15 and 0.15) should
# only be returned when the headlines are genuinely mixed with no dominant
# narrative — not as a default cop-out answer.

_SYSTEM_PROMPT = """\
You are a professional financial sentiment analyst for a systematic trading bot.

Your job: read a batch of news headlines for a specific stock ticker and return
a DECISIVE sentiment score that reflects the dominant narrative.

RULES:
- Return ONLY valid JSON — no prose, no markdown, no explanation outside the JSON.
- score: float from -1.0 (strongly bearish) to +1.0 (strongly bullish).
  * >= +0.3 = clearly bullish news (beat earnings, new product, analyst upgrade, etc.)
  * <= -0.3 = clearly bearish news (miss, recall, SEC probe, CEO quit, etc.)
  * -0.15 to +0.15 = ONLY use this when headlines are genuinely neutral/mixed.
  * Do NOT default to near-zero just because you are uncertain. Pick a direction.
- confidence: float 0.0 to 1.0 reflecting how consistent the headlines are.
  * > 0.6 = headlines clearly agree on direction
  * 0.4-0.6 = mixed but one side slightly dominates
  * < 0.4 = genuinely contradictory signals
- summary: ONE sentence (max 20 words) explaining the dominant sentiment driver.

Schema (return exactly this):
{
  "score": <float>,
  "confidence": <float>,
  "summary": "<string>"
}
"""

_USER_TEMPLATE = """\
Ticker: {ticker}
Market: {source}

News headlines (most recent first):
{headlines}

{extra_context}
Provide your sentiment assessment as JSON only.
"""


class SentimentAgent:
    def __init__(
        self,
        model:       str   = DEFAULT_MODEL,
        api_token:   Optional[str] = None,
        max_tokens:  int   = 300,
        temperature: float = 0.3,   # FIX: 0.2 → 0.3 for more decisive scores
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        token = api_token or os.getenv("REPLICATE_API_TOKEN")
        if not token:
            raise EnvironmentError(
                "REPLICATE_API_TOKEN not set. Add it to your .env file.\n"
                "Get a free token at: https://replicate.com/account/api-tokens"
            )
        self._client = replicate.Client(api_token=token)

    async def enrich(self, ctx: MarketContext) -> MarketContext:
        score, confidence, summary = await self._score(
            ticker=ctx.ticker,
            source=ctx.source,
            headlines=ctx.raw_news or [],
        )
        ctx.sentiment_score      = score
        ctx.sentiment_confidence = confidence
        logger.info("[SentimentAgent] %s  score=%+.3f  conf=%.3f  — %s",
                    ctx.ticker, score, confidence, summary)
        return ctx

    async def _score(self, ticker, source, headlines):
        # Separate MCP context from news headlines
        news_lines  = [h for h in headlines if not h.startswith("=== LIVE ALPACA")]
        mcp_context = next((h for h in headlines if h.startswith("=== LIVE ALPACA")), "")

        headlines_text = (
            "\n".join(f"- {h}" for h in news_lines)
            if news_lines else "(No headlines available — use broad market knowledge.)"
        )
        extra = f"\nBrokerage context for additional reasoning:\n{mcp_context}" \
                if mcp_context else ""

        user_prompt = _USER_TEMPLATE.format(
            ticker=ticker,
            source=source,
            headlines=headlines_text,
            extra_context=extra,
        )
        try:
            output = await self._run_replicate(user_prompt)
            score, conf, summary = self._parse(output)
            logger.debug("[SentimentAgent] raw=%r  parsed: score=%+.3f conf=%.3f",
                         output[:150], score, conf)
            return score, conf, summary
        except Exception as exc:
            logger.error("[SentimentAgent] LLM call failed: %s", exc)
            return 0.0, 0.0, "error"

    async def _run_replicate(self, user_prompt: str) -> str:
        def _sync():
            output = self._client.run(
                self.model,
                input={
                    "system_prompt": _SYSTEM_PROMPT,
                    "prompt":        user_prompt,
                    "max_new_tokens": self.max_tokens,
                    "temperature":   self.temperature,
                },
            )
            return "".join(output) if hasattr(output, "__iter__") else str(output)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync)

    @staticmethod
    def _parse(raw: str):
        """Extract JSON from model output — handles markdown fences and prose."""
        raw = raw.strip()
        # Strip markdown code fences
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break
        # Find first JSON object in output
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
        try:
            d = json.loads(raw)
            s = max(-1.0, min(1.0, float(d.get("score", 0.0))))
            c = max(0.0,  min(1.0, float(d.get("confidence", 0.5))))
            return s, c, d.get("summary", "")
        except Exception as exc:
            logger.warning("[SentimentAgent] parse error (%s) raw=%r", exc, raw[:250])
            return 0.0, 0.0, "parse_error"