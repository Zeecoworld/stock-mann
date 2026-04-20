"""
strategy_engine/sentiment_agent.py
Replicate-hosted LLM sentiment scorer.
Model: meta/meta-llama-3-70b-instruct (swap version string to upgrade)
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

DEFAULT_MODEL = "meta/meta-llama-3-70b-instruct"

_SYSTEM_PROMPT = """\
You are a financial sentiment analyst.
Given a list of news headlines/articles, return ONLY valid JSON — no prose, no markdown.

Schema:
{
  "score": <float between -1.0 (very bearish) and 1.0 (very bullish)>,
  "confidence": <float between 0.0 and 1.0>,
  "summary": "<one sentence explanation>"
}
"""

_USER_TEMPLATE = """\
Ticker: {ticker}
Market: {source}

Headlines:
{headlines}

Return JSON only.
"""


class SentimentAgent:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_token: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        token = api_token or os.getenv("REPLICATE_API_TOKEN")
        if not token:
            raise EnvironmentError("Set REPLICATE_API_TOKEN in your .env file.")
        self._client = replicate.Client(api_token=token)

    async def enrich(self, ctx: MarketContext) -> MarketContext:
        score, confidence, summary = await self._score(
            ticker=ctx.ticker, source=ctx.source, headlines=ctx.raw_news or [])
        ctx.sentiment_score      = score
        ctx.sentiment_confidence = confidence
        logger.info("[SentimentAgent] %s  score=%.3f  conf=%.3f  — %s",
                    ctx.ticker, score, confidence, summary)
        return ctx

    async def _score(self, ticker, source, headlines):
        headlines_text = (
            "\n".join(f"- {h}" for h in headlines)
            if headlines else "(No live headlines.)"
        )
        user_prompt = _USER_TEMPLATE.format(
            ticker=ticker, source=source, headlines=headlines_text)
        try:
            output = await self._run_replicate(user_prompt)
            return self._parse(output)
        except Exception as exc:
            logger.error("[SentimentAgent] LLM call failed: %s", exc)
            return 0.0, 0.0, "error"

    async def _run_replicate(self, user_prompt: str) -> str:
        def _sync():
            output = self._client.run(
                self.model,
                input={"system_prompt": _SYSTEM_PROMPT, "prompt": user_prompt,
                       "max_new_tokens": self.max_tokens, "temperature": self.temperature})
            return "".join(output) if hasattr(output, "__iter__") else str(output)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync)

    @staticmethod
    def _parse(raw: str):
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        try:
            d = json.loads(raw)
            s = max(-1.0, min(1.0, float(d.get("score", 0.0))))
            c = max(0.0,  min(1.0, float(d.get("confidence", 0.5))))
            return s, c, d.get("summary", "")
        except Exception as exc:
            logger.warning("[SentimentAgent] parse error (%s) raw=%r", exc, raw[:200])
            return 0.0, 0.0, "parse_error"
