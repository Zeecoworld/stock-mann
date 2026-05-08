
import http
import asyncio
import json
import logging
import os
import re
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Optional
import xml.etree.ElementTree as ET

import aiohttp
import numpy as np
import pandas as pd
import requests
import websockets
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.models import Position

# ─── Configuration ──────────────────────────────────────────────────────────
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING,
    WATCHLIST, RSS_FEEDS, RISK_PER_TRADE, MAX_PORTFOLIO_RISK,
    MAX_POSITIONS, TRADE_INTERVAL, NEWS_SENTIMENT_WEIGHT,
    TECHNICAL_WEIGHT, RSI_OVERSOLD, RSI_OVERBOUGHT, WS_PORT
)

# ─── Logging Setup ───────────────────────────────────────────────────────────
# Stdout-only logging — no log files written to disk.
# On Render, all stdout is captured in the native Log dashboard.
import sys

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
handler.setStream(sys.stdout)  # force stdout (not stderr) for Render log grouping

logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
log = logging.getLogger("AlphaSentinel")


# ══════════════════════════════════════════════════════════════════════════════
# NEWS ENGINE — RSS Fetcher + Sentiment Analyzer
# ══════════════════════════════════════════════════════════════════════════════

POSITIVE_KEYWORDS = {
    "surge", "soar", "rally", "gain", "profit", "beat", "record", "growth",
    "outperform", "upgrade", "buy", "strong", "bullish", "rise", "jump",
    "breakthrough", "expansion", "revenue", "earnings beat", "dividend",
    "innovation", "partnership", "acquisition", "launch", "approval", "win",
    "milestone", "recovery", "upside", "momentum", "higher", "increase",
}

NEGATIVE_KEYWORDS = {
    "crash", "plunge", "fall", "loss", "miss", "downgrade", "sell", "weak",
    "bearish", "decline", "cut", "layoff", "lawsuit", "fine", "investigation",
    "recall", "shortage", "debt", "bankruptcy", "fraud", "warning", "risk",
    "concern", "volatility", "drop", "lower", "decrease", "headwinds",
    "disappointing", "miss", "restructuring", "hack", "breach", "delay",
}

TICKER_ALIASES = {
    "apple": "AAPL", "aapl": "AAPL",
    "nvidia": "NVDA", "nvda": "NVDA",
    "microsoft": "MSFT", "msft": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL",
    "amazon": "AMZN", "amzn": "AMZN",
    "tesla": "TSLA", "tsla": "TSLA",
    "meta": "META", "facebook": "META",
    "netflix": "NFLX", "nflx": "NFLX",
    "amd": "AMD", "advanced micro": "AMD",
    "intel": "INTC", "intc": "INTC",
    "salesforce": "CRM", "crm": "CRM",
    "paypal": "PYPL", "pypl": "PYPL",
    "shopify": "SHOP", "shop": "SHOP",
    "qualcomm": "QCOM", "qcom": "QCOM",
    "broadcom": "AVGO", "avgo": "AVGO",
    "palantir": "PLTR", "pltr": "PLTR",
    "coinbase": "COIN", "coin": "COIN",
    "jpmorgan": "JPM", "jpm": "JPM",
    "goldman": "GS", "gs": "GS",
    "s&p": "SPY", "spy": "SPY",
}


class NewsEngine:
    """Fetches RSS feeds and scores sentiment for ticker relevance."""

    def __init__(self):
        self.news_cache: deque = deque(maxlen=500)
        self.ticker_sentiment: dict[str, list[float]] = defaultdict(list)
        self.last_fetch: dict[str, float] = {}
        self._lock = threading.Lock()

    def fetch_feed(self, url: str, timeout: int = 10) -> list[dict]:
        """Parse an RSS/Atom feed and return article dicts."""
        articles = []
        try:
            headers = {"User-Agent": "Mozilla/5.0 AlphaSentinel/3.1 (Trading Bot)"}
            resp = requests.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            # Support both RSS 2.0 and Atom
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//item") or root.findall(".//atom:entry", ns)

            for item in items[:20]:  # Cap per feed
                title_el = item.find("title") or item.find("atom:title", ns)
                link_el = item.find("link") or item.find("atom:link", ns)
                pubdate_el = (item.find("pubDate") or item.find("published")
                              or item.find("atom:published", ns))
                desc_el = (item.find("description") or item.find("summary")
                           or item.find("atom:summary", ns))

                title = title_el.text.strip() if title_el is not None and title_el.text else ""
                link = ""
                if link_el is not None:
                    link = link_el.text or link_el.get("href", "")
                pub_date = pubdate_el.text.strip() if pubdate_el is not None and pubdate_el.text else ""
                desc = desc_el.text or "" if desc_el is not None else ""
                desc = re.sub(r"<[^>]+>", "", desc)[:300]  # Strip HTML

                if title:
                    articles.append({
                        "title": title, "link": link,
                        "published": pub_date, "description": desc,
                        "source": url
                    })
        except Exception as e:
            log.warning(f"RSS fetch failed [{url}]: {e}")
        return articles

    def score_sentiment(self, text: str) -> float:
        text_lower = text.lower()
        pos = sum(1.5 if kw in text_lower else 0 for kw in POSITIVE_KEYWORDS
                  if kw in text_lower)
        neg = sum(1.5 if kw in text_lower else 0 for kw in NEGATIVE_KEYWORDS
                  if kw in text_lower)

        # Context modifiers
        if any(w in text_lower for w in ["not", "no ", "doesn't", "failed to"]):
            pos, neg = neg * 0.5, pos * 0.5 + neg * 0.5  # Flip partial

        total = pos + neg
        if total == 0:
            return 0.0
        score = (pos - neg) / total
        return max(-1.0, min(1.0, score))

    def extract_tickers(self, text: str) -> list[str]:
        """Find mentioned tickers/companies in article text."""
        text_lower = text.lower()
        found = set()

        # Direct $TICKER pattern
        for match in re.finditer(r"\$([A-Z]{1,5})\b", text):
            ticker = match.group(1)
            if ticker in {s: s for s in WATCHLIST}:
                found.add(ticker)

        # Company name lookup
        for alias, ticker in TICKER_ALIASES.items():
            if alias in text_lower and ticker in WATCHLIST:
                found.add(ticker)

        # Uppercase ticker pattern (e.g., "AAPL", "NVDA")
        for match in re.finditer(r"\b([A-Z]{2,5})\b", text):
            t = match.group(1)
            if t in WATCHLIST:
                found.add(t)

        return list(found)

    def refresh_all_feeds(self) -> int:
        """Fetch all RSS feeds and update sentiment scores."""
        new_count = 0
        all_articles = []

        for feed_url in RSS_FEEDS:
            now = time.time()
            if now - self.last_fetch.get(feed_url, 0) < 300:  # 5 min cooldown
                continue
            articles = self.fetch_feed(feed_url)
            all_articles.extend(articles)
            self.last_fetch[feed_url] = now

        with self._lock:
            for art in all_articles:
                full_text = f"{art['title']} {art.get('description', '')}"
                sentiment = self.score_sentiment(full_text)
                tickers = self.extract_tickers(full_text)

                art["sentiment"] = round(sentiment, 4)
                art["tickers"] = tickers
                art["timestamp"] = datetime.now(timezone.utc).isoformat()

                self.news_cache.appendleft(art)

                for ticker in tickers:
                    self.ticker_sentiment[ticker].append(sentiment)
                    # Rolling window — keep last 20 scores per ticker
                    self.ticker_sentiment[ticker] = self.ticker_sentiment[ticker][-20:]

                new_count += 1

        log.info(f"[NewsEngine] Refreshed — {new_count} articles processed")
        return new_count

    def get_ticker_score(self, ticker: str) -> float:
        """Return a smoothed sentiment score for a ticker (-1 to +1)."""
        with self._lock:
            scores = self.ticker_sentiment.get(ticker, [])
        if not scores:
            return 0.0
        # Exponentially weighted — recent news matters more
        weights = np.exp(np.linspace(0, 1, len(scores)))
        weighted_avg = np.average(scores, weights=weights)
        return round(float(weighted_avg), 4)

    def get_recent_news(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self.news_cache)[:limit]


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class TechnicalAnalysis:
    """Computes RSI, MACD, Bollinger Bands, EMAs, ATR for decision signals."""

    @staticmethod
    def ema(series: np.ndarray, period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1)
        result = np.zeros_like(series, dtype=float)
        result[0] = series[0]
        for i in range(1, len(series)):
            result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def rsi(closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    @staticmethod
    def macd(closes: np.ndarray,
             fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        if len(closes) < slow + signal:
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "neutral"}
        ema_fast = TechnicalAnalysis.ema(closes, fast)
        ema_slow = TechnicalAnalysis.ema(closes, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalAnalysis.ema(macd_line, signal)
        histogram = macd_line - signal_line
        trend = "bullish" if macd_line[-1] > signal_line[-1] else "bearish"
        return {
            "macd": round(float(macd_line[-1]), 4),
            "signal": round(float(signal_line[-1]), 4),
            "histogram": round(float(histogram[-1]), 4),
            "trend": trend
        }

    @staticmethod
    def bollinger_bands(closes: np.ndarray,
                        period: int = 20, std_dev: float = 2.0) -> dict:
        if len(closes) < period:
            return {"upper": 0, "middle": 0, "lower": 0, "position": 0.5}
        window = closes[-period:]
        middle = float(np.mean(window))
        std = float(np.std(window))
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        current = float(closes[-1])
        band_width = upper - lower
        position = (current - lower) / band_width if band_width > 0 else 0.5
        return {
            "upper": round(upper, 4), "middle": round(middle, 4),
            "lower": round(lower, 4), "position": round(position, 4),
            "current": round(current, 4)
        }

    @staticmethod
    def atr(highs: np.ndarray, lows: np.ndarray,
            closes: np.ndarray, period: int = 14) -> float:
        """Average True Range — used for position sizing."""
        if len(highs) < period + 1:
            return float(closes[-1]) * 0.02
        trs = []
        for i in range(1, period + 1):
            h, l, prev_c = highs[-i], lows[-i], closes[-i - 1]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)
        return round(float(np.mean(trs)), 4)

    @staticmethod
    def sma_crossover(closes: np.ndarray,
                      fast: int = 9, slow: int = 21) -> dict:
        if len(closes) < slow:
            return {"signal": "neutral", "fast_sma": 0, "slow_sma": 0}
        fast_sma = float(np.mean(closes[-fast:]))
        slow_sma = float(np.mean(closes[-slow:]))
        if fast_sma > slow_sma * 1.001:
            sig = "bullish"
        elif fast_sma < slow_sma * 0.999:
            sig = "bearish"
        else:
            sig = "neutral"
        return {"signal": sig,
                "fast_sma": round(fast_sma, 4),
                "slow_sma": round(slow_sma, 4)}

    def full_analysis(self, bars_df: pd.DataFrame) -> dict:
        """Run all indicators on OHLCV dataframe."""
        closes = bars_df["close"].values.astype(float)
        highs = bars_df["high"].values.astype(float)
        lows = bars_df["low"].values.astype(float)
        volumes = bars_df["volume"].values.astype(float)

        rsi_val = self.rsi(closes)
        macd_data = self.macd(closes)
        bb_data = self.bollinger_bands(closes)
        atr_val = self.atr(highs, lows, closes)
        sma_cross = self.sma_crossover(closes)
        vol_sma = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(volumes.mean())
        vol_ratio = round(float(volumes[-1]) / vol_sma, 2) if vol_sma > 0 else 1.0

        return {
            "rsi": rsi_val,
            "macd": macd_data,
            "bollinger": bb_data,
            "atr": atr_val,
            "sma_crossover": sma_cross,
            "volume_ratio": vol_ratio,
            "price": round(float(closes[-1]), 4),
            "price_change_pct": round(
                (closes[-1] - closes[-2]) / closes[-2] * 100, 4
            ) if len(closes) >= 2 else 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL ENGINE — Combines Technical + Sentiment → Trade Decision
# ══════════════════════════════════════════════════════════════════════════════

class SignalEngine:
    """
    Composite scoring model that weighs:
      - RSI (momentum oscillator)
      - MACD trend
      - Bollinger Band position
      - SMA crossover
      - Volume anomaly
      - News sentiment
    Returns: BUY / SELL / HOLD with confidence score
    """

    def compute(self, ticker: str, tech: dict, sentiment_score: float) -> dict:
        signals = []
        reasons = []
        score = 0.0  # -2 to +2

        # ── RSI ──────────────────────────────────────────────────────────────
        rsi = tech["rsi"]
        if rsi < RSI_OVERSOLD:
            score += 0.6
            reasons.append(f"RSI oversold ({rsi})")
            signals.append("BUY")
        elif rsi > RSI_OVERBOUGHT:
            score -= 0.6
            reasons.append(f"RSI overbought ({rsi})")
            signals.append("SELL")
        else:
            reasons.append(f"RSI neutral ({rsi})")

        # ── MACD ─────────────────────────────────────────────────────────────
        if tech["macd"]["trend"] == "bullish":
            score += 0.5
            reasons.append("MACD bullish crossover")
            signals.append("BUY")
        elif tech["macd"]["trend"] == "bearish":
            score -= 0.5
            reasons.append("MACD bearish crossover")
            signals.append("SELL")

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb_pos = tech["bollinger"]["position"]
        if bb_pos < 0.2:
            score += 0.4
            reasons.append(f"Price near lower Bollinger Band ({bb_pos:.2f})")
            signals.append("BUY")
        elif bb_pos > 0.8:
            score -= 0.4
            reasons.append(f"Price near upper Bollinger Band ({bb_pos:.2f})")
            signals.append("SELL")

        # ── SMA Crossover ─────────────────────────────────────────────────────
        sma = tech["sma_crossover"]["signal"]
        if sma == "bullish":
            score += 0.3
            reasons.append("SMA 9/21 bullish crossover")
            signals.append("BUY")
        elif sma == "bearish":
            score -= 0.3
            reasons.append("SMA 9/21 bearish crossover")
            signals.append("SELL")

        # ── Volume Anomaly ────────────────────────────────────────────────────
        vol_ratio = tech["volume_ratio"]
        if vol_ratio > 1.8:
            direction_boost = 0.2 if score > 0 else -0.2
            score += direction_boost
            reasons.append(f"High volume spike ({vol_ratio:.1f}x avg)")

        # ── News Sentiment ────────────────────────────────────────────────────
        news_contribution = sentiment_score * NEWS_SENTIMENT_WEIGHT
        score += news_contribution
        if abs(sentiment_score) > 0.3:
            direction = "positive" if sentiment_score > 0 else "negative"
            reasons.append(f"News sentiment {direction} ({sentiment_score:.2f})")

        # ── Final Decision ────────────────────────────────────────────────────
        confidence = min(abs(score) / 2.0, 1.0)

        if score >= 0.7:
            action = "BUY"
        elif score <= -0.7:
            action = "SELL"
        else:
            action = "HOLD"

        return {
            "ticker": ticker,
            "action": action,
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "reasons": reasons,
            "rsi": rsi,
            "macd_trend": tech["macd"]["trend"],
            "bb_position": bb_pos,
            "sentiment": sentiment_score,
            "volume_ratio": vol_ratio,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# RISK MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Enforces:
      - Max portfolio risk (total open positions as % of equity)
      - Per-trade risk (stop-loss sizing via ATR)
      - Max concurrent positions
      - Daily loss limit circuit breaker
    """

    def __init__(self):
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.circuit_breaker: bool = False
        self.last_reset: datetime = datetime.now(timezone.utc).date()

    def reset_daily_if_needed(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset:
            log.info("[RiskManager] Daily reset triggered.")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.circuit_breaker = False
            self.last_reset = today

    def check_circuit_breaker(self, equity: float) -> bool:
        self.reset_daily_if_needed()
        max_daily_loss = equity * 0.03  # 3% max daily loss
        if self.daily_pnl < -max_daily_loss:
            if not self.circuit_breaker:
                log.warning(
                    f"[RiskManager] CIRCUIT BREAKER TRIGGERED. "
                    f"Daily P&L: ${self.daily_pnl:.2f}"
                )
            self.circuit_breaker = True
        return self.circuit_breaker

    def calculate_position_size(
        self, equity: float, price: float, atr: float,
        confidence: float = 0.5
    ) -> int:
        """
        Kelly-inspired ATR position sizing.
        Risk = equity * RISK_PER_TRADE
        Shares = Risk / (ATR * multiplier)
        Scaled by signal confidence.
        """
        if price <= 0 or atr <= 0:
            return 0
        risk_amount = equity * RISK_PER_TRADE * (0.5 + 0.5 * confidence)
        stop_distance = atr * 1.5  # 1.5× ATR stop loss
        raw_shares = risk_amount / stop_distance
        # Max 5% of equity in single position
        max_shares = (equity * 0.05) / price
        shares = int(min(raw_shares, max_shares))
        return max(1, shares)

    def can_open_position(
        self, ticker: str, open_positions: dict, equity: float
    ) -> tuple[bool, str]:
        if self.circuit_breaker:
            return False, "Circuit breaker active"
        if len(open_positions) >= MAX_POSITIONS:
            return False, f"Max positions ({MAX_POSITIONS}) reached"
        if ticker in open_positions:
            return False, f"Already holding {ticker}"

        # Portfolio heat check
        total_market_value = sum(
            abs(float(p.market_value)) for p in open_positions.values()
        )
        heat = total_market_value / equity if equity > 0 else 1.0
        if heat > MAX_PORTFOLIO_RISK:
            return False, f"Portfolio heat {heat:.1%} exceeds limit"

        return True, "OK"


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET BROADCAST SERVER
# ══════════════════════════════════════════════════════════════════════════════

class BroadcastServer:
    """Pushes live JSON updates to all connected dashboard clients."""

    def __init__(self):
        self.clients: set = set()
        self._lock = asyncio.Lock()
        self.latest_payload: dict = {}

    async def register(self, ws):
        async with self._lock:
            self.clients.add(ws)
        log.info(f"[WS] Client connected. Total: {len(self.clients)}")
        try:
            if self.latest_payload:
                await ws.send(json.dumps(self.latest_payload))
            await ws.wait_closed()
        finally:
            async with self._lock:
                self.clients.discard(ws)
            log.info(f"[WS] Client disconnected. Total: {len(self.clients)}")

    async def broadcast(self, data: dict):
        self.latest_payload = data
        msg = json.dumps(data)
        dead = set()
        async with self._lock:
            clients = set(self.clients)
        for ws in clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self.clients -= dead

    async def start(self):
        log.info(f"[WS] Dashboard server starting on ws://0.0.0.0:{WS_PORT}")
        async def health_check(connection, request):
            if request.method in ("HEAD", "GET") and request.headers.get("upgrade", "").lower() != "websocket":
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            return None

        async with websockets.serve(self.register, "0.0.0.0", WS_PORT, process_request=health_check):
            await asyncio.Future()


class AlphaSentinel:
    def __init__(self):
        log.info("═" * 60)
        log.info("  ALPHA SENTINEL v3.1 — Initializing...")
        log.info("═" * 60)

        # Alpaca clients
        self.trade_client = TradingClient(
            ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING
        )
        self.data_client = StockHistoricalDataClient(
            ALPACA_API_KEY, ALPACA_SECRET_KEY
        )

        # Sub-systems
        self.news_engine = NewsEngine()
        self.ta = TechnicalAnalysis()
        self.signal_engine = SignalEngine()
        self.risk_manager = RiskManager()
        self.broadcast = BroadcastServer()

        # State
        self.positions: dict = {}
        self.account = None
        self.trade_log: deque = deque(maxlen=200)
        self.signals_cache: dict[str, dict] = {}
        self.price_cache: dict[str, float] = {}
        self.bars_cache: dict[str, pd.DataFrame] = {}
        self.running = True

        log.info(f"  Watchlist: {', '.join(WATCHLIST)}")
        log.info(f"  Paper Trading: {PAPER_TRADING}")
        log.info(f"  Trade Interval: {TRADE_INTERVAL}s")

    # ── Account & Position Helpers ────────────────────────────────────────────

    def sync_account(self):
        try:
            self.account = self.trade_client.get_account()
            return self.account
        except Exception as e:
            log.error(f"[Account] Sync failed: {e}")
            return None

    def sync_positions(self):
        try:
            positions = self.trade_client.get_all_positions()
            self.positions = {p.symbol: p for p in positions}
            return self.positions
        except Exception as e:
            log.error(f"[Positions] Sync failed: {e}")
            return {}

    # ── Market Data Fetching ──────────────────────────────────────────────────

    def fetch_bars(self, ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
        """Fetch OHLCV bars for technical analysis."""
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start, end=end,
                feed="iex"
            )
            bars = self.data_client.get_stock_bars(request)
            df = bars.df
            if hasattr(df.index, "levels"):
                df = df.xs(ticker, level="symbol") if ticker in df.index.get_level_values("symbol") else df
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            if len(df) < 10:
                return None
            self.bars_cache[ticker] = df
            return df
        except Exception as e:
            log.warning(f"[Bars] {ticker} fetch error: {e}")
            return None

    def fetch_latest_price(self, ticker: str) -> Optional[float]:
        """Fetch real-time quote for a ticker."""
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=ticker, feed="iex")
            quote = self.data_client.get_stock_latest_quote(req)
            price = None
            if ticker in quote:
                q = quote[ticker]
                price = float((q.ask_price + q.bid_price) / 2)
                if price <= 0:
                    price = float(q.ask_price or q.bid_price or 0)
            if price and price > 0:
                self.price_cache[ticker] = price
                return price
            # Fallback to cached bars close
            if ticker in self.bars_cache and len(self.bars_cache[ticker]) > 0:
                return float(self.bars_cache[ticker]["close"].iloc[-1])
            return None
        except Exception as e:
            log.warning(f"[Quote] {ticker} error: {e}")
            if ticker in self.bars_cache and len(self.bars_cache[ticker]) > 0:
                return float(self.bars_cache[ticker]["close"].iloc[-1])
            return None

    # ── Trading Execution ─────────────────────────────────────────────────────

    def place_buy(self, ticker: str, qty: int, price: float,
                  confidence: float) -> Optional[dict]:
        try:
            order_req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            order = self.trade_client.submit_order(order_req)
            entry = {
                "id": str(order.id),
                "action": "BUY",
                "ticker": ticker,
                "qty": qty,
                "price": round(price, 2),
                "total": round(price * qty, 2),
                "confidence": round(confidence, 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": str(order.status),
            }
            self.trade_log.appendleft(entry)
            log.info(
                f"[TRADE] BUY {qty}× {ticker} @ ~${price:.2f} "
                f"(confidence: {confidence:.1%})"
            )
            return entry
        except Exception as e:
            log.error(f"[Trade] BUY {ticker} failed: {e}")
            return None

    def place_sell(self, ticker: str, qty: int, price: float) -> Optional[dict]:
        try:
            order_req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            order = self.trade_client.submit_order(order_req)
            entry = {
                "id": str(order.id),
                "action": "SELL",
                "ticker": ticker,
                "qty": qty,
                "price": round(price, 2),
                "total": round(price * qty, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": str(order.status),
            }
            self.trade_log.appendleft(entry)
            log.info(f"[TRADE] SELL {qty}× {ticker} @ ~${price:.2f}")
            return entry
        except Exception as e:
            log.error(f"[Trade] SELL {ticker} failed: {e}")
            return None

    def check_stop_loss_take_profit(self):
        """
        Check open positions and exit if:
          - Loss > 2× ATR (stop loss)
          - Gain > 3× ATR (take profit)
        """
        for ticker, pos in list(self.positions.items()):
            try:
                current_price = self.fetch_latest_price(ticker)
                if not current_price:
                    continue
                avg_entry = float(pos.avg_entry_price)
                qty = int(float(pos.qty))
                pnl_pct = (current_price - avg_entry) / avg_entry

                # Get ATR for dynamic SL/TP
                atr = 0.02 * avg_entry  # fallback 2%
                if ticker in self.bars_cache:
                    df = self.bars_cache[ticker]
                    atr = self.ta.atr(
                        df["high"].values, df["low"].values,
                        df["close"].values
                    )
                atr_pct = atr / avg_entry

                # Stop loss: -2× ATR
                if pnl_pct < -(2 * atr_pct):
                    log.warning(f"[SL] {ticker} stop-loss hit: {pnl_pct:.2%}")
                    self.place_sell(ticker, qty, current_price)
                    self.risk_manager.daily_pnl += (current_price - avg_entry) * qty

                # Take profit: +3× ATR
                elif pnl_pct > (3 * atr_pct):
                    log.info(f"[TP] {ticker} take-profit hit: {pnl_pct:.2%}")
                    self.place_sell(ticker, qty, current_price)
                    self.risk_manager.daily_pnl += (current_price - avg_entry) * qty

            except Exception as e:
                log.error(f"[SL/TP] {ticker}: {e}")

    # ── Main Cycle ────────────────────────────────────────────────────────────

    def run_analysis_cycle(self):
        """Run one full analysis + trade decision cycle."""
        log.info("[Cycle] ── Starting analysis cycle ──")

        self.sync_account()
        self.sync_positions()
        equity = float(self.account.equity) if self.account else 100_000.0

        self.risk_manager.reset_daily_if_needed()

        if self.risk_manager.check_circuit_breaker(equity):
            log.warning("[Cycle] Circuit breaker active — skipping trades.")
            return

        # SL/TP check
        self.check_stop_loss_take_profit()

        # Refresh news
        self.news_engine.refresh_all_feeds()

        for ticker in WATCHLIST:
            try:
                df = self.fetch_bars(ticker)
                if df is None or len(df) < 30:
                    log.warning(f"[{ticker}] Insufficient data, skipping.")
                    continue

                price = self.fetch_latest_price(ticker)
                if not price:
                    continue

                tech = self.ta.full_analysis(df)
                sentiment = self.news_engine.get_ticker_score(ticker)
                signal = self.signal_engine.compute(ticker, tech, sentiment)
                signal["price"] = price
                self.signals_cache[ticker] = signal

                log.info(
                    f"[{ticker}] ${price:.2f} | RSI:{tech['rsi']:.0f} | "
                    f"MACD:{tech['macd']['trend']} | Sentiment:{sentiment:+.2f} | "
                    f"→ {signal['action']} (conf:{signal['confidence']:.0%})"
                )

                action = signal["action"]

                # ── BUY Logic ─────────────────────────────────────────────────
                if action == "BUY" and signal["confidence"] >= 0.4:
                    can_open, reason = self.risk_manager.can_open_position(
                        ticker, self.positions, equity
                    )
                    if can_open:
                        qty = self.risk_manager.calculate_position_size(
                            equity, price, tech["atr"], signal["confidence"]
                        )
                        if qty > 0:
                            self.place_buy(ticker, qty, price, signal["confidence"])
                            self.sync_positions()
                    else:
                        log.info(f"[{ticker}] BUY blocked: {reason}")

                # ── SELL Logic ────────────────────────────────────────────────
                elif action == "SELL" and ticker in self.positions:
                    pos = self.positions[ticker]
                    qty = int(float(pos.qty))
                    if qty > 0:
                        self.place_sell(ticker, qty, price)
                        self.sync_positions()

            except Exception as e:
                log.error(f"[{ticker}] Cycle error: {e}", exc_info=True)

        log.info(f"[Cycle] ── Cycle complete. Positions: {len(self.positions)} ──")

    # ── Dashboard Data Builder ────────────────────────────────────────────────

    def build_dashboard_payload(self) -> dict:
        """Build a rich JSON payload for the dashboard WebSocket."""
        account = self.account
        positions_data = []
        for ticker, pos in self.positions.items():
            price = self.price_cache.get(ticker, float(pos.avg_entry_price))
            positions_data.append({
                "ticker": ticker,
                "qty": int(float(pos.qty)),
                "avg_entry": round(float(pos.avg_entry_price), 2),
                "current_price": round(price, 2),
                "market_value": round(float(pos.market_value), 2),
                "unrealized_pl": round(float(pos.unrealized_pl), 2),
                "unrealized_plpc": round(float(pos.unrealized_plpc) * 100, 2),
            })

        # Build watchlist prices + signals
        watchlist_data = []
        for ticker in WATCHLIST:
            price = self.price_cache.get(ticker, 0)
            signal = self.signals_cache.get(ticker, {})
            bars_df = self.bars_cache.get(ticker)
            sparkline = []
            if bars_df is not None and len(bars_df) >= 10:
                sparkline = [
                    round(float(v), 2)
                    for v in bars_df["close"].values[-15:]
                ]
            watchlist_data.append({
                "ticker": ticker,
                "price": round(price, 2),
                "signal": signal.get("action", "—"),
                "confidence": signal.get("confidence", 0),
                "rsi": signal.get("rsi", 0),
                "sentiment": signal.get("sentiment", 0),
                "macd_trend": signal.get("macd_trend", "—"),
                "sparkline": sparkline,
                "score": signal.get("score", 0),
            })

        return {
            "type": "full_update",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "account": {
                "equity": round(float(account.equity), 2) if account else 0,
                "buying_power": round(float(account.buying_power), 2) if account else 0,
                "cash": round(float(account.cash), 2) if account else 0,
                "portfolio_value": round(float(account.portfolio_value), 2) if account else 0,
                "daily_pl": round(self.risk_manager.daily_pnl, 2),
                "circuit_breaker": self.risk_manager.circuit_breaker,
                "paper_trading": PAPER_TRADING,
            },
            "positions": positions_data,
            "watchlist": watchlist_data,
            "trades": list(self.trade_log)[:30],
            "news": self.news_engine.get_recent_news(20),
            "risk": {
                "daily_trades": self.risk_manager.daily_trades,
                "open_positions": len(self.positions),
                "max_positions": MAX_POSITIONS,
                "circuit_breaker": self.risk_manager.circuit_breaker,
            }
        }

    # ── Async Run Loop ────────────────────────────────────────────────────────

    async def trading_loop(self):
        """Main async trading loop — runs every TRADE_INTERVAL seconds."""
        while self.running:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.run_analysis_cycle)

                payload = self.build_dashboard_payload()
                await self.broadcast.broadcast(payload)

                log.info(f"[Loop] Sleeping {TRADE_INTERVAL}s until next cycle...")
                await asyncio.sleep(TRADE_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"[Loop] Unhandled error: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def price_update_loop(self):
        """Lightweight loop — updates prices every 15s for live dashboard."""
        while self.running:
            try:
                for ticker in WATCHLIST:
                    price = self.fetch_latest_price(ticker)
                    if price:
                        self.price_cache[ticker] = price

                payload = self.build_dashboard_payload()
                payload["type"] = "price_update"
                await self.broadcast.broadcast(payload)

                await asyncio.sleep(15)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"[PriceLoop] {e}")
                await asyncio.sleep(15)

    async def run(self):
        """Bootstrap and run all async tasks."""
        log.info("[Boot] Starting Alpha Sentinel systems...")
        await asyncio.gather(
            self.broadcast.start(),
            self.trading_loop(),
            self.price_update_loop(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bot = AlphaSentinel()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log.info("Alpha Sentinel stopped by user.")