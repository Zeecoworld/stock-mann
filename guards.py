"""
paper_trader/guards.py
Production safety guards — market hours, drawdown breaker, signal throttle, trending filter.
"""
from __future__ import annotations

import logging, time, re
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:
    import pytz
    ET = pytz.timezone("America/New_York")


# ── NYSE calendar ─────────────────────────────────────────────────────────────

_NYSE_HOLIDAYS = {
    "2025-01-01","2025-01-20","2025-02-17","2025-04-18","2025-05-26",
    "2025-06-19","2025-07-04","2025-09-01","2025-11-27","2025-12-25",
    "2026-01-01","2026-01-19","2026-02-16","2026-04-03","2026-05-25",
    "2026-06-19","2026-07-03","2026-09-07","2026-11-26","2026-12-25",
}
_EARLY_CLOSE = {
    "2025-07-03","2025-11-28","2025-12-24",
    "2026-07-02","2026-11-27","2026-12-24",
}


class MarketHoursGuard:
    def __init__(self, allow_premarket: bool = False):
        self.allow_premarket = allow_premarket
        self.reason: str = ""

    def is_open(self, now_utc=None) -> bool:
        now_et   = (now_utc or datetime.now(timezone.utc)).astimezone(ET)
        date_str = now_et.strftime("%Y-%m-%d")
        weekday  = now_et.weekday()

        if weekday >= 5:
            self.reason = f"Market closed — weekend ({now_et.strftime('%A')})"; return False
        if date_str in _NYSE_HOLIDAYS:
            self.reason = f"Market closed — NYSE holiday ({date_str})"; return False

        open_t   = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
        close_h  = 13 if date_str in _EARLY_CLOSE else 16
        close_t  = now_et.replace(hour=close_h, minute=0, second=0, microsecond=0)

        if self.allow_premarket:
            pre = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
            if now_et < pre:
                self.reason = "Pre-market not open yet (4:00 AM ET)"; return False
        else:
            if now_et < open_t:
                mins = int((open_t - now_et).total_seconds() / 60)
                self.reason = f"Market opens in {mins} min (9:30 AM ET)"; return False

        if now_et >= close_t:
            self.reason = f"Market closed for the day ({close_h}:00 ET)"; return False

        self.reason = f"Market open {now_et.strftime('%H:%M ET')}"; return True

    def next_open(self) -> datetime:
        now_et    = datetime.now(ET)
        candidate = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        for _ in range(10):
            if candidate <= now_et:
                candidate += timedelta(days=1)
                candidate  = candidate.replace(hour=9, minute=30, second=0)
            date_str = candidate.strftime("%Y-%m-%d")
            if candidate.weekday() < 5 and date_str not in _NYSE_HOLIDAYS:
                return candidate
            candidate += timedelta(days=1)
        return candidate

    def seconds_until_open(self) -> int:
        return max(0, int((self.next_open() - datetime.now(ET)).total_seconds()))


# ── Drawdown Circuit Breaker ──────────────────────────────────────────────────

class DrawdownCircuitBreaker:
    def __init__(self, portfolio, daily_limit=0.03, total_limit=0.10, cooldown_min=60):
        self.portfolio    = portfolio
        self.daily_limit  = daily_limit
        self.total_limit  = total_limit
        self.cooldown_sec = cooldown_min * 60
        self._tripped_at: Optional[float] = None
        self._day_start:  float           = portfolio.starting_cash
        self._day_date:   str             = datetime.now(ET).strftime("%Y-%m-%d")
        self.reason: str                  = ""

    def _reset_day(self, prices):
        today = datetime.now(ET).strftime("%Y-%m-%d")
        if today != self._day_date:
            self._day_date  = today
            self._day_start = self.portfolio.total_value(prices)
            logger.info("[Breaker] New trading day — baseline $%.2f", self._day_start)

    def is_tripped(self, prices: Dict[str,float]) -> bool:
        self._reset_day(prices)
        if self._tripped_at:
            elapsed = time.time() - self._tripped_at
            if elapsed < self.cooldown_sec:
                mins = int((self.cooldown_sec - elapsed) / 60)
                self.reason = f"Circuit breaker cooling down — {mins} min remaining"
                return True
            self._tripped_at = None
            logger.info("[Breaker] Cooldown expired, trading resumed")

        current  = self.portfolio.total_value(prices)
        total_dd = (self.portfolio.starting_cash - current) / self.portfolio.starting_cash
        if total_dd >= self.total_limit:
            self._trip(f"Total drawdown {total_dd:.1%} >= {self.total_limit:.0%} limit")
            return True

        daily_dd = (self._day_start - current) / self._day_start if self._day_start else 0
        if daily_dd >= self.daily_limit:
            self._trip(f"Daily drawdown {daily_dd:.1%} >= {self.daily_limit:.0%} limit")
            return True

        current_pnl = ((current - self.portfolio.starting_cash) / self.portfolio.starting_cash)
        self.reason = f"OK  daily={-daily_dd:+.1%}  total={current_pnl:+.1%}"
        return False

    def _trip(self, reason):
        self._tripped_at = time.time()
        self.reason = f"CIRCUIT BREAKER: {reason}"
        logger.warning("[Breaker] %s", self.reason)


# ── Duplicate Signal Throttle ─────────────────────────────────────────────────

class DuplicateSignalThrottle:
    def __init__(self, cooldown_minutes: int = 60):
        self.cooldown_sec = cooldown_minutes * 60
        self._last: Dict[str,float] = {}

    def is_blocked(self, ticker: str, signal: str) -> bool:
        key  = f"{ticker}:{signal}"
        last = self._last.get(key)
        if last and (time.time() - last) < self.cooldown_sec:
            mins = int((self.cooldown_sec - (time.time() - last)) / 60)
            logger.info("[Throttle] %s %s — %d min cooldown", signal, ticker, mins)
            return True
        return False

    def record(self, ticker: str, signal: str):
        self._last[f"{ticker}:{signal}"] = time.time()


# ── Trending Ticker Filter ────────────────────────────────────────────────────

_TICKER_RE = re.compile(
    r'\b(NVDA|TSLA|AAPL|MSFT|AMZN|META|GOOGL|GOOG|AMD|PLTR|COIN|'
    r'NFLX|DIS|BABA|UBER|LYFT|SNAP|GME|AMC|SPY|QQQ|'
    r'INTC|QCOM|ARM|SMCI|MSTR|HOOD|SOFI|RBLX|SQ|SHOP|'
    r'JPM|BAC|GS|WFC|C|V|MA|PYPL|'
    r'XOM|CVX|OXY|WMT|COST|TGT|HD|LOW|'
    r'BTC|ETH|SOL|DOGE|BNB)\b',
    re.IGNORECASE
)

# 13 free RSS feeds — no key needed
_TREND_FEEDS = [
    "https://news.google.com/rss/search?q=stock+market+today&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=NYSE+NASDAQ+earnings&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=stocks+trading+today&hl=en-US&gl=US&ceid=US:en",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/news/wealth",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://seekingalpha.com/market_currents.xml",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "https://finance.yahoo.com/news/rssindex",
    "https://www.reddit.com/r/investing/hot.rss?limit=25",
    "https://www.reddit.com/r/stocks/hot.rss?limit=25",
]

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept": "application/rss+xml,application/xml,text/xml,*/*",
}


class TrendingTickerFilter:
    MENTION_THRESHOLD = 2

    def __init__(self):
        self._counts:      Dict[str,int] = {}
        self._trending:    Set[str]      = set()
        self._last_refresh: float        = 0
        self._ttl:          int          = 900   # 15 min

    async def refresh(self, force: bool = False):
        if not force and (time.time() - self._last_refresh) < self._ttl:
            return

        import httpx, asyncio

        async def _fetch(url):
            try:
                async with httpx.AsyncClient(timeout=10, headers=_HEADERS) as c:
                    r = await c.get(url, follow_redirects=True)
                    return r.text if r.status_code == 200 else ""
            except Exception: return ""

        texts  = await asyncio.gather(*[_fetch(u) for u in _TREND_FEEDS])
        counts: Dict[str,int] = {}

        for text in texts:
            if not text: continue
            titles = re.findall(r"<title[^>]*>(.*?)</title>", text, re.DOTALL)
            for title in titles:
                clean = re.sub(r"<[^>]+>|<!\[CDATA\[|\]\]>", "", title).strip()
                for m in _TICKER_RE.finditer(clean):
                    sym = m.group(0).upper()
                    counts[sym] = counts.get(sym, 0) + 1

        self._counts       = counts
        self._trending     = {s for s,c in counts.items() if c >= self.MENTION_THRESHOLD}
        self._last_refresh = time.time()

        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("[TrendFilter] Top: %s", "  ".join(f"{s}({c})" for s,c in top))

    def is_trending(self, ticker: str, min_mentions: int = None) -> bool:
        if not self._trending: return True   # fail-open
        sym  = ticker.lstrip("$").upper()
        return self._counts.get(sym, 0) >= (min_mentions or self.MENTION_THRESHOLD)

    def mention_count(self, ticker: str) -> int:
        return self._counts.get(ticker.lstrip("$").upper(), 0)

    def top_trending(self, n: int = 10):
        return sorted(self._counts.items(), key=lambda x: x[1], reverse=True)[:n]
