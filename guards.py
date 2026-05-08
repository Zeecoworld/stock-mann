"""
guards.py  — v2
─────────────────────────────────────────────────────────────────────────────
CHANGES vs v1:
  1. Open/close 30-min window blocked by default (widest spreads, most noise).
     9:30-10:00 ET and 3:30-4:00 ET skipped unless opted in via env var.
  2. next_open() accounts for effective 10:00 open when window is blocked.
  3. NYSE calendar extended through 2027.

ENV:
  TRADE_OPEN_WINDOW=true   -- allow 9:30-10:00 ET  (default: blocked)
  TRADE_CLOSE_WINDOW=true  -- allow 3:30-4:00 ET   (default: blocked)
"""
from __future__ import annotations

import logging, os, re, time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:
    import pytz
    ET = pytz.timezone("America/New_York")

_TRADE_OPEN_WINDOW  = os.getenv("TRADE_OPEN_WINDOW",  "false").lower() == "true"
_TRADE_CLOSE_WINDOW = os.getenv("TRADE_CLOSE_WINDOW", "false").lower() == "true"


_NYSE_HOLIDAYS = {
    "2025-01-01","2025-01-20","2025-02-17","2025-04-18","2025-05-26",
    "2025-06-19","2025-07-04","2025-09-01","2025-11-27","2025-12-25",
    "2026-01-01","2026-01-19","2026-02-16","2026-04-03","2026-05-25",
    "2026-06-19","2026-07-03","2026-09-07","2026-11-26","2026-12-25",
    "2027-01-01","2027-01-18","2027-02-15","2027-04-26","2027-05-31",
    "2027-06-18","2027-07-05","2027-09-06","2027-11-25","2027-12-24",
}
_EARLY_CLOSE = {
    "2025-07-03","2025-11-28","2025-12-24",
    "2026-07-02","2026-11-27","2026-12-24",
    "2027-07-02","2027-11-26","2027-12-23",
}


class MarketHoursGuard:
    def __init__(
        self,
        allow_premarket:    bool = False,
        block_open_window:  bool = not _TRADE_OPEN_WINDOW,
        block_close_window: bool = not _TRADE_CLOSE_WINDOW,
    ):
        self.allow_premarket    = allow_premarket
        self.block_open_window  = block_open_window
        self.block_close_window = block_close_window
        self.reason: str        = ""

    def is_open(self, now_utc=None) -> bool:
        now_et   = (now_utc or datetime.now(timezone.utc)).astimezone(ET)
        date_str = now_et.strftime("%Y-%m-%d")

        if now_et.weekday() >= 5:
            self.reason = f"Market closed -- weekend ({now_et.strftime('%A')})"; return False
        if date_str in _NYSE_HOLIDAYS:
            self.reason = f"Market closed -- NYSE holiday ({date_str})"; return False

        open_t   = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
        eff_open = now_et.replace(hour=10, minute=0,  second=0, microsecond=0)
        close_h  = 13 if date_str in _EARLY_CLOSE else 16
        close_t  = now_et.replace(hour=close_h, minute=0,  second=0, microsecond=0)
        pre_cls  = now_et.replace(hour=15, minute=30, second=0, microsecond=0)

        if self.allow_premarket:
            pre = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
            if now_et < pre:
                self.reason = "Pre-market not open yet (4:00 AM ET)"; return False
        else:
            if now_et < open_t:
                mins = int((open_t - now_et).total_seconds() / 60)
                self.reason = f"Market opens in {mins}m (9:30 AM ET)"; return False

        if self.block_open_window and open_t <= now_et < eff_open:
            mins = int((eff_open - now_et).total_seconds() / 60)
            self.reason = f"Opening 30-min blocked -- wide spreads (resumes 10:00 ET, {mins}m)"; return False

        if now_et >= close_t:
            self.reason = f"Market closed ({close_h}:00 ET)"; return False

        if self.block_close_window and pre_cls <= now_et < close_t:
            mins = int((close_t - now_et).total_seconds() / 60)
            self.reason = f"Closing 30-min blocked -- end-of-day noise (closes in {mins}m)"; return False

        self.reason = f"Market open  {now_et.strftime('%H:%M ET')}"
        return True

    def next_open(self) -> datetime:
        now_et    = datetime.now(ET)
        oh, om    = (10, 0) if self.block_open_window else (9, 30)
        candidate = now_et.replace(hour=oh, minute=om, second=0, microsecond=0)
        for _ in range(14):
            if candidate <= now_et:
                candidate += timedelta(days=1)
                candidate  = candidate.replace(hour=oh, minute=om, second=0)
            if candidate.weekday() < 5 and candidate.strftime("%Y-%m-%d") not in _NYSE_HOLIDAYS:
                return candidate
            candidate += timedelta(days=1)
        return candidate

    def seconds_until_open(self) -> int:
        return max(0, int((self.next_open() - datetime.now(ET)).total_seconds()))


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
            logger.info("[Breaker] New day -- baseline $%.2f", self._day_start)

    def is_tripped(self, prices: Dict[str, float]) -> bool:
        self._reset_day(prices)
        if self._tripped_at:
            elapsed = time.time() - self._tripped_at
            if elapsed < self.cooldown_sec:
                mins = int((self.cooldown_sec - elapsed) / 60)
                self.reason = f"Circuit breaker cooling -- {mins}m remaining"
                return True
            self._tripped_at = None

        current  = self.portfolio.total_value(prices)
        total_dd = (self.portfolio.starting_cash - current) / self.portfolio.starting_cash
        if total_dd >= self.total_limit:
            self._trip(f"Total drawdown {total_dd:.1%} >= {self.total_limit:.0%}"); return True

        daily_dd = (self._day_start - current) / self._day_start if self._day_start else 0
        if daily_dd >= self.daily_limit:
            self._trip(f"Daily drawdown {daily_dd:.1%} >= {self.daily_limit:.0%}"); return True

        pnl = (current - self.portfolio.starting_cash) / self.portfolio.starting_cash
        self.reason = f"OK  daily={-daily_dd:+.1%}  total={pnl:+.1%}"
        return False

    def _trip(self, reason):
        self._tripped_at = time.time()
        self.reason = f"CIRCUIT BREAKER: {reason}"
        logger.warning("[Breaker] %s", self.reason)


class DuplicateSignalThrottle:
    def __init__(self, cooldown_minutes: int = 60):
        self.cooldown_sec = cooldown_minutes * 60
        self._last: Dict[str, float] = {}

    def is_blocked(self, ticker: str, signal: str) -> bool:
        key  = f"{ticker}:{signal}"
        last = self._last.get(key)
        if last and (time.time() - last) < self.cooldown_sec:
            mins = int((self.cooldown_sec - (time.time() - last)) / 60)
            logger.info("[Throttle] %s %s -- %dm cooldown", signal, ticker, mins)
            return True
        return False

    def record(self, ticker: str, signal: str):
        self._last[f"{ticker}:{signal}"] = time.time()


_TICKER_RE = re.compile(
    r'\b(NVDA|TSLA|AAPL|MSFT|AMZN|META|GOOGL|GOOG|AMD|PLTR|COIN|'
    r'NFLX|DIS|BABA|UBER|LYFT|SNAP|GME|AMC|SPY|QQQ|'
    r'INTC|QCOM|ARM|SMCI|MSTR|HOOD|SOFI|RBLX|SQ|SHOP|'
    r'JPM|BAC|GS|WFC|C|V|MA|PYPL|'
    r'XOM|CVX|OXY|WMT|COST|TGT|HD|LOW|'
    r'BTC|ETH|SOL|DOGE|BNB)\b',
    re.IGNORECASE,
)
_TREND_FEEDS = [
    "https://news.google.com/rss/search?q=stock+market+today&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=NYSE+NASDAQ+earnings&hl=en-US&gl=US&ceid=US:en",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
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
        self._counts:       Dict[str, int] = {}
        self._trending:     Set[str]       = set()
        self._last_refresh: float          = 0
        self._ttl:          int            = 900

    async def refresh(self, force: bool = False):
        if not force and (time.time() - self._last_refresh) < self._ttl:
            return
        import httpx, asyncio

        async def _fetch(url):
            try:
                async with httpx.AsyncClient(timeout=10, headers=_HEADERS) as c:
                    r = await c.get(url, follow_redirects=True)
                    return r.text if r.status_code == 200 else ""
            except Exception:
                return ""

        texts  = await asyncio.gather(*[_fetch(u) for u in _TREND_FEEDS])
        counts: Dict[str, int] = {}
        for text in texts:
            if not text: continue
            for title in re.findall(r"<title[^>]*>(.*?)</title>", text, re.DOTALL):
                clean = re.sub(r"<[^>]+>|<!\[CDATA\[|\]\]>", "", title).strip()
                for m in _TICKER_RE.finditer(clean):
                    sym = m.group(0).upper()
                    counts[sym] = counts.get(sym, 0) + 1

        self._counts       = counts
        self._trending     = {s for s, c in counts.items() if c >= self.MENTION_THRESHOLD}
        self._last_refresh = time.time()
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("[TrendFilter] %s", "  ".join(f"{s}({c})" for s, c in top))

    def is_trending(self, ticker: str, min_mentions: int = None) -> bool:
        if not self._trending: return True
        return self._counts.get(ticker.lstrip("$").upper(), 0) >= (
            min_mentions or self.MENTION_THRESHOLD)

    def mention_count(self, ticker: str) -> int:
        return self._counts.get(ticker.lstrip("$").upper(), 0)

    def top_trending(self, n: int = 10):
        return sorted(self._counts.items(), key=lambda x: x[1], reverse=True)[:n]