"""
strategy_engine/news_fetcher.py
Free, key-optional news fetchers covering multiple niches.

FREE / NO-KEY sources:
  GoogleNewsRSSFetcher  — any topic
  RedditRSSFetcher      — subreddit feeds
  HackerNewsFetcher     — tech/startup (Algolia API)
  BBCRSSFetcher         — world/politics/sports/tech
  ReutersRSSFetcher     — finance, politics, world
  CNBCRSSFetcher        — markets, business
  MarketWatchFetcher    — top stories
  SeekingAlphaFetcher   — market currents (public RSS)
  YahooFinanceRSSFetcher— finance news

FREE WITH KEY (silently skipped if no key):
  FinnhubNewsFetcher    — FINNHUB_API_KEY
  NewsAPIFetcher        — NEWSAPI_KEY
  CryptoPanicFetcher    — CRYPTOPANIC_KEY (optional)

ROUTING:
  NicheFetcher      — auto-selects by niche tag
  CompositeFetcher  — parallel fetch + deduplicate
"""
from __future__ import annotations

import asyncio, logging, os, re, urllib.parse
from typing import Dict, List, Literal, Optional

import httpx

logger  = logging.getLogger(__name__)
_TIMEOUT = httpx.Timeout(15.0)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

Niche = Literal["finance","crypto","sports","politics","tech","world","general"]


class BaseNewsFetcher:
    name: str = "base"

    async def fetch(self, query: str, max_results: int = 10) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def _parse_rss_titles(xml: str, max_results: int) -> List[str]:
        items = re.findall(r"<item[^>]*>.*?</item>", xml, re.DOTALL)
        titles: List[str] = []
        for item in items[:max_results * 2]:
            m = re.search(r"<title[^>]*>(.*?)</title>", item, re.DOTALL)
            if not m: continue
            t = m.group(1)
            t = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", t, flags=re.DOTALL)
            t = re.sub(r"<[^>]+>", "", t).strip()
            t = (t.replace("&amp;","&").replace("&lt;","<")
                  .replace("&gt;",">").replace("&quot;",'"').replace("&#39;","'"))
            if t and len(t) > 8:
                titles.append(t)
                if len(titles) >= max_results: break
        return titles


# ── Google News RSS ───────────────────────────────────────────────────────────

class GoogleNewsRSSFetcher(BaseNewsFetcher):
    name = "google_news"
    BASE = "https://news.google.com/rss/search"

    def __init__(self, language="en-US", country="US"):
        self.language = language; self.country = country

    async def fetch(self, query: str, max_results: int = 10) -> List[str]:
        clean  = query.lstrip("$")
        params = {"q": clean, "hl": self.language, "gl": self.country,
                  "ceid": f"{self.country}:{self.language[:2]}"}
        url = f"{self.BASE}?{urllib.parse.urlencode(params)}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as c:
                r = await c.get(url, follow_redirects=True)
                r.raise_for_status()
                titles = self._parse_rss_titles(r.text, max_results)
                logger.info("[GoogleNews] '%s' -> %d", clean[:40], len(titles))
                return titles
        except Exception as e:
            logger.error("[GoogleNews] %s: %s", clean[:40], e); return []


# ── Reddit RSS ────────────────────────────────────────────────────────────────

_REDDIT_NICHES: Dict[str, List[str]] = {
    "finance":  ["investing","stocks","finance"],
    "crypto":   ["cryptocurrency","bitcoin","ethfinance"],
    "sports":   ["sports","soccer","nba","nfl"],
    "politics": ["politics","worldnews","geopolitics"],
    "tech":     ["technology","programming","MachineLearning"],
    "world":    ["worldnews","news"],
    "general":  ["news","worldnews"],
}

class RedditRSSFetcher(BaseNewsFetcher):
    name = "reddit"

    def __init__(self, niche: Niche = "general", subreddit=None, sort="hot"):
        self.subreddits = [subreddit] if subreddit else _REDDIT_NICHES.get(niche, ["news"])
        self.sort = sort

    async def fetch(self, query: str, max_results: int = 10) -> List[str]:
        tasks   = [self._fetch_sub(s, max_results) for s in self.subreddits[:2]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        seen, titles = set(), []
        for batch in results:
            if isinstance(batch, Exception): continue
            for t in batch:
                if t not in seen: seen.add(t); titles.append(t)
        return titles[:max_results]

    async def _fetch_sub(self, sub: str, n: int) -> List[str]:
        url = f"https://www.reddit.com/r/{sub}/{self.sort}.rss?limit={n}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as c:
                r = await c.get(url, follow_redirects=True); r.raise_for_status()
                entries = re.findall(r"<entry>.*?</entry>", r.text, re.DOTALL)
                titles  = []
                for entry in entries[:n]:
                    m = re.search(r"<title[^>]*>(.*?)</title>", entry, re.DOTALL)
                    if m:
                        t = re.sub(r"<[^>]+>", "", m.group(1)).strip()
                        if t and t.lower() not in ("","reddit"): titles.append(t)
                return titles
        except Exception as e:
            logger.warning("[Reddit/r/%s] %s", sub, e); return []


# ── BBC RSS ───────────────────────────────────────────────────────────────────

_BBC_FEEDS: Dict[str,str] = {
    "finance":  "https://feeds.bbci.co.uk/news/business/rss.xml",
    "world":    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "politics": "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "sports":   "https://feeds.bbci.co.uk/sport/rss.xml",
    "tech":     "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "general":  "https://feeds.bbci.co.uk/news/rss.xml",
    "crypto":   "https://feeds.bbci.co.uk/news/business/rss.xml",
}

class BBCRSSFetcher(BaseNewsFetcher):
    name = "bbc"
    def __init__(self, niche: Niche = "general"):
        self.feed_url = _BBC_FEEDS.get(niche, _BBC_FEEDS["general"])

    async def fetch(self, query: str, max_results: int = 10) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as c:
                r = await c.get(self.feed_url, follow_redirects=True); r.raise_for_status()
                return self._parse_rss_titles(r.text, max_results)
        except Exception as e:
            logger.error("[BBC] %s", e); return []


# ── Reuters RSS ───────────────────────────────────────────────────────────────

_REUTERS_FEEDS: Dict[str,str] = {
    "world":    "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "politics": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "sports":   "https://feeds.bbci.co.uk/sport/rss.xml",
    "general":  "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "crypto":   "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
}

class ReutersRSSFetcher(BaseNewsFetcher):
    name = "reuters"
    def __init__(self, niche: Niche = "general"):
        self.feed_url = _REUTERS_FEEDS.get(niche, _REUTERS_FEEDS["general"])

    async def fetch(self, query: str, max_results: int = 10) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as c:
                r = await c.get(self.feed_url, follow_redirects=True); r.raise_for_status()
                return self._parse_rss_titles(r.text, max_results)
        except Exception as e:
            logger.error("[Reuters] %s", e); return []


# ── CNBC RSS ──────────────────────────────────────────────────────────────────

class CNBCRSSFetcher(BaseNewsFetcher):
    name = "cnbc"
    FEEDS = [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",   # markets
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",    # money
    ]

    async def fetch(self, query: str, max_results: int = 10) -> List[str]:
        titles = []
        for url in self.FEEDS:
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as c:
                    r = await c.get(url, follow_redirects=True); r.raise_for_status()
                    titles += self._parse_rss_titles(r.text, max_results)
            except Exception as e:
                logger.debug("[CNBC] %s", e)
        return list(dict.fromkeys(titles))[:max_results]


# ── MarketWatch RSS ───────────────────────────────────────────────────────────

class MarketWatchFetcher(BaseNewsFetcher):
    name = "marketwatch"
    URL  = "https://feeds.marketwatch.com/marketwatch/topstories/"

    async def fetch(self, query: str, max_results: int = 10) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as c:
                r = await c.get(self.URL, follow_redirects=True); r.raise_for_status()
                return self._parse_rss_titles(r.text, max_results)
        except Exception as e:
            logger.debug("[MarketWatch] %s", e); return []


# ── Seeking Alpha RSS ─────────────────────────────────────────────────────────

class SeekingAlphaFetcher(BaseNewsFetcher):
    name = "seekingalpha"
    URL  = "https://seekingalpha.com/market_currents.xml"

    async def fetch(self, query: str, max_results: int = 10) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as c:
                r = await c.get(self.URL, follow_redirects=True); r.raise_for_status()
                return self._parse_rss_titles(r.text, max_results)
        except Exception as e:
            logger.debug("[SeekingAlpha] %s", e); return []


# ── Yahoo Finance RSS ─────────────────────────────────────────────────────────

class YahooFinanceRSSFetcher(BaseNewsFetcher):
    name = "yahoo_finance"
    URL  = "https://finance.yahoo.com/news/rssindex"

    async def fetch(self, query: str, max_results: int = 10) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as c:
                r = await c.get(self.URL, follow_redirects=True); r.raise_for_status()
                return self._parse_rss_titles(r.text, max_results)
        except Exception as e:
            logger.debug("[YahooFinance] %s", e); return []


# ── Hacker News ───────────────────────────────────────────────────────────────

class HackerNewsFetcher(BaseNewsFetcher):
    name = "hackernews"
    BASE = "https://hn.algolia.com/api/v1/search"

    async def fetch(self, query: str, max_results: int = 10) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
                r = await c.get(self.BASE, params={"query": query, "tags": "story",
                                                    "hitsPerPage": max_results})
                r.raise_for_status()
                return [h["title"] for h in r.json().get("hits",[]) if h.get("title")][:max_results]
        except Exception as e:
            logger.error("[HN] %s", e); return []


# ── Finnhub ───────────────────────────────────────────────────────────────────

class FinnhubNewsFetcher(BaseNewsFetcher):
    name = "finnhub"
    BASE = "https://finnhub.io/api/v1/company-news"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY","")

    async def fetch(self, query: str, max_results: int = 10) -> List[str]:
        if not self.api_key: return []
        from datetime import datetime, timedelta
        ticker  = query.lstrip("$").upper()
        today   = datetime.utcnow().strftime("%Y-%m-%d")
        weekago = (datetime.utcnow()-timedelta(days=7)).strftime("%Y-%m-%d")
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
                r = await c.get(self.BASE, params={"symbol":ticker,"from":weekago,
                                                    "to":today,"token":self.api_key})
                r.raise_for_status()
                return [i["headline"] for i in r.json()[:max_results]]
        except Exception as e:
            logger.error("[Finnhub] %s", e); return []


# ── NewsAPI ───────────────────────────────────────────────────────────────────

class NewsAPIFetcher(BaseNewsFetcher):
    name = "newsapi"
    BASE = "https://newsapi.org/v2/everything"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("NEWSAPI_KEY","")

    async def fetch(self, query: str, max_results: int = 10) -> List[str]:
        if not self.api_key: return []
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
                r = await c.get(self.BASE, params={"q":query.replace("$",""),
                    "sortBy":"publishedAt","pageSize":min(max_results,100),"apiKey":self.api_key})
                r.raise_for_status()
                return [a["title"] for a in r.json().get("articles",[]) if a.get("title")][:max_results]
        except Exception as e:
            logger.error("[NewsAPI] %s", e); return []


# ── CryptoPanic ───────────────────────────────────────────────────────────────

class CryptoPanicFetcher(BaseNewsFetcher):
    name = "cryptopanic"
    BASE = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("CRYPTOPANIC_KEY","")

    async def fetch(self, query: str, max_results: int = 10) -> List[str]:
        params: dict = {"public":"true","filter":"news"}
        if self.api_key: params["auth_token"] = self.api_key
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
                r = await c.get(self.BASE, params=params); r.raise_for_status()
                return [i["title"] for i in r.json().get("results",[])[:max_results]]
        except Exception as e:
            logger.error("[CryptoPanic] %s", e); return []


# ── CompositeFetcher ──────────────────────────────────────────────────────────

class CompositeFetcher(BaseNewsFetcher):
    name = "composite"

    def __init__(self, fetchers: List[BaseNewsFetcher]):
        self.fetchers = fetchers

    async def fetch(self, query: str, max_results: int = 15) -> List[str]:
        tasks   = [f.fetch(query, max_results) for f in self.fetchers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        seen, headlines = set(), []
        for batch in results:
            if isinstance(batch, (Exception, BaseException)): continue
            for h in batch:
                h = h.strip()
                if h and h not in seen: seen.add(h); headlines.append(h)
        logger.info("[Composite] %d unique headlines", len(headlines))
        return headlines[:max_results]


# ── NicheFetcher — smart auto-routing ────────────────────────────────────────

class NicheFetcher(BaseNewsFetcher):
    """
    Auto-selects the best free source stack by niche.

    finance  → GoogleNews + Yahoo + CNBC + MarketWatch + Reuters + BBC + Reddit + Finnhub* + NewsAPI*
    crypto   → GoogleNews + CryptoPanic + Reddit + NewsAPI*
    sports   → GoogleNews + BBC + Reuters + Reddit
    politics → GoogleNews + BBC + Reuters + Reddit
    tech     → GoogleNews + HackerNews + BBC + Reddit
    world    → GoogleNews + BBC + Reuters + Reddit
    general  → GoogleNews + BBC + Reuters + CNBC + MarketWatch
    """
    name = "niche"

    def __init__(self, niche: Niche = "general"):
        self.niche      = niche
        self._composite = CompositeFetcher(self._build(niche))

    @staticmethod
    def _build(niche: Niche) -> List[BaseNewsFetcher]:
        stacks = {
            "finance": [
                GoogleNewsRSSFetcher(),
                YahooFinanceRSSFetcher(),
                CNBCRSSFetcher(),
                MarketWatchFetcher(),
                SeekingAlphaFetcher(),
                ReutersRSSFetcher(niche="finance"),
                BBCRSSFetcher(niche="finance"),
                RedditRSSFetcher(niche="finance"),
                FinnhubNewsFetcher(),
                NewsAPIFetcher(),
            ],
            "crypto": [
                GoogleNewsRSSFetcher(),
                CryptoPanicFetcher(),
                RedditRSSFetcher(niche="crypto"),
                NewsAPIFetcher(),
            ],
            "sports": [
                GoogleNewsRSSFetcher(),
                BBCRSSFetcher(niche="sports"),
                ReutersRSSFetcher(niche="sports"),
                RedditRSSFetcher(niche="sports"),
            ],
            "politics": [
                GoogleNewsRSSFetcher(),
                BBCRSSFetcher(niche="politics"),
                ReutersRSSFetcher(niche="politics"),
                RedditRSSFetcher(niche="politics"),
            ],
            "tech": [
                GoogleNewsRSSFetcher(),
                HackerNewsFetcher(),
                BBCRSSFetcher(niche="tech"),
                RedditRSSFetcher(niche="tech"),
            ],
            "world": [
                GoogleNewsRSSFetcher(),
                BBCRSSFetcher(niche="world"),
                ReutersRSSFetcher(niche="world"),
                RedditRSSFetcher(niche="world"),
            ],
            "general": [
                GoogleNewsRSSFetcher(),
                BBCRSSFetcher(niche="general"),
                ReutersRSSFetcher(niche="general"),
                CNBCRSSFetcher(),
                MarketWatchFetcher(),
            ],
        }
        return stacks.get(niche, stacks["general"])

    async def fetch(self, query: str, max_results: int = 15) -> List[str]:
        return await self._composite.fetch(query, max_results)
