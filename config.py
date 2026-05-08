"""
Alpha Sentinel — Configuration
================================
Edit this file before running the bot.
Never commit API keys to version control.
"""

import os

# ─── Alpaca API Credentials ──────────────────────────────────────────────────
# Get keys from: https://alpaca.markets → Your Account → API Keys
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY",    "YOUR_ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY")
PAPER_TRADING     = os.getenv("PAPER_TRADING", "true").lower() == "true"

# ─── Watchlist ───────────────────────────────────────────────────────────────
# US equities to monitor and trade
WATCHLIST = [
    "AAPL",   # Apple
    "NVDA",   # NVIDIA
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "TSLA",   # Tesla
    "META",   # Meta
    "AMD",    # AMD
    "NFLX",   # Netflix
    "PLTR",   # Palantir
    "INTC",   # Intel
    "CRM",    # Salesforce
    "AVGO",   # Broadcom
    "QCOM",   # Qualcomm
    "JPM",    # JPMorgan
]

# ─── RSS News Feeds ───────────────────────────────────────────────────────────
# Mix of market-moving financial news sources
RSS_FEEDS = [
    # MarketWatch
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",

    # CNBC
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # Top News
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",   # US Business
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",   # Technology

    # Seeking Alpha (public)
    "https://seekingalpha.com/market_currents.xml",

    # Investopedia
    "https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=rss_articles",

    # Yahoo Finance
    "https://finance.yahoo.com/rss/topstories",

    # The Wall Street Journal (public RSS)
    "https://feeds.wsj.com/wsj/xml/rss/3_7085.xml",

    # Bloomberg (limited public)
    "https://feeds.bloomberg.com/markets/news.rss",

    # Barron's
    "https://www.barrons.com/xml/rss/3_7518.xml",

    # Business Insider
    "https://markets.businessinsider.com/rss/news",

    # Financial Times (public headlines)
    "https://www.ft.com/rss/home",

    # SEC EDGAR — official filings
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=20&search_text=&output=atom",
]

# ─── Risk Management ─────────────────────────────────────────────────────────
RISK_PER_TRADE      = 0.01    # 1% of equity per trade (position sizing)
MAX_PORTFOLIO_RISK  = 0.50    # Max 50% of equity in open positions
MAX_POSITIONS       = 8       # Max concurrent open positions
DAILY_LOSS_LIMIT    = 0.03    # Circuit breaker: stop trading after -3% daily loss

# ─── Signal Weights ───────────────────────────────────────────────────────────
NEWS_SENTIMENT_WEIGHT = 0.8   # How much news sentiment influences score (0-1)
TECHNICAL_WEIGHT      = 1.0   # Technical analysis base weight

# ─── Technical Indicator Parameters ──────────────────────────────────────────
RSI_OVERSOLD   = 30           # RSI below this = buy signal
RSI_OVERBOUGHT = 70           # RSI above this = sell signal
RSI_PERIOD     = 14

MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9

BB_PERIOD  = 20
BB_STD_DEV = 2.0

SMA_FAST = 9
SMA_SLOW = 21

ATR_PERIOD     = 14
ATR_SL_MULT    = 2.0   # Stop loss = ATR × multiplier
ATR_TP_MULT    = 3.0   # Take profit = ATR × multiplier

# ─── Timing ───────────────────────────────────────────────────────────────────
TRADE_INTERVAL    = 300   # Full analysis cycle every 5 minutes (300s)
NEWS_REFRESH      = 300   # News fetch every 5 minutes
PRICE_UPDATE      = 15    # Live price push to dashboard every 15s

# ─── Dashboard WebSocket ──────────────────────────────────────────────────────
WS_PORT = 8765            # Dashboard WebSocket port
WS_HOST = "0.0.0.0"

# ─── Logging ──────────────────────────────────────────────────────────────────
# Stdout only — no log file. Render captures stdout in its native log dashboard.
LOG_LEVEL = "INFO"

# ─── Market Hours (Eastern Time) ─────────────────────────────────────────────
MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MIN    = 30
MARKET_CLOSE_HOUR  = 16
MARKET_CLOSE_MIN   = 0
TRADE_ONLY_MARKET_HOURS = False  # Set True to restrict trades to market hours