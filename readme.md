# ⟠ ALPHA SENTINEL — Algorithmic Trading Bot

> Production-ready Python trading bot using Alpaca API + RSS News Sentiment + Technical Analysis
> with a live WebSocket-powered HTML dashboard.

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Keys

**Option A — Environment Variables (recommended)**
```bash
export ALPACA_API_KEY="your_key_here"
export ALPACA_SECRET_KEY="your_secret_here"
export PAPER_TRADING=true
```

**Option B — Edit `config.py` directly**
```python
ALPACA_API_KEY    = "your_key_here"
ALPACA_SECRET_KEY = "your_secret_here"
PAPER_TRADING     = True   # Set False for LIVE trading ⚠️
```

Get your keys at: https://alpaca.markets → Your Account → API Keys

### 3. Run the Bot
```bash
python trading_bot.py
```

### 4. Open Dashboard
Open `dashboard.html` in a browser. It auto-connects to `ws://localhost:8765`.

---

## 🏗 Architecture

```
trading_bot.py
│
├── NewsEngine          RSS feed fetcher + keyword sentiment scoring
│   ├── 14 financial RSS feeds (Reuters, CNBC, MarketWatch, WSJ...)
│   ├── Ticker extraction (alias + regex pattern matching)
│   └── Exponentially-weighted sentiment per ticker
│
├── TechnicalAnalysis   Pure-NumPy indicator library
│   ├── RSI (14-period)
│   ├── MACD (12/26/9)
│   ├── Bollinger Bands (20, 2σ)
│   ├── SMA Crossover (9/21)
│   └── ATR (14) — for position sizing
│
├── SignalEngine        Composite scoring model
│   ├── Scores: RSI + MACD + BB + SMA + Volume + Sentiment
│   ├── Output: BUY / SELL / HOLD + confidence score
│   └── Threshold: ±0.7 score → trade, else HOLD
│
├── RiskManager         Capital preservation layer
│   ├── ATR-based position sizing (1% risk/trade)
│   ├── Max 50% portfolio heat
│   ├── Max 8 concurrent positions
│   ├── Stop-loss: 2× ATR below entry
│   ├── Take-profit: 3× ATR above entry
│   └── Circuit breaker: halt if daily P&L < -3%
│
├── BroadcastServer     WebSocket server (port 8765)
│   ├── Full state push on analysis cycle
│   └── Price-only update every 15 seconds
│
└── AlphaSentinel       Main loop (runs every 5 minutes)
    ├── Sync account + positions
    ├── Fetch 60-day OHLCV bars per ticker
    ├── Get latest quotes (real-time)
    ├── Compute signals → execute trades
    └── Push WebSocket payload to dashboard
```

---

## 📊 Dashboard Features

- **Live ticker strip** — scrolling price + score bar
- **Account stats** — equity, buying power, daily P&L, positions
- **Watchlist** — sparklines, signals, RSI, confidence per ticker
- **Open positions** — entry, current price, unrealized P&L with bar visualization
- **Signal breakdown** — click any ticker for RSI, MACD, sentiment, composite score
- **Risk monitor** — portfolio heat, circuit breaker status, position utilization
- **Trade log** — every executed order with details
- **News sentiment feed** — articles with sentiment score + ticker tags

---

## ⚙️ Configuration (config.py)

| Parameter            | Default | Description                              |
|----------------------|---------|------------------------------------------|
| `WATCHLIST`          | 15 stocks| Tickers to monitor and trade            |
| `RISK_PER_TRADE`     | 1%      | Portfolio % risked per trade             |
| `MAX_POSITIONS`      | 8       | Max concurrent open positions            |
| `MAX_PORTFOLIO_RISK` | 50%     | Max portfolio heat (open position value) |
| `TRADE_INTERVAL`     | 300s    | Full cycle frequency (5 min)             |
| `RSI_OVERSOLD`       | 30      | RSI buy threshold                        |
| `RSI_OVERBOUGHT`     | 70      | RSI sell threshold                       |
| `NEWS_SENTIMENT_WEIGHT`| 0.8   | News influence on composite score        |
| `ATR_SL_MULT`        | 2.0     | Stop-loss = ATR × this                   |
| `ATR_TP_MULT`        | 3.0     | Take-profit = ATR × this                 |
| `WS_PORT`            | 8765    | Dashboard WebSocket port                 |

---

## ⚠️ Disclaimer

This bot is for **educational and paper-trading purposes**.
- Always test thoroughly on paper before risking real capital.
- Past performance of any algorithm does not guarantee future results.
- The author is not responsible for any financial losses.
- Always comply with Alpaca's terms of service and applicable laws.

---

## 📁 File Structure

```
trading_bot/
├── trading_bot.py      # Main bot (all logic)
├── config.py           # Configuration
├── requirements.txt    # Python dependencies
├── dashboard.html      # Live monitoring dashboard
├── README.md           # This file
└── trading_bot.log     # Generated at runtime
```