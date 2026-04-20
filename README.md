# Trading Bot — Production Ready

Sentiment-driven stock trading bot powered by **Replicate (Llama 3 70B)** + 13 free RSS news feeds.
Paper trades by default. One variable swap from live trading via Alpaca.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        bot.py (TradingBot)                  │
│                                                             │
│  Guards (every scan cycle):                                 │
│  1. MarketHoursGuard     → NYSE calendar check             │
│  2. DrawdownCircuitBreaker → halt at 3% daily / 10% total  │
│  3. TrendingTickerFilter → 13 free RSS feeds, mentions≥2   │
│  4. DuplicateSignalThrottle → 60 min cooldown              │
│                                                             │
│  Pipeline (per ticker):                                     │
│  news_fetcher → NicheFetcher (composite, 9 free sources)   │
│       ↓                                                     │
│  sentiment_agent → Replicate / Llama-3-70B → JSON score    │
│       ↓                                                     │
│  stock_strategy → BUY / SELL / HOLD / SKIP                 │
│       ↓                                                     │
│  portfolio → execute + stop-loss + take-profit             │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
nano .env   # add REPLICATE_API_TOKEN (required)
```
Get a free Replicate token at: https://replicate.com

### 3. Run (paper mode — no real money)
```bash
# Single scan
python bot.py

# Continuous loop every 15 min
python bot.py --loop

# Custom watchlist + interval
python bot.py --loop --interval 10 --tickers '$NVDA' '$TSLA' '$AAPL'

# Skip market hours guard (for testing on weekends)
python bot.py --no-hours --tickers '$NVDA'
```

### 4. Dashboard API
```bash
uvicorn dashboard.server:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs
```

---

## Deploy to VPS (Ubuntu)

### Option A — One-command deploy (systemd)
```bash
# On your server:
git clone <your-repo> paper_trader
cd paper_trader
cp .env.example .env && nano .env    # fill in your keys
sudo ./deploy.sh
```

### Option B — Docker
```bash
cp .env.example .env && nano .env
docker compose up -d

# View logs
docker compose logs -f bot
```

---

## Go Live (Alpaca)

**Step 1** — Get free Alpaca account: https://alpaca.markets

**Step 2** — Update `.env`:
```bash
PRODUCTION=true
ALPACA_API_KEY=PKxxxx
ALPACA_SECRET_KEY=xxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets   # ← keep this until ready
```

**Step 3** — In `bot.py`, replace the portfolio line:
```python
# Paper (current)
self.portfolio = PaperPortfolio()

# Live (production)
from broker_alpaca import AlpacaBroker
self.portfolio = AlpacaBroker()
```

**Step 4** — Test on Alpaca paper endpoint for 2+ weeks

**Step 5** — When satisfied, change URL to live:
```bash
ALPACA_BASE_URL=https://api.alpaca.markets
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--loop` | off | Run continuously |
| `--interval` | 15 | Minutes between scans |
| `--tickers` | watchlist | Custom tickers to scan |
| `--threshold` | 0.30 | Min sentiment score to act |
| `--no-hours` | off | Disable market hours guard |
| `--daily-dd` | 0.03 | Daily drawdown limit (3%) |
| `--total-dd` | 0.10 | Total drawdown limit (10%) |
| `--trend-min` | 2 | Min RSS mentions to trade ticker |
| `--model` | llama-3-70b | Any Replicate model ID |

---

## Free News Sources (no key needed)

| Source | Feed | Coverage |
|--------|------|----------|
| Google News RSS | 3 finance queries | Any ticker |
| Reuters RSS | Business + wealth | Finance/world |
| BBC Business RSS | business feed | Finance |
| CNBC RSS | markets + money | US markets |
| MarketWatch RSS | top stories | US markets |
| Seeking Alpha RSS | market currents | Stocks |
| Yahoo Finance RSS | news index | Finance |
| Reddit RSS | r/investing + r/stocks | Retail sentiment |

## Optional API Keys (free tiers)

| Key | Source | Benefit |
|-----|--------|---------|
| `FINNHUB_API_KEY` | finnhub.io | Ticker-specific news |
| `NEWSAPI_KEY` | newsapi.org | 100 req/day, broader coverage |
| `CRYPTOPANIC_KEY` | cryptopanic.com | Better crypto signals |

---

## Risk Controls

| Control | Default | File |
|---------|---------|------|
| Risk per trade (Kelly) | 2% | portfolio.py |
| Max open positions | 10 | portfolio.py |
| Stop-loss | 5% | portfolio.py |
| Take-profit | 10% | portfolio.py |
| Daily drawdown limit | 3% | guards.py |
| Total drawdown limit | 10% | guards.py |
| Circuit breaker cooldown | 60 min | guards.py |
| Signal cooldown per ticker | 60 min | guards.py |
| Min news mentions to trade | 2 | guards.py |

---

## Production Checklist

- [ ] Paper trade for ≥ 2 weeks
- [ ] Win rate > 55% on ≥ 20 closed trades
- [ ] Max drawdown < 10% in any test period
- [ ] Switch Alpaca URL from paper → live
- [ ] Lower `risk_per_trade` to 0.5% for first live week
- [ ] Set up server monitoring (UptimeRobot or similar)
- [ ] Add Slack/email alerts for circuit breaker trips

---

## Disclaimer
Educational software only. Not financial advice.
Past simulated performance does not guarantee future returns.
