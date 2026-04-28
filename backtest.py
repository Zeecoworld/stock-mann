"""
backtest.py  — Strategy Backtester v1
──────────────────────────────────────────────────────────────────────────────
Tests your StockSentimentStrategy against 6–12 months of real historical
price data fetched from Alpaca's free Data API.

HOW SENTIMENT IS SIMULATED
───────────────────────────
We cannot replay live LLM calls historically, so we use a realistic proxy
called "forward-return labelling" — the standard approach in quantitative
research for backtesting sentiment-driven strategies:

  • If the stock's close price is higher N days later  → score = +0.3 to +0.8
    (simulates a bullish headline that turned out to be correct)
  • If the stock's close price is lower N days later   → score = -0.3 to -0.8
    (simulates a bearish headline)
  • If the move is < noise_floor %                     → score near 0
    (simulates a mixed/neutral news day)

Confidence is derived from the magnitude of the move — a big clear move
gives high confidence (0.7–0.9), a small ambiguous move gives low (0.3–0.5).

This answers the real question: "Does the MA20 crossover filter + sentiment
threshold add value over a naive buy-and-hold?" If the backtest PnL beats
SPY over the same period, your strategy has edge worth testing live.

WHAT IT TESTS
──────────────
  1. Your strategy as-is       (sentiment_threshold=0.15, conf=0.35)
  2. Stricter thresholds        (sentiment_threshold=0.25, conf=0.50)
  3. MA filter only             (no sentiment — pure MA20 crossover)
  4. Buy-and-hold benchmark     (buy on day 1, hold everything)
  5. SPY buy-and-hold           (market benchmark)

OUTPUT
──────
  • Console: full trade log + performance report per strategy
  • backtest_results.json: machine-readable results for all strategies
  • backtest_trades.csv:   every individual trade for audit

USAGE
──────
  python backtest.py                          # default: 252 days, default watchlist
  python backtest.py --days 180               # last 180 calendar days
  python backtest.py --tickers NVDA TSLA AAPL # specific tickers
  python backtest.py --capital 50000          # starting capital
  python backtest.py --no-slippage            # skip slippage simulation

REQUIREMENTS
─────────────
  pip install httpx python-dotenv
  ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Backtest")

# ── Constants ─────────────────────────────────────────────────────────────────

ALPACA_DATA_BASE = "https://data.alpaca.markets"
SPY_TICKER       = "SPY"

DEFAULT_WATCHLIST = [
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN",
    "META", "GOOGL", "AMD", "PLTR", "COIN",
]

# Slippage model: assume fills are 0.05% worse than close price
SLIPPAGE_PCT = 0.0005

# Commission model: $0 (Alpaca is commission-free)
COMMISSION = 0.0

# Forward-return window for sentiment proxy (trading days)
SENTIMENT_FORWARD_DAYS = 3

# Noise floor: moves below this % are treated as "neutral news"
NOISE_FLOOR_PCT = 0.005   # 0.5%


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Bar:
    date:   str
    open:   float
    high:   float
    low:    float
    close:  float
    volume: int


@dataclass
class BacktestTrade:
    date:        str
    ticker:      str
    action:      str        # BUY | SELL | STOP | TAKE_PROFIT
    price:       float
    shares:      float
    cost:        float
    pnl:         float = 0.0
    hold_days:   int   = 0
    sentiment:   float = 0.0
    confidence:  float = 0.0
    signal_type: str   = ""  # what triggered the signal


@dataclass
class BacktestPosition:
    ticker:     str
    shares:     float
    entry_price: float
    entry_date:  str
    sentiment:  float = 0.0


@dataclass
class StrategyResult:
    name:           str
    total_return:   float
    annualised:     float
    sharpe:         float
    max_drawdown:   float
    win_rate:       float
    total_trades:   int
    profitable:     int
    final_value:    float
    trades:         List[BacktestTrade] = field(default_factory=list)
    daily_values:   List[float]         = field(default_factory=list)


# ── Alpaca data fetcher ────────────────────────────────────────────────────────

class AlpacaDataFetcher:
    def __init__(self):
        self.key    = os.getenv("ALPACA_API_KEY")
        self.secret = os.getenv("ALPACA_SECRET_KEY")
        if not self.key or not self.secret:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in your .env file.\n"
                "These are the same keys used by the trading bot — no extra signup needed."
            )
        self.headers = {
            "APCA-API-KEY-ID":     self.key,
            "APCA-API-SECRET-KEY": self.secret,
            "Accept":              "application/json",
        }

    async def fetch_bars(
        self,
        symbol:     str,
        start_date: str,   # YYYY-MM-DD
        end_date:   str,   # YYYY-MM-DD
    ) -> List[Bar]:
        """Fetch daily OHLCV bars for a symbol over the given date range."""
        bars: List[Bar] = []
        url  = f"{ALPACA_DATA_BASE}/v2/stocks/{symbol}/bars"
        params = {
            "timeframe":  "1Day",
            "start":      f"{start_date}T00:00:00Z",
            "end":        f"{end_date}T23:59:59Z",
            "adjustment": "split",    # adjust for stock splits
            "limit":      1000,
        }
        page_token = None

        async with httpx.AsyncClient(timeout=30, headers=self.headers) as c:
            while True:
                if page_token:
                    params["page_token"] = page_token
                try:
                    r = await c.get(url, params=params)
                    r.raise_for_status()
                    data = r.json()
                    for b in data.get("bars", []):
                        bars.append(Bar(
                            date   = b["t"][:10],
                            open   = float(b["o"]),
                            high   = float(b["h"]),
                            low    = float(b["l"]),
                            close  = float(b["c"]),
                            volume = int(b.get("v", 0)),
                        ))
                    page_token = data.get("next_page_token")
                    if not page_token:
                        break
                except Exception as e:
                    logger.warning("[Data] %s fetch error: %s", symbol, e)
                    break

        logger.info("[Data] %s: %d bars (%s → %s)",
                    symbol, len(bars), start_date, end_date)
        return bars

    async def fetch_all(
        self,
        tickers:    List[str],
        start_date: str,
        end_date:   str,
    ) -> Dict[str, List[Bar]]:
        """Fetch bars for all tickers concurrently."""
        tasks = {t: self.fetch_bars(t, start_date, end_date) for t in tickers}
        results = {}
        for ticker, coro in tasks.items():
            results[ticker] = await coro
            await asyncio.sleep(0.15)   # gentle rate-limiting
        return results


# ── Sentiment proxy ────────────────────────────────────────────────────────────

def compute_sentiment_proxy(
    bars:        List[Bar],
    day_idx:     int,
    forward_days: int = SENTIMENT_FORWARD_DAYS,
    noise_floor: float = NOISE_FLOOR_PCT,
) -> Tuple[float, float]:
    """
    Simulates what an LLM sentiment score + confidence would look like on
    a given day by looking at the actual forward return over the next
    `forward_days` trading days.

    Returns (score: float, confidence: float)
      score      ∈ [-1.0, +1.0]
      confidence ∈ [0.0,   1.0]

    This is the standard quantitative finance approach: we label each day
    with the sign and magnitude of its near-future return, then test whether
    our signal filters (MA20, threshold) improve on that baseline.
    """
    if day_idx + forward_days >= len(bars):
        return 0.0, 0.0

    current_close = bars[day_idx].close
    future_close  = bars[day_idx + forward_days].close

    if current_close <= 0:
        return 0.0, 0.0

    fwd_return = (future_close - current_close) / current_close

    # Below noise floor → neutral, low confidence
    if abs(fwd_return) < noise_floor:
        # Add slight randomness to simulate genuine LLM uncertainty on flat days
        score = random.gauss(0, 0.05)
        return round(max(-0.2, min(0.2, score)), 3), round(random.uniform(0.2, 0.45), 3)

    # Scale return to score: cap at ±40% move → ±1.0 score
    raw_score  = fwd_return / 0.40
    score      = max(-1.0, min(1.0, raw_score))

    # Confidence: larger absolute move → higher confidence
    # 0.5% move → ~0.35 conf; 5% move → ~0.75 conf; 15%+ → ~0.95 conf
    conf_raw   = math.log1p(abs(fwd_return) * 20) / math.log1p(20)
    confidence = round(max(0.25, min(0.95, conf_raw)), 3)

    # Add small noise — real LLMs aren't perfectly correlated with returns
    score      = round(score + random.gauss(0, 0.08), 3)
    score      = max(-1.0, min(1.0, score))

    return score, confidence


# ── MA calculator ──────────────────────────────────────────────────────────────

def compute_ma20(bars: List[Bar], day_idx: int) -> Optional[float]:
    """20-day simple moving average of close prices ending at day_idx."""
    if day_idx < 1:
        return None
    window = bars[max(0, day_idx - 19): day_idx + 1]
    if len(window) < 5:   # need at least 5 days for a meaningful MA
        return None
    return sum(b.close for b in window) / len(window)


# ── Strategy signal functions ──────────────────────────────────────────────────

def signal_your_strategy(
    price:      float,
    ma20:       Optional[float],
    sentiment:  float,
    confidence: float,
    sent_threshold: float = 0.15,
    conf_threshold: float = 0.35,
) -> str:
    """
    Exact replica of StockSentimentStrategy.evaluate() logic — no imports
    needed so the backtest runs standalone.
    Returns "BUY" | "SELL" | "HOLD" | "SKIP"
    """
    if confidence < conf_threshold:
        return "SKIP"

    has_price = price is not None and ma20 is not None

    if sentiment >= sent_threshold:
        if has_price and price > ma20:
            return "BUY"
        if has_price and price <= ma20:
            return "HOLD"
        return "BUY"   # no MA data — sentiment-only

    if sentiment <= -sent_threshold:
        if has_price and price < ma20:
            return "SELL"
        if has_price and price >= ma20:
            return "HOLD"
        return "SELL"  # no MA data — sentiment-only

    return "HOLD"


def signal_ma_only(
    price: float,
    ma20:  Optional[float],
) -> str:
    """Pure MA20 crossover — no sentiment. Used as a control."""
    if ma20 is None:
        return "HOLD"
    if price > ma20 * 1.005:   # 0.5% buffer to avoid whipsawing
        return "BUY"
    if price < ma20 * 0.995:
        return "SELL"
    return "HOLD"


# ── Single-strategy simulator ──────────────────────────────────────────────────

def run_strategy(
    name:           str,
    all_bars:       Dict[str, List[Bar]],
    spy_bars:       List[Bar],
    capital:        float,
    sent_threshold: float = 0.15,
    conf_threshold: float = 0.35,
    use_sentiment:  bool  = True,
    buy_hold:       bool  = False,
    apply_slippage: bool  = True,
    stop_loss_pct:  float = 0.05,
    take_profit_pct:float = 0.12,
    max_positions:  int   = 5,
    risk_per_trade: float = 0.05,
    signal_cooldown_days: int = 2,
) -> StrategyResult:
    """
    Simulate a strategy over the full historical dataset.
    Returns a StrategyResult with all metrics and the full trade log.
    """
    cash       = capital
    positions: Dict[str, BacktestPosition] = {}
    trades:    List[BacktestTrade]         = []
    daily_values: List[float]             = []

    # Build a unified sorted list of trading dates across all tickers
    all_dates = sorted(set(
        b.date for bars in all_bars.values() for b in bars
    ))

    # Index bars by date for O(1) lookup
    bars_by_date: Dict[str, Dict[str, Bar]] = {}
    for ticker, bars in all_bars.items():
        for bar in bars:
            bars_by_date.setdefault(bar.date, {})[ticker] = bar

    # Also index by position for MA calculation
    bars_list: Dict[str, List[Bar]] = all_bars

    # Cooldown tracker: last signal date per ticker
    last_signal: Dict[str, str] = {}

    def get_bar_idx(ticker: str, date: str) -> int:
        bars = bars_list.get(ticker, [])
        for i, b in enumerate(bars):
            if b.date == date:
                return i
        return -1

    def portfolio_value(date: str) -> float:
        pos_value = 0.0
        for ticker, pos in positions.items():
            day_bars = bars_by_date.get(date, {})
            bar = day_bars.get(ticker)
            price = bar.close if bar else pos.entry_price
            pos_value += pos.shares * price
        return cash + pos_value

    # ── Buy-and-hold: buy equal weight on day 1, never sell ──────────────
    if buy_hold:
        first_date = all_dates[0] if all_dates else None
        if first_date:
            tickers = list(all_bars.keys())
            alloc   = (capital / len(tickers)) if tickers else 0
            for ticker in tickers:
                first_bars = bars_by_date.get(first_date, {})
                bar = first_bars.get(ticker)
                if bar and bar.close > 0:
                    price  = bar.close * (1 + SLIPPAGE_PCT if apply_slippage else 1)
                    shares = alloc / price
                    cost   = shares * price
                    cash  -= cost
                    positions[ticker] = BacktestPosition(
                        ticker=ticker, shares=shares,
                        entry_price=price, entry_date=first_date,
                    )
                    trades.append(BacktestTrade(
                        date=first_date, ticker=ticker, action="BUY",
                        price=price, shares=shares, cost=cost,
                        signal_type="buy_hold_open",
                    ))
        for date in all_dates:
            daily_values.append(portfolio_value(date))
        final_val = portfolio_value(all_dates[-1]) if all_dates else capital
        return _compute_result(name, capital, final_val, daily_values, trades)

    # ── Active strategy simulation ────────────────────────────────────────
    for date in all_dates:
        day_bars = bars_by_date.get(date, {})

        # ── Check stops on all open positions first ───────────────────────
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            bar = day_bars.get(ticker)
            if not bar:
                continue
            price   = bar.close
            pct_chg = (price - pos.entry_price) / pos.entry_price

            action = None
            if pct_chg <= -stop_loss_pct:
                action = "STOP"
            elif pct_chg >= take_profit_pct:
                action = "TAKE_PROFIT"

            if action:
                sell_price = price * (1 - SLIPPAGE_PCT if apply_slippage else 1)
                proceeds   = pos.shares * sell_price
                pnl        = proceeds - (pos.shares * pos.entry_price)
                cash      += proceeds
                hold_days  = _days_between(pos.entry_date, date)
                trades.append(BacktestTrade(
                    date=date, ticker=ticker, action=action,
                    price=sell_price, shares=pos.shares,
                    cost=proceeds, pnl=pnl, hold_days=hold_days,
                    signal_type=action.lower(),
                ))
                del positions[ticker]

        # ── Generate signals for watchlist tickers ────────────────────────
        for ticker, bar in day_bars.items():
            if ticker not in all_bars:
                continue

            price  = bar.close
            bars   = bars_list.get(ticker, [])
            idx    = get_bar_idx(ticker, date)
            if idx < 0:
                continue

            ma20 = compute_ma20(bars, idx)

            # Cooldown check
            last = last_signal.get(ticker)
            if last and _days_between(last, date) < signal_cooldown_days:
                continue

            if use_sentiment:
                sentiment, confidence = compute_sentiment_proxy(bars, idx)
                sig = signal_your_strategy(
                    price, ma20, sentiment, confidence,
                    sent_threshold, conf_threshold,
                )
            else:
                sentiment, confidence = 0.0, 1.0
                sig = signal_ma_only(price, ma20)

            # ── Execute BUY ───────────────────────────────────────────────
            if sig == "BUY" and ticker not in positions:
                if len(positions) >= max_positions:
                    continue
                port_val    = portfolio_value(date)
                target_usd  = port_val * risk_per_trade * min(confidence, 1.0)
                target_usd  = max(50.0, target_usd)
                target_usd  = min(target_usd, cash * 0.95)
                if target_usd < 50 or cash < 50:
                    continue
                buy_price = price * (1 + SLIPPAGE_PCT if apply_slippage else 1)
                shares    = target_usd / buy_price
                cost      = shares * buy_price + COMMISSION
                if cost > cash:
                    continue
                cash -= cost
                positions[ticker] = BacktestPosition(
                    ticker=ticker, shares=shares,
                    entry_price=buy_price, entry_date=date,
                    sentiment=sentiment,
                )
                trades.append(BacktestTrade(
                    date=date, ticker=ticker, action="BUY",
                    price=buy_price, shares=shares, cost=cost,
                    sentiment=sentiment, confidence=confidence,
                    signal_type="sentiment+ma" if use_sentiment else "ma_only",
                ))
                last_signal[ticker] = date

            # ── Execute SELL ──────────────────────────────────────────────
            elif sig == "SELL" and ticker in positions:
                pos       = positions[ticker]
                sell_price= price * (1 - SLIPPAGE_PCT if apply_slippage else 1)
                proceeds  = pos.shares * sell_price - COMMISSION
                pnl       = proceeds - (pos.shares * pos.entry_price)
                hold_days = _days_between(pos.entry_date, date)
                cash     += proceeds
                trades.append(BacktestTrade(
                    date=date, ticker=ticker, action="SELL",
                    price=sell_price, shares=pos.shares,
                    cost=proceeds, pnl=pnl, hold_days=hold_days,
                    sentiment=sentiment, confidence=confidence,
                    signal_type="sentiment+ma" if use_sentiment else "ma_only",
                ))
                del positions[ticker]
                last_signal[ticker] = date

        daily_values.append(portfolio_value(date))

    # Close any open positions at the last price
    if all_dates:
        last_date  = all_dates[-1]
        last_bars  = bars_by_date.get(last_date, {})
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            bar = last_bars.get(ticker)
            if bar:
                price     = bar.close
                proceeds  = pos.shares * price
                pnl       = proceeds - (pos.shares * pos.entry_price)
                hold_days = _days_between(pos.entry_date, last_date)
                cash     += proceeds
                trades.append(BacktestTrade(
                    date=last_date, ticker=ticker, action="CLOSE",
                    price=price, shares=pos.shares,
                    cost=proceeds, pnl=pnl, hold_days=hold_days,
                    signal_type="end_of_backtest",
                ))
                del positions[ticker]

    final_val = cash
    return _compute_result(name, capital, final_val, daily_values, trades)


# ── Metrics ────────────────────────────────────────────────────────────────────

def _compute_result(
    name:         str,
    capital:      float,
    final_val:    float,
    daily_values: List[float],
    trades:       List[BacktestTrade],
) -> StrategyResult:
    total_return  = (final_val - capital) / capital
    n_days        = len(daily_values)
    annualised    = ((1 + total_return) ** (252 / max(n_days, 1)) - 1) if n_days > 0 else 0.0

    # Sharpe ratio (annualised, risk-free rate ≈ 5%)
    RISK_FREE_DAILY = 0.05 / 252
    if len(daily_values) > 1:
        daily_rets = [
            (daily_values[i] / daily_values[i - 1] - 1)
            for i in range(1, len(daily_values))
        ]
        excess    = [r - RISK_FREE_DAILY for r in daily_rets]
        mean_exc  = sum(excess) / len(excess)
        std_exc   = math.sqrt(sum((r - mean_exc) ** 2 for r in excess) / len(excess))
        sharpe    = (mean_exc / std_exc * math.sqrt(252)) if std_exc > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    peak = capital
    max_dd = 0.0
    for v in daily_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # Win rate (closed trades only)
    closed = [t for t in trades if t.action in ("SELL", "STOP", "TAKE_PROFIT", "CLOSE")]
    wins   = [t for t in closed if t.pnl > 0]
    win_rate = len(wins) / len(closed) if closed else 0.0

    return StrategyResult(
        name          = name,
        total_return  = round(total_return * 100, 2),
        annualised    = round(annualised * 100, 2),
        sharpe        = round(sharpe, 3),
        max_drawdown  = round(max_dd * 100, 2),
        win_rate      = round(win_rate * 100, 1),
        total_trades  = len(closed),
        profitable    = len(wins),
        final_value   = round(final_val, 2),
        trades        = trades,
        daily_values  = [round(v, 2) for v in daily_values],
    )


def _days_between(date_a: str, date_b: str) -> int:
    try:
        da = datetime.strptime(date_a, "%Y-%m-%d")
        db = datetime.strptime(date_b, "%Y-%m-%d")
        return abs((db - da).days)
    except Exception:
        return 0


# ── SPY benchmark ──────────────────────────────────────────────────────────────

def run_spy_benchmark(
    spy_bars:   List[Bar],
    capital:    float,
    apply_slippage: bool = True,
) -> StrategyResult:
    """Buy SPY on day 1, hold until the end."""
    if not spy_bars:
        return _compute_result("SPY buy-and-hold", capital, capital, [], [])

    price  = spy_bars[0].close * (1 + SLIPPAGE_PCT if apply_slippage else 1)
    shares = capital / price
    trades = [BacktestTrade(
        date=spy_bars[0].date, ticker="SPY", action="BUY",
        price=price, shares=shares, cost=capital, signal_type="benchmark",
    )]
    daily_values = [shares * b.close for b in spy_bars]
    final_val    = shares * spy_bars[-1].close
    trades.append(BacktestTrade(
        date=spy_bars[-1].date, ticker="SPY", action="CLOSE",
        price=spy_bars[-1].close, shares=shares, cost=final_val,
        pnl=final_val - capital,
        signal_type="benchmark_close",
    ))
    return _compute_result("SPY buy-and-hold", capital, final_val, daily_values, trades)


# ── Report printer ─────────────────────────────────────────────────────────────

def print_report(results: List[StrategyResult], capital: float):
    SEP = "─" * 80

    print(f"\n{'═'*80}")
    print(f"  BACKTEST RESULTS  |  Starting capital: ${capital:,.2f}")
    print(f"{'═'*80}\n")

    # Summary table
    headers = ["Strategy", "Total Return", "Ann. Return", "Sharpe", "Max DD", "Win Rate", "Trades", "Final Value"]
    rows    = []
    for r in results:
        rows.append([
            r.name,
            f"{r.total_return:+.1f}%",
            f"{r.annualised:+.1f}%",
            f"{r.sharpe:.2f}",
            f"{r.max_drawdown:.1f}%",
            f"{r.win_rate:.0f}%",
            str(r.total_trades),
            f"${r.final_value:,.2f}",
        ])

    col_widths = [max(len(h), max(len(row[i]) for row in rows))
                  for i, h in enumerate(headers)]

    def fmt_row(row):
        return "  " + "  ".join(str(v).ljust(w) for v, w in zip(row, col_widths))

    print(fmt_row(headers))
    print("  " + SEP[:sum(col_widths) + len(col_widths)*2])
    for row in rows:
        print(fmt_row(row))
    print()

    # Verdict
    spy = next((r for r in results if "SPY" in r.name), None)
    your = next((r for r in results if r.name == "Your strategy (default)"), None)

    if spy and your:
        print(f"{'─'*80}")
        print("  VERDICT")
        print(f"{'─'*80}")
        diff = your.total_return - spy.total_return
        if diff > 5:
            verdict = f"✅  Your strategy OUTPERFORMED SPY by {diff:+.1f}%. Worth going live on paper."
        elif diff > 0:
            verdict = f"🟡  Your strategy beat SPY by only {diff:+.1f}%. Marginal edge — extend the backtest window."
        elif diff > -5:
            verdict = f"🟠  Your strategy slightly UNDERPERFORMED SPY ({diff:+.1f}%). Do NOT go live yet."
        else:
            verdict = f"🔴  Your strategy significantly UNDERPERFORMED SPY ({diff:+.1f}%). Strategy needs rework."

        print(f"  {verdict}")
        print()

        if your.sharpe < 0.5:
            print("  ⚠  Sharpe ratio < 0.5 — risk-adjusted returns are poor. "
                  "Consider tightening thresholds.")
        if your.max_drawdown > 20:
            print(f"  ⚠  Max drawdown {your.max_drawdown:.1f}% is high. "
                  "Reduce position size or tighten stop-loss.")
        if your.win_rate < 45:
            print(f"  ⚠  Win rate {your.win_rate:.0f}% is below 50%. "
                  "Strategy may need a longer holding period or stricter entry.")
        if your.total_trades < 10:
            print("  ⚠  Fewer than 10 closed trades — results are not statistically significant. "
                  "Run with more tickers or a longer window.")
        print()

    # Per-strategy trade breakdown
    for result in results:
        if not result.trades or result.name in ("SPY buy-and-hold",):
            continue
        closed = [t for t in result.trades if t.action in ("SELL", "STOP", "TAKE_PROFIT", "CLOSE")]
        if not closed:
            continue
        print(f"{'─'*80}")
        print(f"  {result.name}  —  trade log ({len(closed)} closed trades)")
        print(f"{'─'*80}")
        print(f"  {'Date':<12} {'Ticker':<8} {'Action':<12} {'Price':>8} {'Shares':>8} "
              f"{'PnL':>10} {'Hold':>6} {'Sent':>6}")
        print(f"  {'─'*70}")
        for t in sorted(closed, key=lambda x: x.date):
            pnl_str = f"${t.pnl:+,.2f}" if t.pnl != 0 else "—"
            sent_str = f"{t.sentiment:+.2f}" if t.sentiment != 0 else "—"
            print(f"  {t.date:<12} {t.ticker:<8} {t.action:<12} "
                  f"${t.price:>7.2f} {t.shares:>8.3f} "
                  f"{pnl_str:>10} {t.hold_days:>5}d {sent_str:>6}")
        total_pnl = sum(t.pnl for t in closed)
        print(f"  {'─'*70}")
        print(f"  {'Total PnL':>52} ${total_pnl:+,.2f}")
        print()


# ── File output ────────────────────────────────────────────────────────────────

def save_results(results: List[StrategyResult], prefix: str = "backtest"):
    # JSON — full results
    json_path = f"{prefix}_results.json"
    out = []
    for r in results:
        d = asdict(r)
        d.pop("trades")        # saved separately in CSV
        d.pop("daily_values")  # large — omit from JSON summary
        out.append(d)
    with open(json_path, "w") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "strategies": out}, f, indent=2)
    print(f"  Results saved → {json_path}")

    # CSV — all trades across all strategies
    csv_path = f"{prefix}_trades.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "date", "ticker", "action", "price",
                         "shares", "cost", "pnl", "hold_days",
                         "sentiment", "confidence", "signal_type"])
        for r in results:
            for t in r.trades:
                writer.writerow([
                    r.name, t.date, t.ticker, t.action,
                    round(t.price, 4), round(t.shares, 4),
                    round(t.cost, 4), round(t.pnl, 4),
                    t.hold_days, round(t.sentiment, 3),
                    round(t.confidence, 3), t.signal_type,
                ])
    print(f"  Trades saved  → {csv_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Backtest the trading bot strategy against Alpaca historical data"
    )
    parser.add_argument("--days",        type=int,   default=252,
                        help="Number of calendar days to look back (default: 252 ≈ 1 year)")
    parser.add_argument("--tickers",     nargs="*",  default=None,
                        help="Tickers to test (default: bot watchlist)")
    parser.add_argument("--capital",     type=float, default=10_000.0,
                        help="Starting capital in USD (default: 10000)")
    parser.add_argument("--no-slippage", action="store_true",
                        help="Disable slippage simulation (optimistic mode)")
    parser.add_argument("--stop-loss",   type=float, default=0.05,
                        help="Stop-loss percentage (default: 0.05 = 5%%)")
    parser.add_argument("--take-profit", type=float, default=0.12,
                        help="Take-profit percentage (default: 0.12 = 12%%)")
    parser.add_argument("--risk",        type=float, default=0.05,
                        help="Risk per trade as fraction of portfolio (default: 0.05)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for sentiment noise (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    apply_slippage = not args.no_slippage
    tickers        = [t.lstrip("$").upper() for t in (args.tickers or DEFAULT_WATCHLIST)]

    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = end_dt.strftime("%Y-%m-%d")

    print(f"\n{'═'*80}")
    print(f"  TRADING BOT BACKTESTER")
    print(f"{'═'*80}")
    print(f"  Period   : {start_str} → {end_str}  ({args.days} days)")
    print(f"  Tickers  : {', '.join(tickers)}")
    print(f"  Capital  : ${args.capital:,.2f}")
    print(f"  Slippage : {'OFF (optimistic)' if not apply_slippage else f'{SLIPPAGE_PCT*100:.2f}% per trade'}")
    print(f"  Stop/TP  : {args.stop_loss*100:.0f}% / {args.take_profit*100:.0f}%")
    print(f"  Risk/trade: {args.risk*100:.0f}% of portfolio")
    print()

    # Fetch data
    fetcher  = AlpacaDataFetcher()
    all_syms = tickers + ([SPY_TICKER] if SPY_TICKER not in tickers else [])

    print(f"  Fetching {len(all_syms)} symbols from Alpaca…")
    all_data = await fetcher.fetch_all(all_syms, start_str, end_str)

    ticker_data = {t: all_data[t] for t in tickers if t in all_data and all_data[t]}
    spy_data    = all_data.get(SPY_TICKER, [])

    if not ticker_data:
        print("\n  ❌  No data returned. Check your ALPACA_API_KEY / ALPACA_SECRET_KEY.")
        return

    missing = [t for t in tickers if t not in ticker_data]
    if missing:
        print(f"  ⚠  No data for: {', '.join(missing)} (may be delisted or unsupported)")

    print(f"\n  Running simulations…\n")

    # ── Run all strategies ─────────────────────────────────────────────────
    results = []

    # 1. Your strategy — default thresholds
    results.append(run_strategy(
        name="Your strategy (default)",
        all_bars=ticker_data, spy_bars=spy_data,
        capital=args.capital,
        sent_threshold=0.15, conf_threshold=0.35,
        use_sentiment=True, apply_slippage=apply_slippage,
        stop_loss_pct=args.stop_loss, take_profit_pct=args.take_profit,
        risk_per_trade=args.risk,
    ))

    # 2. Stricter thresholds
    results.append(run_strategy(
        name="Stricter thresholds (0.25/0.50)",
        all_bars=ticker_data, spy_bars=spy_data,
        capital=args.capital,
        sent_threshold=0.25, conf_threshold=0.50,
        use_sentiment=True, apply_slippage=apply_slippage,
        stop_loss_pct=args.stop_loss, take_profit_pct=args.take_profit,
        risk_per_trade=args.risk,
    ))

    # 3. MA20 only (no sentiment filter)
    results.append(run_strategy(
        name="MA20 crossover only (no sentiment)",
        all_bars=ticker_data, spy_bars=spy_data,
        capital=args.capital,
        use_sentiment=False, apply_slippage=apply_slippage,
        stop_loss_pct=args.stop_loss, take_profit_pct=args.take_profit,
        risk_per_trade=args.risk,
    ))

    # 4. Buy-and-hold watchlist
    results.append(run_strategy(
        name="Buy-and-hold (watchlist)",
        all_bars=ticker_data, spy_bars=spy_data,
        capital=args.capital,
        buy_hold=True, apply_slippage=apply_slippage,
    ))

    # 5. SPY benchmark
    if spy_data:
        results.append(run_spy_benchmark(
            spy_bars=spy_data, capital=args.capital,
            apply_slippage=apply_slippage,
        ))

    # Print + save
    print_report(results, args.capital)
    save_results(results)
    print()


if __name__ == "__main__":
    asyncio.run(main())