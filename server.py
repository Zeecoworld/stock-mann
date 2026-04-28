"""
dashboard/server.py
FastAPI backend for the trading dashboard.

Run: uvicorn dashboard.server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio, json, os, sys, pathlib, time, logging

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from bot import TradingBot, DEFAULT_WATCHLIST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Dashboard")

app = FastAPI(title="Trading Bot Dashboard", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Global state ─────────────────────────────────────────────────────────────
_bot:           Optional[TradingBot] = None
_last_snapshot: dict  = {}
_is_scanning:   bool  = False
_scan_count:    int   = 0
_started_at:    float = time.time()


def get_bot() -> TradingBot:
    global _bot
    if _bot is None:
        _bot = TradingBot()
    return _bot


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/snapshot")
async def snapshot():
    global _last_snapshot
    bot = get_bot()
    if not _last_snapshot:
        _last_snapshot = bot._snapshot()
    return JSONResponse({
        **_last_snapshot,
        "is_scanning": _is_scanning,
        "scan_count":  _scan_count,
        "uptime_sec":  int(time.time() - _started_at),
    })


@app.post("/api/scan")
async def trigger_scan(background_tasks: BackgroundTasks):
    global _is_scanning
    if _is_scanning:
        return {"status": "already_running", "message": "Scan already in progress"}
    _is_scanning = True
    background_tasks.add_task(_run_scan)
    return {"status": "started"}


async def _run_scan():
    global _last_snapshot, _is_scanning, _scan_count
    try:
        _last_snapshot = await get_bot().run_once()
        _scan_count   += 1
    except Exception as e:
        logger.error("[Dashboard] scan error: %s", e)
    finally:
        _is_scanning = False


@app.get("/api/status")
async def status():
    bot = get_bot()
    hours = bot.hours_guard
    market_open = hours.is_open() if hours else True
    return {
        "is_scanning":  _is_scanning,
        "scan_count":   _scan_count,
        "uptime_sec":   int(time.time() - _started_at),
        "market_open":  market_open,
        "market_reason": hours.reason if hours else "no guard",
        "breaker_ok":   not bot.breaker.is_tripped(bot._prices),
        "breaker_msg":  bot.breaker.reason,
        "timestamp":    time.time(),
    }


class WatchlistUpdate(BaseModel):
    tickers: List[str]

@app.get("/api/watchlist")
async def get_watchlist():
    return {"watchlist": get_bot().watchlist}

@app.post("/api/watchlist")
async def set_watchlist(body: WatchlistUpdate):
    get_bot().watchlist = body.tickers
    return {"watchlist": body.tickers}


@app.get("/api/trending")
async def get_trending():
    bot = get_bot()
    return {
        "trending": bot.trend_filter.top_trending(20),
        "counts":   dict(bot.trend_filter._counts),
    }


@app.get("/api/trades")
async def get_trades():
    bot = get_bot()
    return {
        "trades":   [t.to_dict() for t in bot.portfolio.trade_log],
        "win_rate": round(bot.portfolio.win_rate() * 100, 2),
        "realised_pnl": round(bot.portfolio.realised_pnl(), 2),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}