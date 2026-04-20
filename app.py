"""
paper_trader_flask/app.py
─────────────────────────────────────────────────────────────────────────────
Flask wrapper for the Production Trading Bot.

Routes:
  GET  /                      ← Dashboard UI
  GET  /api/status            ← Bot status + portfolio snapshot
  POST /api/scan              ← Trigger one scan cycle
  POST /api/start             ← Start continuous loop (background thread)
  POST /api/stop              ← Stop the loop
  GET  /api/signals           ← Last 50 signals
  GET  /api/portfolio         ← Portfolio positions + PnL
  POST /api/watchlist         ← Update watchlist  { "tickers": ["$NVDA", ...] }
  GET  /api/trending          ← Current trending ticker counts
  POST /api/config            ← Update bot config at runtime

Run:
  python app.py               ← dev server (debug mode)
  gunicorn "app:create_app()" ← production
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FlaskApp")


# ─────────────────────────────────────────────────────────────────────────────
# Bot state container  (shared across threads)
# ─────────────────────────────────────────────────────────────────────────────

class BotState:
    """Thread-safe wrapper around TradingBot + loop management."""

    def __init__(self):
        self._lock          = threading.Lock()
        self._bot           = None          # TradingBot instance
        self._loop_thread: Optional[threading.Thread] = None
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
        self._running       = False
        self._last_snapshot = {}
        self._last_scan_ts  = None
        self._config        = {
            "watchlist":            list(DEFAULT_WATCHLIST),
            "interval_minutes":     15,
            "sentiment_threshold":  0.30,
            "require_market_hours": True,
            "daily_drawdown_limit": 0.03,
            "total_drawdown_limit": 0.10,
            "signal_cooldown_min":  60,
            "trend_min_mentions":   2,
            "replicate_model":      "meta/meta-llama-3-70b-instruct",
        }

    # ── Bot lifecycle ─────────────────────────────────────────────────────

    def _build_bot(self):
        """(Re)create TradingBot from current config. Must hold _lock."""
        from bot_core import TradingBot   # local import — avoids circular refs
        cfg = self._config
        self._bot = TradingBot(
            watchlist=cfg["watchlist"],
            interval_minutes=cfg["interval_minutes"],
            sentiment_threshold=cfg["sentiment_threshold"],
            require_market_hours=cfg["require_market_hours"],
            daily_drawdown_limit=cfg["daily_drawdown_limit"],
            total_drawdown_limit=cfg["total_drawdown_limit"],
            signal_cooldown_min=cfg["signal_cooldown_min"],
            trend_min_mentions=cfg["trend_min_mentions"],
            replicate_model=cfg["replicate_model"],
        )

    def get_or_build_bot(self):
        with self._lock:
            if self._bot is None:
                self._build_bot()
            return self._bot

    # ── Single scan ───────────────────────────────────────────────────────

    def run_scan_sync(self, tickers=None) -> dict:
        """Run one scan in a fresh event loop (called from Flask route thread)."""
        bot = self.get_or_build_bot()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            snap = loop.run_until_complete(bot.run_once(tickers))
        finally:
            loop.close()

        with self._lock:
            self._last_snapshot = snap
            self._last_scan_ts  = time.strftime("%Y-%m-%d %H:%M:%S")
        return snap

    # ── Continuous loop ───────────────────────────────────────────────────

    def _loop_worker(self):
        """Background thread: runs asyncio loop continuously."""
        self._asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._asyncio_loop)

        async def _inner():
            bot = self.get_or_build_bot()
            interval = self._config["interval_minutes"]
            logger.info("Background loop started — interval=%d min", interval)
            while self._running:
                try:
                    snap = await bot.run_once()
                    bot.portfolio.save("portfolio.json")
                    with self._lock:
                        self._last_snapshot = snap
                        self._last_scan_ts  = time.strftime("%Y-%m-%d %H:%M:%S")

                    # Smart sleep: skip to market open if blocked
                    blocked = snap.get("blocked", "")
                    if blocked and "Market" in blocked and bot.hours_guard:
                        wait  = bot.hours_guard.seconds_until_open()
                        sleep = min(wait, interval * 60)
                    else:
                        sleep = interval * 60

                    # Sleep in 1-second chunks so we can stop cleanly
                    for _ in range(int(sleep)):
                        if not self._running:
                            break
                        await asyncio.sleep(1)
                except Exception as exc:
                    logger.error("Loop error: %s", exc, exc_info=True)
                    await asyncio.sleep(60)   # back-off on error

        try:
            self._asyncio_loop.run_until_complete(_inner())
        finally:
            self._asyncio_loop.close()
            logger.info("Background loop exited.")

    def start_loop(self) -> bool:
        with self._lock:
            if self._running:
                return False   # already running
            self._running = True

        self._loop_thread = threading.Thread(
            target=self._loop_worker, daemon=True, name="BotLoop"
        )
        self._loop_thread.start()
        logger.info("Bot loop thread started.")
        return True

    def stop_loop(self) -> bool:
        with self._lock:
            if not self._running:
                return False
            self._running = False
        logger.info("Bot loop stop requested.")
        return True

    # ── Config ────────────────────────────────────────────────────────────

    def update_config(self, patch: dict):
        with self._lock:
            self._config.update(patch)
            self._bot = None   # force rebuild on next scan

    def get_config(self) -> dict:
        with self._lock:
            return dict(self._config)

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        with self._lock:
            bot = self._bot
            snap = dict(self._last_snapshot)

        portfolio_info = {}
        if bot:
            try:
                portfolio_info = bot.portfolio.snapshot(bot._prices)
            except Exception:
                pass

        return {
            "running":       self._running,
            "last_scan":     self._last_scan_ts,
            "config":        self.get_config(),
            "portfolio":     portfolio_info,
            "signals":       snap.get("signals", []),
            "trending":      snap.get("trending", {}),
            "skipped":       snap.get("skipped", []),
            "blocked":       snap.get("blocked", ""),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WATCHLIST = [
    "$NVDA","$TSLA","$AAPL","$MSFT","$AMZN",
    "$META","$GOOGL","$AMD","$PLTR","$COIN",
]

bot_state = BotState()   # module-level singleton


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_SORT_KEYS"] = False

    # ── Dashboard ─────────────────────────────────────────────────────────

    @app.get("/")
    def dashboard():
        return render_template("dashboard.html")

    # ── Status ────────────────────────────────────────────────────────────

    @app.get("/api/status")
    def api_status():
        return jsonify(bot_state.status())

    # ── Single scan ───────────────────────────────────────────────────────

    @app.post("/api/scan")
    def api_scan():
        """
        Optional JSON body: { "tickers": ["$NVDA", "$TSLA"] }
        If omitted, uses configured watchlist.
        """
        if bot_state._running:
            return jsonify({"error": "Loop is running — stop it first or wait for next tick"}), 409

        tickers = None
        if request.is_json:
            tickers = request.json.get("tickers")

        try:
            snap = bot_state.run_scan_sync(tickers)
            return jsonify({"ok": True, "snapshot": snap})
        except Exception as exc:
            logger.error("Scan error: %s", exc, exc_info=True)
            return jsonify({"error": str(exc)}), 500

    # ── Start / Stop loop ─────────────────────────────────────────────────

    @app.post("/api/start")
    def api_start():
        started = bot_state.start_loop()
        if not started:
            return jsonify({"ok": False, "message": "Already running"}), 200
        return jsonify({"ok": True, "message": "Bot loop started"})

    @app.post("/api/stop")
    def api_stop():
        stopped = bot_state.stop_loop()
        if not stopped:
            return jsonify({"ok": False, "message": "Not running"}), 200
        return jsonify({"ok": True, "message": "Bot loop stopping…"})

    # ── Signals ───────────────────────────────────────────────────────────

    @app.get("/api/signals")
    def api_signals():
        status = bot_state.status()
        return jsonify({"signals": status.get("signals", [])})

    # ── Portfolio ─────────────────────────────────────────────────────────

    @app.get("/api/portfolio")
    def api_portfolio():
        status = bot_state.status()
        return jsonify(status.get("portfolio", {}))

    # ── Watchlist ─────────────────────────────────────────────────────────

    @app.post("/api/watchlist")
    def api_watchlist():
        data = request.get_json(silent=True) or {}
        tickers = data.get("tickers")
        if not isinstance(tickers, list) or not tickers:
            return jsonify({"error": "Provide { 'tickers': ['$NVDA', ...] }"}), 400
        bot_state.update_config({"watchlist": tickers})
        return jsonify({"ok": True, "watchlist": tickers})

    # ── Trending ──────────────────────────────────────────────────────────

    @app.get("/api/trending")
    def api_trending():
        status = bot_state.status()
        return jsonify({"trending": status.get("trending", {})})

    # ── Config ────────────────────────────────────────────────────────────

    @app.post("/api/config")
    def api_config():
        """
        Patch any subset of config keys at runtime.
        Bot is rebuilt on next scan automatically.

        Example body:
          { "interval_minutes": 5, "sentiment_threshold": 0.25 }
        """
        data = request.get_json(silent=True) or {}
        ALLOWED = {
            "interval_minutes", "sentiment_threshold",
            "require_market_hours", "daily_drawdown_limit",
            "total_drawdown_limit", "signal_cooldown_min",
            "trend_min_mentions", "replicate_model",
        }
        patch = {k: v for k, v in data.items() if k in ALLOWED}
        if not patch:
            return jsonify({"error": f"No valid keys. Allowed: {sorted(ALLOWED)}"}), 400
        bot_state.update_config(patch)
        return jsonify({"ok": True, "updated": patch, "config": bot_state.get_config()})

    # ── Health ────────────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "ts": time.time()})

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Dev server entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)