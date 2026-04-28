"""
paper_trader_flask/app.py  — v2
─────────────────────────────────────────────────────────────────────────────
Flask wrapper for the Production Trading Bot.

FIXES vs v1:
  1. CONFIG FIX: sentiment_threshold 0.30 → 0.15, confidence_threshold added
     0.35, trend_min_mentions 2 → 0 — all aligned with bot.py v3 defaults
     so app.py doesn't silently override the fixes in bot.py with stale values.
  2. require_market_hours default changed to None so bot.py auto-derives from
     PRODUCTION env var — was hardcoded True, blocking all paper scans.
  3. Added /api/config GET endpoint so you can inspect live config.
  4. Added use_alpaca_mcp config key so MCP can be toggled via API.
  5. Better error messages: scan failures now return the actual exception.
  6. Added /api/mcp_status endpoint to check if MCP server is reachable.

Routes:
  GET  /                    Dashboard UI
  GET  /api/status          Bot status + portfolio snapshot
  POST /api/scan            One scan cycle  { "tickers": [...] }
  POST /api/start           Start continuous loop
  POST /api/stop            Stop the loop
  GET  /api/signals         Last 50 signals
  GET  /api/portfolio       Portfolio positions + PnL
  POST /api/watchlist       { "tickers": ["$NVDA", ...] }
  GET  /api/trending        Trending ticker mention counts
  GET  /api/config          Current config
  POST /api/config          Patch config keys at runtime
  GET  /api/mcp_status      Check alpaca-mcp-server reachability
  GET  /health              Healthcheck

Run:
  python app.py                           dev
  gunicorn "app:create_app()" --workers 1 production (single worker — bot is stateful)
"""
from __future__ import annotations

import asyncio, logging, os, threading, time
from typing import Optional

import httpx
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FlaskApp")

ALPACA_MCP_URL = os.getenv("ALPACA_MCP_URL", "http://localhost:3000")

DEFAULT_WATCHLIST = [
    "$NVDA","$TSLA","$AAPL","$MSFT","$AMZN",
    "$META","$GOOGL","$AMD","$PLTR","$COIN",
]


# ─────────────────────────────────────────────────────────────────────────────
# Bot state container  (shared across threads)
# ─────────────────────────────────────────────────────────────────────────────

class BotState:

    def __init__(self):
        self._lock          = threading.Lock()
        self._bot           = None
        self._loop_thread: Optional[threading.Thread] = None
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
        self._running       = False
        self._last_snapshot = {}
        self._last_scan_ts  = None
        self._config        = {
            "watchlist":            list(DEFAULT_WATCHLIST),
            "interval_minutes":     15,
            # FIX: aligned with bot.py v3 defaults — these were the source of no-signals
            "sentiment_threshold":  0.15,    # was 0.30
            "confidence_threshold": 0.35,    # NEW — was missing, bot used 0.45 hardcoded
            "require_market_hours": None,    # None = auto (False=paper, True=prod)
            "daily_drawdown_limit": 0.03,
            "total_drawdown_limit": 0.10,
            "signal_cooldown_min":  30,
            "trend_min_mentions":   0,       # was 2 — was silently skipping all tickers
            "replicate_model":      "meta/meta-llama-3-70b-instruct",
            "use_alpaca_mcp":       True,
        }

    def _build_bot(self):
        from bot_core import TradingBot
        cfg = self._config
        self._bot = TradingBot(
            watchlist=cfg["watchlist"],
            interval_minutes=cfg["interval_minutes"],
            sentiment_threshold=cfg["sentiment_threshold"],
            confidence_threshold=cfg.get("confidence_threshold", 0.35),
            require_market_hours=cfg.get("require_market_hours"),
            daily_drawdown_limit=cfg["daily_drawdown_limit"],
            total_drawdown_limit=cfg["total_drawdown_limit"],
            signal_cooldown_min=cfg["signal_cooldown_min"],
            trend_min_mentions=cfg["trend_min_mentions"],
            replicate_model=cfg["replicate_model"],
            use_alpaca_mcp=cfg.get("use_alpaca_mcp", True),
        )

    def get_or_build_bot(self):
        with self._lock:
            if self._bot is None:
                self._build_bot()
            return self._bot

    # ── Single scan ───────────────────────────────────────────────────────

    def run_scan_sync(self, tickers=None) -> dict:
        bot  = self.get_or_build_bot()
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
        self._asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._asyncio_loop)

        async def _inner():
            bot      = self.get_or_build_bot()
            interval = self._config["interval_minutes"]
            logger.info("Background loop started — interval=%d min", interval)
            while self._running:
                try:
                    snap = await bot.run_once()
                    if hasattr(bot.portfolio, 'save'):
                        bot.portfolio.save(bot._portfolio_path)
                    with self._lock:
                        self._last_snapshot = snap
                        self._last_scan_ts  = time.strftime("%Y-%m-%d %H:%M:%S")

                    blocked = snap.get("blocked", "")
                    if blocked and "Market" in blocked and bot.hours_guard:
                        wait  = bot.hours_guard.seconds_until_open()
                        sleep = min(wait, interval * 60)
                    else:
                        sleep = interval * 60

                    for _ in range(int(sleep)):
                        if not self._running: break
                        await asyncio.sleep(1)
                except Exception as exc:
                    logger.error("Loop error: %s", exc, exc_info=True)
                    await asyncio.sleep(60)

        try:
            self._asyncio_loop.run_until_complete(_inner())
        finally:
            self._asyncio_loop.close()
            logger.info("Background loop exited.")

    def start_loop(self) -> bool:
        with self._lock:
            if self._running: return False
            self._running = True
        self._loop_thread = threading.Thread(
            target=self._loop_worker, daemon=True, name="BotLoop")
        self._loop_thread.start()
        return True

    def stop_loop(self) -> bool:
        with self._lock:
            if not self._running: return False
            self._running = False
        return True

    def update_config(self, patch: dict):
        with self._lock:
            self._config.update(patch)
            self._bot = None   # force rebuild on next scan

    def get_config(self) -> dict:
        with self._lock:
            return dict(self._config)

    def status(self) -> dict:
        with self._lock:
            bot  = self._bot
            snap = dict(self._last_snapshot)
        portfolio_info = {}
        if bot:
            try:
                portfolio_info = bot.portfolio.snapshot(bot._prices)
            except Exception:
                pass
        return {
            "running":    self._running,
            "last_scan":  self._last_scan_ts,
            "config":     self.get_config(),
            "portfolio":  portfolio_info,
            "signals":    snap.get("signals", []),
            "trending":   snap.get("trending", {}),
            "skipped":    snap.get("skipped", []),
            "blocked":    snap.get("blocked", ""),
        }


bot_state = BotState()


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_SORT_KEYS"] = False

    @app.get("/")
    def dashboard():
        return render_template("dashboard.html")

    @app.get("/api/status")
    def api_status():
        return jsonify(bot_state.status())

    @app.post("/api/scan")
    def api_scan():
        if bot_state._running:
            return jsonify({"error": "Loop running — stop first or wait for tick"}), 409
        tickers = None
        if request.is_json:
            tickers = request.json.get("tickers")
        try:
            snap = bot_state.run_scan_sync(tickers)
            return jsonify({"ok": True, "snapshot": snap})
        except Exception as exc:
            logger.error("Scan error: %s", exc, exc_info=True)
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/start")
    def api_start():
        started = bot_state.start_loop()
        return jsonify({"ok": started,
                        "message": "Bot loop started" if started else "Already running"})

    @app.post("/api/stop")
    def api_stop():
        stopped = bot_state.stop_loop()
        return jsonify({"ok": stopped,
                        "message": "Stopping…" if stopped else "Not running"})

    @app.get("/api/signals")
    def api_signals():
        return jsonify({"signals": bot_state.status().get("signals", [])})

    @app.get("/api/portfolio")
    def api_portfolio():
        return jsonify(bot_state.status().get("portfolio", {}))

    @app.post("/api/watchlist")
    def api_watchlist():
        data    = request.get_json(silent=True) or {}
        tickers = data.get("tickers")
        if not isinstance(tickers, list) or not tickers:
            return jsonify({"error": "Provide { 'tickers': ['$NVDA', ...] }"}), 400
        bot_state.update_config({"watchlist": tickers})
        return jsonify({"ok": True, "watchlist": tickers})

    @app.get("/api/trending")
    def api_trending():
        return jsonify({"trending": bot_state.status().get("trending", {})})

    @app.get("/api/config")
    def api_config_get():
        return jsonify(bot_state.get_config())

    @app.post("/api/config")
    def api_config_post():
        data = request.get_json(silent=True) or {}
        ALLOWED = {
            "interval_minutes", "sentiment_threshold", "confidence_threshold",
            "require_market_hours", "daily_drawdown_limit", "total_drawdown_limit",
            "signal_cooldown_min", "trend_min_mentions", "replicate_model",
            "use_alpaca_mcp",
        }
        patch = {k: v for k, v in data.items() if k in ALLOWED}
        if not patch:
            return jsonify({"error": f"No valid keys. Allowed: {sorted(ALLOWED)}"}), 400
        bot_state.update_config(patch)
        return jsonify({"ok": True, "updated": patch, "config": bot_state.get_config()})

    @app.get("/api/alpaca_status")
    def api_alpaca_status():
        """
        Check Alpaca API connectivity and return account details.
        Tries to connect using current env vars and returns:
          - connected: bool
          - mode: "paper" | "live" | "internal"
          - account: { cash, equity, buying_power, status } or {}
          - error: str (only if not connected)
        """
        key    = os.getenv("ALPACA_API_KEY", "")
        secret = os.getenv("ALPACA_SECRET_KEY", "")
        base   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        use_alpaca = os.getenv("USE_ALPACA_PAPER", "false").lower() == "true" \
                     or os.getenv("PRODUCTION", "false").lower() == "true"

        if not key or not secret:
            return jsonify({
                "connected": False,
                "mode": "internal",
                "account": {},
                "error": "ALPACA_API_KEY or ALPACA_SECRET_KEY not set in environment"
            })

        if not use_alpaca:
            return jsonify({
                "connected": False,
                "mode": "internal",
                "account": {},
                "error": "USE_ALPACA_PAPER=false — bot is using internal paper portfolio, not Alpaca"
            })

        is_paper = "paper-api" in base
        mode = "paper" if is_paper else "live"

        try:
            from alpaca.trading.client import TradingClient
            client = TradingClient(key, secret, paper=is_paper)
            acct   = client.get_account()
            return jsonify({
                "connected":  True,
                "mode":       mode,
                "url":        base,
                "account": {
                    "cash":          str(acct.cash),
                    "equity":        str(acct.equity),
                    "buying_power":  str(acct.buying_power),
                    "status":        acct.status.value if hasattr(acct.status, 'value') else str(acct.status),
                    "currency":      str(acct.currency),
                    "pattern_day_trader": bool(acct.pattern_day_trader),
                },
            })
        except Exception as e:
            return jsonify({
                "connected": False,
                "mode":      mode,
                "account":   {},
                "error":     str(e),
            })

    @app.get("/api/mcp_status")
    def api_mcp_status():
        """Check whether the alpaca-mcp-server is reachable."""
        try:
            import requests as req
            r = req.get(f"{ALPACA_MCP_URL}/health", timeout=4)
            return jsonify({"reachable": r.status_code == 200,
                            "url": ALPACA_MCP_URL,
                            "status_code": r.status_code})
        except Exception as e:
            return jsonify({"reachable": False, "url": ALPACA_MCP_URL, "error": str(e)})

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "ts": time.time()})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)