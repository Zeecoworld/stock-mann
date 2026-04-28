"""
run_bot.py  — Unified Bot Launcher
──────────────────────────────────────────────────────────────────────────────
Single entry point that starts the trading bot + dashboard together.

WHY YOUR ALPACA ORDERS WEREN'T SHOWING:
  Your bot was generating real signals (you saw them in the dashboard) but
  executing them against an internal PaperPortfolio — a local Python object
  that never calls Alpaca's API. No orders were sent to Alpaca at all.

  The fix is simple: set USE_ALPACA_PAPER=true in your .env. This tells the
  bot to use AlpacaBroker instead of PaperPortfolio, so every BUY/SELL signal
  becomes a real order on Alpaca's paper trading account.

  PAPER mode  (USE_ALPACA_PAPER=false, default):
    Orders stay local — nothing reaches Alpaca. Good for testing the logic.

  ALPACA PAPER (USE_ALPACA_PAPER=true):
    Orders go to paper-api.alpaca.markets — you'll see them in the dashboard.
    No real money. This is what you want right now.

  LIVE mode   (PRODUCTION=true):
    Orders go to api.alpaca.markets with REAL MONEY. Only flip this after
    60+ days of successful Alpaca paper trading.

USAGE:
  python run_bot.py               # starts bot + Flask dashboard on port 5000
  python run_bot.py --mode paper  # internal paper only (no Alpaca calls)
  python run_bot.py --mode alpaca # Alpaca paper trading (orders show in dashboard)
  python run_bot.py --scan-once   # run one scan then exit (good for testing)
  python run_bot.py --port 8080   # use a different port for the dashboard

REQUIREMENTS:
  pip install flask httpx python-dotenv replicate alpaca-py
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Launcher")


# ── Env validation ─────────────────────────────────────────────────────────────

def validate_env(mode: str) -> bool:
    """Check all required env vars are present before starting."""
    errors   = []
    warnings = []

    # Always required
    if not os.getenv("REPLICATE_API_TOKEN"):
        errors.append("REPLICATE_API_TOKEN is missing — the sentiment LLM won't work.\n"
                      "  Get one free at: https://replicate.com/account/api-tokens")

    # Required for Alpaca paper/live
    if mode in ("alpaca", "live"):
        if not os.getenv("ALPACA_API_KEY"):
            errors.append("ALPACA_API_KEY is missing.\n"
                          "  Get paper keys at: https://app.alpaca.markets/paper/dashboard/overview")
        if not os.getenv("ALPACA_SECRET_KEY"):
            errors.append("ALPACA_SECRET_KEY is missing.")

    # Warnings (non-fatal)
    if not os.getenv("NEWSAPI_KEY"):
        warnings.append("NEWSAPI_KEY not set — news will use free RSS feeds only (still works)")
    if not os.getenv("FINNHUB_TOKEN"):
        warnings.append("FINNHUB_TOKEN not set — Finnhub headlines disabled (still works)")

    if warnings:
        print("\n⚠  WARNINGS (non-fatal):")
        for w in warnings:
            print(f"   • {w}")

    if errors:
        print("\n❌  ERRORS — cannot start:")
        for e in errors:
            print(f"   • {e}")
        print("\n  Create a .env file in this folder with the missing values.")
        print("  See .env.example for the full template.\n")
        return False

    return True


# ── Mode setup ─────────────────────────────────────────────────────────────────

def configure_mode(mode: str):
    """
    Set the right env vars for the chosen mode so bot.py picks them up.

    INTERNAL PAPER:  bot uses PaperPortfolio (local only, no Alpaca calls)
    ALPACA PAPER:    bot uses AlpacaBroker → paper-api.alpaca.markets
    LIVE:            bot uses AlpacaBroker → api.alpaca.markets (REAL MONEY)
    """
    if mode == "paper":
        os.environ["PRODUCTION"]        = "false"
        os.environ["USE_ALPACA_PAPER"]  = "false"
        os.environ["ALPACA_BASE_URL"]   = "https://paper-api.alpaca.markets"
        print("\n📄  MODE: Internal paper trading (no Alpaca orders sent)")

    elif mode == "alpaca":
        # This is the key setting — uses AlpacaBroker with paper endpoint
        os.environ["PRODUCTION"]        = "true"   # tells bot.py to use AlpacaBroker
        os.environ["USE_ALPACA_PAPER"]  = "true"
        os.environ["ALPACA_BASE_URL"]   = "https://paper-api.alpaca.markets"
        print("\n📊  MODE: Alpaca PAPER trading — orders will appear in your Alpaca dashboard")
        print("          URL: https://paper-api.alpaca.markets")

    elif mode == "live":
        os.environ["PRODUCTION"]        = "true"
        os.environ["USE_ALPACA_PAPER"]  = "false"
        os.environ["ALPACA_BASE_URL"]   = "https://api.alpaca.markets"
        print("\n🔴  MODE: LIVE trading — REAL MONEY — double-check your keys!")

    print(f"    ALPACA_BASE_URL = {os.environ['ALPACA_BASE_URL']}")
    print(f"    PRODUCTION      = {os.environ['PRODUCTION']}\n")


# ── Bot runner ─────────────────────────────────────────────────────────────────

def run_bot_loop(scan_once: bool = False):
    """Run the trading bot in a background thread."""
    try:
        from bot import TradingBot, DEFAULT_WATCHLIST
    except ImportError as e:
        logger.error("Cannot import bot.py: %s", e)
        return

    async def _inner():
        bot = TradingBot()
        logger.info("[Bot] Watchlist: %s", ", ".join(bot.watchlist))

        if scan_once:
            logger.info("[Bot] Running single scan…")
            snap = await bot.run_once()
            logger.info("[Bot] Scan complete — signals: %d", len(snap.get("signals", [])))
            return

        logger.info("[Bot] Starting continuous loop (interval=%ds)", bot.interval)
        await bot.run_loop()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_inner())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


# ── Flask dashboard runner ─────────────────────────────────────────────────────

def run_flask_dashboard(port: int = 5000):
    """Start the Flask dashboard on the given port."""
    try:
        from app import create_app
        app = create_app()
        logger.info("[Dashboard] Starting on http://localhost:%d", port)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    except ImportError as e:
        logger.warning("[Dashboard] app.py not found (%s) — running bot only", e)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trading Bot Launcher")
    parser.add_argument(
        "--mode",
        choices=["paper", "alpaca", "live"],
        default=None,
        help=(
            "paper  = internal paper only, no Alpaca orders (default if USE_ALPACA_PAPER not set)\n"
            "alpaca = send orders to Alpaca paper account (you'll see them in dashboard)\n"
            "live   = send orders to Alpaca live account (REAL MONEY)"
        ),
    )
    parser.add_argument("--scan-once",  action="store_true",
                        help="Run one scan cycle then exit")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Run bot only, skip the Flask dashboard")
    parser.add_argument("--port", type=int, default=5000,
                        help="Dashboard port (default: 5000)")
    args = parser.parse_args()

    # Determine mode
    if args.mode:
        mode = args.mode
    elif os.getenv("USE_ALPACA_PAPER", "false").lower() == "true":
        mode = "alpaca"
    elif os.getenv("PRODUCTION", "false").lower() == "true":
        mode = "live"
    else:
        mode = "paper"

    print("─" * 60)
    print("  TRADING BOT LAUNCHER")
    print("─" * 60)

    # Validate env before doing anything
    if not validate_env(mode):
        sys.exit(1)

    # Apply mode settings
    configure_mode(mode)

    # Start dashboard in background thread (unless --no-dashboard)
    if not args.no_dashboard and not args.scan_once:
        dash_thread = threading.Thread(
            target=run_flask_dashboard,
            kwargs={"port": args.port},
            daemon=True,
            name="Dashboard",
        )
        dash_thread.start()
        time.sleep(1)   # give Flask a moment to bind
        print(f"\n  Dashboard → http://localhost:{args.port}")
        print(f"  API       → http://localhost:{args.port}/api/status\n")

    # Run the bot (blocks until Ctrl+C or scan_once completes)
    try:
        run_bot_loop(scan_once=args.scan_once)
    except KeyboardInterrupt:
        print("\n\n  Stopped by user.\n")


if __name__ == "__main__":
    main()