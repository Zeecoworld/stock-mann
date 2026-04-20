#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — one-command deploy to Ubuntu VPS (20.04 / 22.04 / 24.04)
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# What it does:
#   1. Installs Python 3.11 + pip if not present
#   2. Creates a dedicated 'botuser' system account
#   3. Sets up a virtualenv and installs all deps
#   4. Copies .env if it exists
#   5. Installs and starts the systemd service
#   6. Prints live logs
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

APP_DIR="/home/botuser/paper_trader"
SERVICE="trading-bot"
PYTHON="python3.11"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          Trading Bot — Production Deploy Script          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Check running as root ──────────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
  echo "❌  Please run as root: sudo ./deploy.sh"; exit 1
fi

# ── 2. Install system dependencies ───────────────────────────────────────────
echo "📦  Installing system packages…"
apt-get update -qq
apt-get install -y --no-install-recommends \
  python3.11 python3.11-venv python3-pip \
  tzdata curl git 2>/dev/null || true

# ── 3. Create botuser ─────────────────────────────────────────────────────────
if ! id botuser &>/dev/null; then
  echo "👤  Creating botuser…"
  useradd -m -s /bin/bash botuser
fi

# ── 4. Copy project files ─────────────────────────────────────────────────────
echo "📁  Copying project to $APP_DIR…"
mkdir -p "$APP_DIR"
rsync -a --exclude=".git" --exclude="__pycache__" --exclude="*.pyc" \
  "$(pwd)/" "$APP_DIR/"
chown -R botuser:botuser "$APP_DIR"

# ── 5. Copy .env if it exists ─────────────────────────────────────────────────
if [ -f ".env" ]; then
  cp .env "$APP_DIR/.env"
  chown botuser:botuser "$APP_DIR/.env"
  chmod 600 "$APP_DIR/.env"
  echo "🔐  .env copied and secured"
else
  echo "⚠️   No .env found — copy .env.example to .env and fill in your keys"
  echo "    cp $APP_DIR/.env.example $APP_DIR/.env"
  echo "    nano $APP_DIR/.env"
fi

# ── 6. Create virtualenv and install dependencies ────────────────────────────
echo "🐍  Setting up Python virtualenv…"
sudo -u botuser $PYTHON -m venv "$APP_DIR/venv"
sudo -u botuser "$APP_DIR/venv/bin/pip" install --quiet --upgrade pip
sudo -u botuser "$APP_DIR/venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"
echo "✅  Dependencies installed"

# ── 7. Create data directory ─────────────────────────────────────────────────
mkdir -p "$APP_DIR/data"
chown botuser:botuser "$APP_DIR/data"

# ── 8. Install systemd service ────────────────────────────────────────────────
echo "⚙️   Installing systemd service…"

# Patch the service file to use the venv python
sed "s|/home/botuser/paper_trader/venv/bin/python|$APP_DIR/venv/bin/python|g" \
  "$APP_DIR/trading-bot.service" > /etc/systemd/system/$SERVICE.service

systemctl daemon-reload
systemctl enable $SERVICE
systemctl restart $SERVICE

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅  Trading Bot deployed and running!                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Useful commands:"
echo "  systemctl status  $SERVICE     # check status"
echo "  journalctl -fu    $SERVICE     # live logs"
echo "  systemctl stop    $SERVICE     # stop bot"
echo "  systemctl restart $SERVICE     # restart bot"
echo ""
echo "Live logs (Ctrl+C to exit):"
echo "──────────────────────────────────────────────────────────"
sleep 2
journalctl -fu $SERVICE --no-pager
