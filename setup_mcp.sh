#!/usr/bin/env bash
# setup_mcp.sh — Install and start alpaca-mcp-server
# Source: https://github.com/tedlikeskix/alpaca-mcp-server
#
# Usage:
#   chmod +x setup_mcp.sh
#   ./setup_mcp.sh
#
# The MCP server must be running BEFORE you start the trading bot.
# It exposes Alpaca account tools (get_account, get_positions, get_orders,
# place_order, cancel_order) as HTTP endpoints at http://localhost:3000.

set -e

echo ""
echo "═══════════════════════════════════════════════"
echo "   Alpaca MCP Server Setup"
echo "═══════════════════════════════════════════════"

# Check Node.js
if ! command -v node &>/dev/null; then
    echo "❌ Node.js not found. Install from https://nodejs.org (v18+)"
    exit 1
fi
NODE_VER=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VER" -lt 18 ]; then
    echo "❌ Node.js v18+ required. Found: $(node -v)"
    exit 1
fi
echo "✅ Node.js $(node -v)"

# Clone or update
if [ -d "alpaca-mcp-server" ]; then
    echo "📦 Updating alpaca-mcp-server..."
    cd alpaca-mcp-server && git pull && cd ..
else
    echo "📦 Cloning alpaca-mcp-server..."
    git clone https://github.com/tedlikeskix/alpaca-mcp-server.git
fi

cd alpaca-mcp-server
npm install

# Load .env from parent directory if it exists
if [ -f "../.env" ]; then
    export $(grep -v '^#' ../.env | xargs) 2>/dev/null || true
fi

# Check credentials
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_SECRET_KEY" ]; then
    echo ""
    echo "⚠  ALPACA_API_KEY or ALPACA_SECRET_KEY not set."
    echo "   Add them to your .env file and re-run this script."
    echo "   Paper keys:  https://app.alpaca.markets/paper/dashboard/overview"
    exit 1
fi

echo ""
echo "✅ Alpaca credentials found"
echo "✅ Starting alpaca-mcp-server on http://localhost:3000"
echo ""
echo "   Leave this running in a separate terminal."
echo "   Then start the trading bot in another terminal."
echo ""

# Start the server (stays running in foreground)
ALPACA_API_KEY="$ALPACA_API_KEY" \
ALPACA_SECRET_KEY="$ALPACA_SECRET_KEY" \
ALPACA_BASE_URL="${ALPACA_BASE_URL:-https://paper-api.alpaca.markets}" \
node index.js