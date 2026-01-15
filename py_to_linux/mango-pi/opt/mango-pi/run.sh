#!/bin/bash

set -e

APP_DIR="/opt/mango-pi"
PYTHON_BIN="/usr/bin/python3"

echo "🍃 Mango-Pi starting..."
echo "📁 App directory: $APP_DIR"

# Safety checks
if [ ! -f "$APP_DIR/server.py" ]; then
    echo "❌ server.py not found"
    exit 1
fi

if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ Python3 not found"
    exit 1
fi

# Do NOT touch system Python
# Do NOT install packages here
# Do NOT access camera here

cd "$APP_DIR"

echo "🚀 Launching Flask server..."
exec "$PYTHON_BIN" server.py