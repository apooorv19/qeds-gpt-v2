#!/bin/sh
set -e

python src/worker.py &

exec uvicorn api.main:app \
  --app-dir src \
  --host 0.0.0.0 \
  --port "${PORT:-10000}"
