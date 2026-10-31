#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
if .venv/bin/python - <<'PY'
import json,urllib.request,sys
try:
    with urllib.request.urlopen('http://127.0.0.1:8765/api/v1/health',timeout=1) as response:
        data=json.load(response)
    sys.exit(0 if data.get('deployment')=='local-single-user' else 1)
except Exception:
    sys.exit(1)
PY
then
  printf '\n趋势研究所已经在运行： http://127.0.0.1:8765\n\n'
  exit 0
fi
if ! .venv/bin/python -c 'import fastapi,torch,akshare,deap' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements.txt
fi
if [ ! -d frontend/node_modules ]; then
  npm --prefix frontend ci
fi
npm --prefix frontend run build
printf '\n趋势研究所已启动，打开 http://127.0.0.1:8765\n\n'
exec .venv/bin/python -m uvicorn backend.api:app --host 127.0.0.1 --port 8765
