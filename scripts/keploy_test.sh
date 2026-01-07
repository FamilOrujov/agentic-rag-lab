set -euo pipefail

APP_CMD='uv run uvicorn agentic_rag.api.main:app --app-dir src --host 127.0.0.1 --port 8000'

sudo -E env "PATH=$PATH" keploy test \
  -c "$APP_CMD" \
  -p ./keploy \
  --delay 2 \
  --pass-through-ports 11434 \
  --debug
