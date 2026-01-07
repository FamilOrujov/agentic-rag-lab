set -euo pipefail

# Separate directories for Keploy so I don't destroy my real dev data
export CHROMA_DIR="${CHROMA_DIR:-./.chroma_keploy}"
export UPLOAD_DIR="${UPLOAD_DIR:-./data/uploads_keploy}"

rm = -rf "$CHROMA_DIR" "$UPLOAD_DIR"
mkdir -p "$CHROMA_DIR" "$UPLOAD_DIR"

exec uv run uvicorn agentic_rag.api.main:app --app-dir src --host 127.0.0.1 --port 8000
