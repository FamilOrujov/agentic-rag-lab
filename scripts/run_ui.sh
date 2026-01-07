# Run the Streamlit UI for Agentic RAG

cd "$(dirname "$0")/.." || exit 1

echo "🚀 Starting Agentic RAG UI..."
echo "   Make sure the FastAPI backend is running on http://127.0.0.1:8000"
echo ""

source .venv/bin/activate
uv run streamlit run src/agentic_rag/ui/app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --theme.base dark \
    --browser.gatherUsageStats false

