#!/usr/bin/env bash
# Start Qdrant + vLLM (docker compose), then the API and the Streamlit GUI. Ctrl+C stops the API and GUI.
set -euo pipefail
cd "$(dirname "$0")"

docker compose up -d
until curl -sf localhost:6333/readyz >/dev/null; do sleep 2; done
echo "Qdrant ready; waiting for vLLM (first start downloads the model)..."
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done

trap 'kill 0' EXIT
uv run rag serve &
until curl -sf localhost:8080/health >/dev/null; do sleep 2; done   # models load at startup
uv run streamlit run src/rag/ui/app.py &
wait
