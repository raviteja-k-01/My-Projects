#!/bin/bash
# run.sh — Start both backend and frontend

echo "🚀 Starting Multimodal AI Assistant..."

# Check for .env
if [ ! -f .env ]; then
  echo "⚠️  .env file not found. Copying from .env.example..."
  cp .env.example .env
  echo "👉  Please add your OPENAI_API_KEY to .env and re-run."
  exit 1
fi

# Start FastAPI backend
echo "📡 Starting FastAPI backend on http://localhost:8000 ..."
uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

sleep 2

# Start Streamlit frontend
echo "🖥️  Starting Streamlit frontend on http://localhost:8501 ..."
streamlit run frontend/app.py --server.port 8501

# Cleanup on exit
kill $BACKEND_PID
