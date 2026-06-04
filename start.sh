#!/bin/bash
set -e

echo "Starting JSAV..."

# Backend
cd backend
if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "⚠️  Created backend/.env — add your API keys before using."
fi

uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

# Frontend
cd ../frontend
npm install --silent
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✓ Backend:  http://localhost:8000"
echo "✓ Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers."

cleanup() {
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  exit 0
}
trap cleanup INT TERM
wait
