#!/bin/bash
# Docker Entrypoint Script
# =========================
# This script starts the API and/or Frontend based on the command

set -e

echo "============================================"
echo "🐳 Twitter Sentiment Analyzer - Docker"
echo "============================================"

# Function to start API
start_api() {
    echo "🚀 Starting API server..."
    uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    echo "✅ API started (PID: $API_PID)"
}

# Function to start Frontend
start_frontend() {
    echo "🎨 Starting Frontend..."
    streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
    FRONTEND_PID=$!
    echo "✅ Frontend started (PID: $FRONTEND_PID)"
}

# Handle different commands
case "$1" in
    api)
        echo "📡 Running API only..."
        start_api
        wait $API_PID
        ;;
    frontend)
        echo "🎨 Running Frontend only..."
        start_frontend
        wait $FRONTEND_PID
        ;;
    both)
        echo "🚀 Running both API and Frontend..."
        start_api
        sleep 5  # Wait for API to be ready
        start_frontend
        
        echo ""
        echo "============================================"
        echo "✅ Application is running!"
        echo "============================================"
        echo "📍 API:      http://localhost:8000"
        echo "📍 Frontend: http://localhost:8501"
        echo "📚 API Docs: http://localhost:8000/"
        echo "============================================"
        echo ""
        
        # Wait for both processes
        wait $API_PID $FRONTEND_PID
        ;;
    *)
        echo "Usage: docker-entrypoint.sh {api|frontend|both}"
        echo "Running both by default..."
        exec "$0" both
        ;;
esac
