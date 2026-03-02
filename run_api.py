"""
Run API Server
==============
Simple script to start the FastAPI server.
"""

import uvicorn
import sys
from pathlib import Path

# Add app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))


def main():
    """
    Start the FastAPI server.
    """
    print("=" * 70)
    print("🚀 Starting Twitter Sentiment Analyzer API")
    print("=" * 70)
    print("\n📍 Server will start at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/")
    print("💊 Health Check: http://localhost:8000/health")
    print("\n⚠️  Press CTRL+C to stop the server\n")
    print("=" * 70)
    
    # Run the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )


if __name__ == "__main__":
    main()
