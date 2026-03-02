"""
Run Full Application
====================
Starts both API and Frontend together.
"""

import subprocess
import sys
import time
from pathlib import Path
import os


def main():
    """
    Start both API server and Frontend.
    """
    print("=" * 70)
    print("🚀 Starting COMPLETE Twitter Sentiment Analyzer Application")
    print("=" * 70)
    print("\n📦 Starting services...")
    print("   1. API Server (http://localhost:8000)")
    print("   2. Frontend UI (http://localhost:8501)")
    print("\n⏳ Please wait while services start...")
    print("\n⚠️  Press CTRL+C to stop all services\n")
    print("=" * 70)
    
    # Start API server in background
    print("\n🔧 Starting API server...")
    api_process = subprocess.Popen([
        sys.executable,
        "run_api.py"
    ])
    
    # Wait for API to start
    print("⏳ Waiting for API to be ready...")
    time.sleep(5)
    
    # Start Frontend
    print("\n🎨 Starting Frontend...")
    frontend_path = Path(__file__).parent / "frontend" / "app.py"
    
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(frontend_path),
            "--server.headless=true",
            "--server.port=8501",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down services...")
        api_process.terminate()
        print("✅ All services stopped!")


if __name__ == "__main__":
    main()
