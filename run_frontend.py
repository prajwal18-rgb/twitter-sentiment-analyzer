"""
Run Frontend Server
===================
Simple script to start the Streamlit frontend.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """
    Start the Streamlit frontend.
    """
    print("=" * 70)
    print("🎨 Starting Twitter Sentiment Analyzer Frontend")
    print("=" * 70)
    print("\n📍 Frontend will start at: http://localhost:8501")
    print("⚠️  Make sure the API is running at: http://localhost:8000")
    print("   (Start API with: python run_api.py)")
    print("\n⚠️  Press CTRL+C to stop the frontend\n")
    print("=" * 70)
    print("\n")
    
    # Get the path to the frontend app
    frontend_path = Path(__file__).parent / "frontend" / "app.py"
    
    # Run streamlit
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


if __name__ == "__main__":
    main()
