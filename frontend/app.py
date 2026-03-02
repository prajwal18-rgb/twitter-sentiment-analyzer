"""
Streamlit Frontend - Twitter Sentiment Analyzer
================================================
Beautiful web interface for sentiment analysis.

Run with: streamlit run frontend/app.py
"""

import streamlit as st
import requests
import json
import time
from typing import Optional, Dict

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1DA1F2;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #0c85d0;
        border: none;
    }
    .sentiment-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """
    Check if the API is running and healthy.
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200 and response.json().get("model_loaded", False)
    except:
        return False


def predict_sentiment(text: str) -> Optional[Dict]:
    """
    Send text to API and get sentiment prediction.
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API! Make sure the API server is running.")
        st.info("💡 Start the API with: `python run_api.py`")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The API might be overloaded.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None


def display_sentiment_result(result: Dict):
    """
    Display the sentiment analysis results beautifully.
    """
    sentiment = result['sentiment']
    confidence = result['confidence']
    probabilities = result['probabilities']
    
    # Main result card with color based on sentiment
    if sentiment == 'positive':
        st.markdown(f"""
            <div class="sentiment-positive">
                <h2>😊 POSITIVE</h2>
                <p style="font-size: 24px; margin: 0;"><strong>{confidence:.2f}%</strong> confident</p>
            </div>
        """, unsafe_allow_html=True)
    elif sentiment == 'negative':
        st.markdown(f"""
            <div class="sentiment-negative">
                <h2>😞 NEGATIVE</h2>
                <p style="font-size: 24px; margin: 0;"><strong>{confidence:.2f}%</strong> confident</p>
            </div>
        """, unsafe_allow_html=True)
    else:  # neutral
        st.markdown(f"""
            <div class="sentiment-neutral">
                <h2>😐 NEUTRAL</h2>
                <p style="font-size: 24px; margin: 0;"><strong>{confidence:.2f}%</strong> confident</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Detailed probability breakdown
    st.markdown("### 📊 Detailed Probability Breakdown")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="😊 Positive",
            value=f"{probabilities.get('positive', 0):.2f}%",
            delta=None
        )
        st.progress(probabilities.get('positive', 0) / 100)
    
    with col2:
        st.metric(
            label="😐 Neutral",
            value=f"{probabilities.get('neutral', 0):.2f}%",
            delta=None
        )
        st.progress(probabilities.get('neutral', 0) / 100)
    
    with col3:
        st.metric(
            label="😞 Negative",
            value=f"{probabilities.get('negative', 0):.2f}%",
            delta=None
        )
        st.progress(probabilities.get('negative', 0) / 100)


def main():
    """
    Main application.
    """
    # Header
    st.title("🐦 Twitter Sentiment Analyzer")
    st.markdown("""
        Analyze the sentiment of any text using **Machine Learning**!
        
        Enter your text below and click **Analyze** to see if it's positive, negative, or neutral.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
            This app uses a **Logistic Regression** model trained on tweet data to classify sentiment.
            
            **Model Details:**
            - Algorithm: Logistic Regression
            - Features: TF-IDF
            - Classes: Positive, Negative, Neutral
            - Accuracy: ~98% (on sample data)
        """)
        
        st.markdown("---")
        
        st.header("📝 Sample Texts")
        st.markdown("""
            Try these examples:
            - "I love this product! It's amazing!"
            - "This is terrible. Very disappointed."
            - "It's okay, nothing special."
        """)
        
        st.markdown("---")
        
        # API Status
        st.header("🔌 API Status")
        if check_api_health():
            st.success("✅ Connected")
        else:
            st.error("❌ Disconnected")
            st.info("Start API with: `python run_api.py`")
    
    # Main input area
    st.markdown("---")
    
    # Text input
    user_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste your text here...\n\nExample: I absolutely love this product! It exceeded all my expectations!",
        help="Enter any text (tweet, review, comment, etc.) to analyze its sentiment."
    )
    
    # Character counter
    if user_input:
        char_count = len(user_input)
        st.caption(f"📝 {char_count} characters")
        
        if char_count > 5000:
            st.warning("⚠️ Text is too long! Maximum 5000 characters allowed.")
    
    # Analyze button
    if st.button("🚀 Analyze Sentiment", type="primary"):
        if not user_input or user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            # Check API health first
            if not check_api_health():
                st.error("❌ API is not running! Please start the API server first.")
                st.code("python run_api.py", language="bash")
            else:
                # Show loading spinner
                with st.spinner("🔮 Analyzing sentiment..."):
                    # Add a small delay for better UX
                    time.sleep(0.5)
                    
                    # Get prediction
                    result = predict_sentiment(user_input)
                    
                    if result:
                        st.success("✅ Analysis complete!")
                        
                        # Display results
                        display_sentiment_result(result)
                        
                        # Show original text
                        with st.expander("📄 View Original Text"):
                            st.write(result['text'])
    
    # Footer with quick examples
    st.markdown("---")
    st.markdown("### 💡 Quick Test Examples")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("😊 Positive Example"):
            st.session_state['example_text'] = "I absolutely love this product! It's the best thing I've ever bought. Highly recommend to everyone!"
            st.rerun()
    
    with col2:
        if st.button("😐 Neutral Example"):
            st.session_state['example_text'] = "The product is okay. It works as expected. Nothing special, but it does the job."
            st.rerun()
    
    with col3:
        if st.button("😞 Negative Example"):
            st.session_state['example_text'] = "This is the worst purchase I've ever made. Complete waste of money. Very disappointed!"
            st.rerun()
    
    # Handle example button clicks
    if 'example_text' in st.session_state:
        st.info(f"📋 Example loaded! Click 'Analyze Sentiment' to see results.")
        st.session_state.clear()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>Built with ❤️ using FastAPI + Streamlit + scikit-learn</p>
            <p>Phase 2 of End-to-End ML Project</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
