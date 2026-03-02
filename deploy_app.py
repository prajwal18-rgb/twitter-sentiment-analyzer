"""
Standalone Streamlit App - For Cloud Deployment
================================================
This version includes the API client logic directly in the frontend,
making it easier to deploy as a single service.

For Render deployment: streamlit run deploy_app.py
"""

import streamlit as st
import joblib
import re
import time
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model and vectorizer."""
    try:
        model_path = Path("model/sentiment_model.pkl")
        vectorizer_path = Path("model/tfidf_vectorizer.pkl")
        
        if not model_path.exists() or not vectorizer_path.exists():
            st.error("❌ Model files not found! Please ensure model files are in the 'model/' directory.")
            return None, None
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None


def preprocess_text(text):
    """Preprocess text before prediction."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_sentiment(text, model, vectorizer):
    """Make sentiment prediction."""
    try:
        # Preprocess
        clean_text = preprocess_text(text)
        
        if not clean_text:
            return None, "Text contains no valid words after preprocessing"
        
        # Vectorize and predict
        text_vector = vectorizer.transform([clean_text])
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        
        # Get confidence
        confidence = float(max(probabilities) * 100)
        
        # Create probability dictionary
        classes = model.classes_
        prob_dict = {
            cls: round(float(prob * 100), 2)
            for cls, prob in zip(classes, probabilities)
        }
        
        return {
            'sentiment': prediction,
            'confidence': round(confidence, 2),
            'probabilities': prob_dict
        }, None
        
    except Exception as e:
        return None, str(e)


def display_sentiment_result(result):
    """Display the sentiment analysis results."""
    sentiment = result['sentiment']
    confidence = result['confidence']
    probabilities = result['probabilities']
    
    # Main result card
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
    else:
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
            value=f"{probabilities.get('positive', 0):.2f}%"
        )
        st.progress(probabilities.get('positive', 0) / 100)
    
    with col2:
        st.metric(
            label="😐 Neutral",
            value=f"{probabilities.get('neutral', 0):.2f}%"
        )
        st.progress(probabilities.get('neutral', 0) / 100)
    
    with col3:
        st.metric(
            label="😞 Negative",
            value=f"{probabilities.get('negative', 0):.2f}%"
        )
        st.progress(probabilities.get('negative', 0) / 100)


def main():
    """Main application."""
    # Header
    st.title("🐦 Twitter Sentiment Analyzer")
    st.markdown("""
        Analyze the sentiment of any text using **Machine Learning**!
        
        Enter your text below and click **Analyze** to see if it's positive, negative, or neutral.
    """)
    
    # Load model
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.error("⚠️ Model not loaded. Cannot make predictions.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
            This app uses a **Logistic Regression** model trained on tweet data.
            
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
            - "I love this product! Amazing!"
            - "This is terrible. Very disappointed."
            - "It's okay, nothing special."
        """)
        
        st.markdown("---")
        st.success("✅ Model Loaded")
    
    # Main input area
    st.markdown("---")
    
    # Text input
    user_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste your text here...\n\nExample: I absolutely love this product!",
        help="Enter any text to analyze its sentiment."
    )
    
    # Character counter
    if user_input:
        char_count = len(user_input)
        st.caption(f"📝 {char_count} characters")
        
        if char_count > 5000:
            st.warning("⚠️ Text is too long! Maximum 5000 characters.")
    
    # Analyze button
    if st.button("🚀 Analyze Sentiment", type="primary"):
        if not user_input or user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            with st.spinner("🔮 Analyzing sentiment..."):
                time.sleep(0.3)  # Small delay for UX
                
                result, error = predict_sentiment(user_input, model, vectorizer)
                
                if error:
                    st.error(f"❌ Error: {error}")
                elif result:
                    st.success("✅ Analysis complete!")
                    display_sentiment_result(result)
                    
                    with st.expander("📄 View Original Text"):
                        st.write(user_input)
    
    # Footer
    st.markdown("---")
    st.markdown("### 💡 Quick Test Examples")
    
    col1, col2, col3 = st.columns(3)
    
    examples = {
        "positive": "I absolutely love this product! Best purchase ever!",
        "neutral": "The product is okay. Nothing special.",
        "negative": "This is terrible. Complete waste of money!"
    }
    
    with col1:
        if st.button("😊 Positive Example"):
            st.session_state['example'] = examples['positive']
            st.rerun()
    
    with col2:
        if st.button("😐 Neutral Example"):
            st.session_state['example'] = examples['neutral']
            st.rerun()
    
    with col3:
        if st.button("😞 Negative Example"):
            st.session_state['example'] = examples['negative']
            st.rerun()
    
    # Handle example clicks
    if 'example' in st.session_state:
        st.info(f"📋 Example loaded: {st.session_state['example'][:50]}...")
        del st.session_state['example']
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>Built with ❤️ using Streamlit + scikit-learn</p>
            <p>End-to-End ML Project - Phase 4 Deployed!</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
