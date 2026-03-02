"""
Test Script - Load and Test Trained Model
==========================================
This script loads the trained model and tests it with custom inputs.
Run this AFTER training the model.
"""

import joblib
import re
from pathlib import Path


def preprocess_text(text):
    """
    Preprocess text (same as in training).
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_model():
    """
    Load the trained model and vectorizer.
    """
    print("📦 Loading model and vectorizer...")
    
    model_path = Path('model/sentiment_model.pkl')
    vectorizer_path = Path('model/tfidf_vectorizer.pkl')
    
    if not model_path.exists() or not vectorizer_path.exists():
        print("❌ Error: Model files not found!")
        print("   Please run 'python train_model.py' first to train the model.")
        return None, None
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    print("✅ Model loaded successfully!")
    return model, vectorizer


def predict_sentiment(text, model, vectorizer):
    """
    Predict sentiment for a given text.
    """
    # Preprocess
    clean_text = preprocess_text(text)
    
    # Vectorize
    text_vector = vectorizer.transform([clean_text])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
    # Get confidence
    confidence = max(probabilities) * 100
    
    return prediction, confidence, probabilities


def main():
    """
    Main testing function.
    """
    print("=" * 70)
    print("🧪 SENTIMENT ANALYZER - TESTING SCRIPT")
    print("=" * 70)
    
    # Load model
    model, vectorizer = load_model()
    
    if model is None:
        return
    
    print("\n" + "=" * 70)
    print("💬 INTERACTIVE TESTING")
    print("=" * 70)
    print("Type 'quit' to exit\n")
    
    # Test some predefined examples first
    print("📝 Testing with predefined examples:\n")
    
    test_cases = [
        "I love this movie! It's absolutely fantastic!",
        "This is terrible. Waste of time and money.",
        "It's okay, nothing special.",
        "Best day ever! So happy right now!",
        "Disappointed with the service. Not recommended.",
        "Average experience. Could be better.",
    ]
    
    for text in test_cases:
        prediction, confidence, probs = predict_sentiment(text, model, vectorizer)
        
        print(f"Text: {text}")
        print(f"→ Sentiment: {prediction.upper()} ({confidence:.2f}% confidence)")
        print()
    
    # Interactive testing
    print("=" * 70)
    print("🎮 Now try your own text!")
    print("=" * 70)
    
    while True:
        print("\n" + "-" * 70)
        user_input = input("Enter text to analyze (or 'quit' to exit): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            print("⚠️  Please enter some text.")
            continue
        
        # Predict
        prediction, confidence, probs = predict_sentiment(user_input, model, vectorizer)
        
        # Display results
        print("\n📊 Results:")
        print(f"   Sentiment: {prediction.upper()}")
        print(f"   Confidence: {confidence:.2f}%")
        print(f"\n   Probability Distribution:")
        
        # Get class names from the model
        classes = model.classes_
        for cls, prob in zip(classes, probs):
            print(f"      {cls.capitalize()}: {prob*100:.2f}%")


if __name__ == "__main__":
    main()
