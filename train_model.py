"""
Twitter Sentiment Analyzer - Model Training Script (Windows Fixed)
===================================================
This script trains a sentiment analysis model using Logistic Regression and TF-IDF.

Author: Your Name
Date: February 2026
"""

# FIX: Set matplotlib to non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows

import pandas as pd
import numpy as np
import re
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class SentimentAnalyzer:
    """
    A complete sentiment analysis pipeline with training and prediction capabilities.
    """
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        
    def load_data(self):
        """
        Load the sentiment dataset.
        We'll use a sample dataset for this project.
        """
        print("📥 Loading dataset...")
        
        # For this project, we'll create a function to download a public dataset
        # You can also manually download from Kaggle: Sentiment140 dataset
        
        try:
            # Try to load from local data folder
            df = pd.read_csv('data/tweets.csv', encoding='latin-1')
            print(f"✅ Loaded {len(df)} tweets from local file")
        except FileNotFoundError:
            print("⚠️  Local dataset not found. Creating a sample dataset...")
            print("💡 For production, download the Sentiment140 dataset from Kaggle")
            
            # Create a sample dataset for demonstration
            # In production, replace this with actual Twitter data
            sample_data = self._create_sample_dataset()
            df = pd.DataFrame(sample_data)
            
            # Save it for future use
            Path('data').mkdir(exist_ok=True)
            df.to_csv('data/tweets.csv', index=False)
            print(f"✅ Created sample dataset with {len(df)} tweets")
        
        return df
    
    def _create_sample_dataset(self):
        """
        Create a sample dataset for demonstration purposes.
        In production, you should use real Twitter data.
        """
        positive_tweets = [
            "I love this product! It's amazing!",
            "What a wonderful day! Feeling great!",
            "Best experience ever! Highly recommend!",
            "This is fantastic! Really enjoying it!",
            "Absolutely brilliant! 5 stars!",
            "So happy with this purchase!",
            "Excellent service and great quality!",
            "I'm thrilled with the results!",
            "Outstanding performance!",
            "This made my day! Thank you!",
            "Perfect! Exactly what I needed!",
            "Awesome product! Love it!",
            "Great value for money!",
            "Superb quality and fast delivery!",
            "Impressive! Will buy again!",
        ] * 100  # Repeat to get more samples
        
        negative_tweets = [
            "This is terrible. Very disappointed.",
            "Worst purchase ever. Don't buy this!",
            "Complete waste of money and time.",
            "Horrible experience. Never again!",
            "This product is awful. Broke immediately.",
            "Terrible customer service!",
            "I hate this. Total disaster!",
            "Disappointing quality. Not worth it.",
            "Bad experience. Would not recommend.",
            "This is garbage. Save your money!",
            "Frustrated and angry with this purchase!",
            "Poor quality and overpriced!",
            "Worst decision ever!",
            "Absolutely useless product!",
            "Terrible! Wants refund immediately!",
        ] * 100  # Repeat to get more samples
        
        neutral_tweets = [
            "It's okay, nothing special.",
            "Average product. Does the job.",
            "It works as expected.",
            "Standard quality. Nothing remarkable.",
            "Received the product. It's fine.",
            "Delivered on time. Product is okay.",
            "Not bad, not great either.",
            "It's alright. Could be better.",
            "Meets basic expectations.",
            "Decent but not impressive.",
        ] * 150  # Repeat to get more samples
        
        # Combine all tweets
        tweets = positive_tweets + negative_tweets + neutral_tweets
        sentiments = (['positive'] * len(positive_tweets) + 
                     ['negative'] * len(negative_tweets) + 
                     ['neutral'] * len(neutral_tweets))
        
        return {'text': tweets, 'sentiment': sentiments}
    
    def preprocess_text(self, text):
        """
        Clean and preprocess tweet text.
        
        Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove mentions (@username)
        4. Remove hashtags (keep the text)
        5. Remove special characters
        6. Remove extra whitespace
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the text)
        text = re.sub(r'#', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_data(self, df):
        """
        Prepare data for training.
        """
        print("\n🔧 Preprocessing text data...")
        
        # Apply preprocessing
        df['clean_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['clean_text'].str.len() > 0]
        
        print(f"✅ Preprocessed {len(df)} tweets")
        
        # Display sentiment distribution
        print("\n📊 Sentiment Distribution:")
        print(df['sentiment'].value_counts())
        
        return df
    
    def train(self, X_train, y_train):
        """
        Train the sentiment analysis model.
        """
        print("\n🎯 Training the model...")
        
        # Create TF-IDF vectorizer
        print("   Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,      # Use top 5000 words
            min_df=5,                # Ignore words that appear in less than 5 documents
            max_df=0.8,              # Ignore words that appear in more than 80% of documents
            ngram_range=(1, 2),      # Use unigrams and bigrams
            stop_words='english'     # Remove common English stop words
        )
        
        # Transform training data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"   ✅ Created {X_train_tfidf.shape[1]} TF-IDF features")
        
        # Train Logistic Regression model
        print("   Training Logistic Regression classifier...")
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,                   # Regularization strength
            solver='lbfgs'           # Optimization algorithm
        )
        
        self.model.fit(X_train_tfidf, y_train)
        print("   ✅ Model training complete!")
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        """
        print("\n📈 Evaluating model performance...")
        
        # Transform test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_tfidf)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Display classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create confusion matrix
        print("\n🔍 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, y_test)
        
        return accuracy
    
    def _plot_confusion_matrix(self, cm, y_test):
        """
        Plot confusion matrix heatmap.
        """
        try:
            plt.figure(figsize=(10, 8))
            
            # Get unique labels
            labels = sorted(y_test.unique())
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=labels, yticklabels=labels)
            plt.title('Confusion Matrix - Sentiment Analysis', fontsize=16, pad=20)
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            
            # Save plot
            Path('model').mkdir(exist_ok=True)
            plt.savefig('model/confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("   ✅ Confusion matrix saved to 'model/confusion_matrix.png'")
            plt.close()
        except Exception as e:
            print(f"   ⚠️  Could not save confusion matrix plot: {e}")
            print("   ℹ️  This is okay - the model is still trained successfully!")
    
    def save_model(self):
        """
        Save the trained model and vectorizer.
        """
        print("\n💾 Saving model and vectorizer...")
        
        # Create model directory
        Path('model').mkdir(exist_ok=True)
        
        # Save vectorizer
        joblib.dump(self.vectorizer, 'model/tfidf_vectorizer.pkl')
        print("   ✅ Vectorizer saved to 'model/tfidf_vectorizer.pkl'")
        
        # Save model
        joblib.dump(self.model, 'model/sentiment_model.pkl')
        print("   ✅ Model saved to 'model/sentiment_model.pkl'")
    
    def predict(self, texts):
        """
        Predict sentiment for new texts.
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Preprocess texts
        clean_texts = [self.preprocess_text(text) for text in texts]
        
        # Transform to TF-IDF
        X_tfidf = self.vectorizer.transform(clean_texts)
        
        # Predict
        predictions = self.model.predict(X_tfidf)
        probabilities = self.model.predict_proba(X_tfidf)
        
        return predictions, probabilities


def main():
    """
    Main function to train and save the sentiment analysis model.
    """
    print("=" * 70)
    print("🚀 TWITTER SENTIMENT ANALYZER - MODEL TRAINING")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Load data
    df = analyzer.load_data()
    
    # Prepare data
    df = analyzer.prepare_data(df)
    
    # Split data
    print("\n✂️  Splitting data into train and test sets...")
    X = df['clean_text']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Train model
    analyzer.train(X_train, y_train)
    
    # Evaluate model
    accuracy = analyzer.evaluate(X_test, y_test)
    
    # Save model
    analyzer.save_model()
    
    # Test predictions on sample texts
    print("\n" + "=" * 70)
    print("🧪 TESTING MODEL WITH SAMPLE PREDICTIONS")
    print("=" * 70)
    
    sample_texts = [
        "I absolutely love this product! It's amazing!",
        "This is the worst thing I've ever bought. Terrible!",
        "It's okay, nothing special about it.",
        "Best purchase ever! Highly recommend!",
        "Waste of money. Very disappointed."
    ]
    
    predictions, probabilities = analyzer.predict(sample_texts)
    
    print("\nSample Predictions:")
    print("-" * 70)
    for text, pred, prob in zip(sample_texts, predictions, probabilities):
        max_prob = max(prob) * 100
        print(f"\nText: {text}")
        print(f"Sentiment: {pred.upper()} (Confidence: {max_prob:.2f}%)")
    
    print("\n" + "=" * 70)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print("\n📦 Saved Files:")
    print("   - model/sentiment_model.pkl")
    print("   - model/tfidf_vectorizer.pkl")
    print("   - model/confusion_matrix.png (if visualization worked)")
    print("\n🎯 Next Steps:")
    print("   1. Review the model performance metrics above")
    print("   2. Test with: python test_model.py")
    print("   3. Proceed to Phase 1: Build the FastAPI backend")
    print("\n")


if __name__ == "__main__":
    main()
