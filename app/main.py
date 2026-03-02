"""
FastAPI Backend - Twitter Sentiment Analyzer
=============================================
REST API for serving sentiment analysis predictions.

Endpoints:
- GET  /health   - Health check
- POST /predict  - Sentiment prediction
- GET  /         - API documentation
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import re
from pathlib import Path
import logging
from typing import Optional

from schemas import PredictionRequest, PredictionResponse, HealthResponse, ErrorResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Twitter Sentiment Analyzer API",
    description="API for analyzing sentiment in text using Machine Learning",
    version="1.0.0",
    docs_url="/",  # Swagger UI at root
    redoc_url="/redoc"
)

# Enable CORS (allows frontend from different domain to access API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and vectorizer
model = None
vectorizer = None
model_loaded = False


def preprocess_text(text: str) -> str:
    """
    Preprocess text before prediction.
    Same preprocessing as used during training.
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


def load_model():
    """
    Load the trained model and vectorizer at startup.
    """
    global model, vectorizer, model_loaded
    
    try:
        logger.info("Loading model and vectorizer...")
        
        # Path to model files
        model_path = Path(__file__).parent.parent / "model" / "sentiment_model.pkl"
        vectorizer_path = Path(__file__).parent.parent / "model" / "tfidf_vectorizer.pkl"
        
        # Check if files exist
        if not model_path.exists():
            logger.error(f"Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not vectorizer_path.exists():
            logger.error(f"Vectorizer file not found at: {vectorizer_path}")
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
        
        # Load model and vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        model_loaded = True
        logger.info("✅ Model and vectorizer loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        model_loaded = False
        raise


@app.on_event("startup")
async def startup_event():
    """
    Run when API starts - load the model.
    """
    logger.info("🚀 Starting Twitter Sentiment Analyzer API...")
    load_model()
    logger.info("✅ API ready to serve predictions!")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns API status and whether the model is loaded.
    Used by cloud services to monitor if the API is running.
    """
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        version="1.0.0"
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid input"},
        500: {"description": "Server error"}
    }
)
async def predict_sentiment(request: PredictionRequest):
    """
    Predict sentiment for the given text.
    
    **Input:**
    - text: String to analyze (1-5000 characters)
    
    **Output:**
    - sentiment: Predicted class (positive/negative/neutral)
    - confidence: Confidence score (0-100)
    - probabilities: Probability for each class
    
    **Example Request:**
    ```json
    {
        "text": "I love this product! It's amazing!"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "text": "I love this product! It's amazing!",
        "sentiment": "positive",
        "confidence": 99.87,
        "probabilities": {
            "positive": 99.87,
            "negative": 0.08,
            "neutral": 0.05
        }
    }
    ```
    """
    try:
        # Check if model is loaded
        if not model_loaded or model is None or vectorizer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please contact the administrator."
            )
        
        # Get input text
        input_text = request.text.strip()
        
        # Validate input
        if not input_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )
        
        # Preprocess text
        clean_text = preprocess_text(input_text)
        
        # Check if preprocessing resulted in empty text
        if not clean_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text contains no valid words after preprocessing"
            )
        
        # Vectorize text
        text_vector = vectorizer.transform([clean_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        
        # Get confidence (max probability)
        confidence = float(max(probabilities) * 100)
        
        # Create probability dictionary
        classes = model.classes_
        prob_dict = {
            cls: round(float(prob * 100), 2) 
            for cls, prob in zip(classes, probabilities)
        }
        
        # Log prediction
        logger.info(f"Prediction: {prediction} (confidence: {confidence:.2f}%)")
        
        # Return response
        return PredictionResponse(
            text=input_text,
            sentiment=prediction,
            confidence=round(confidence, 2),
            probabilities=prob_dict
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/info", tags=["Info"])
async def api_info():
    """
    Get information about the API and model.
    """
    return {
        "api_name": "Twitter Sentiment Analyzer",
        "version": "1.0.0",
        "model_type": "Logistic Regression + TF-IDF",
        "supported_sentiments": ["positive", "negative", "neutral"],
        "max_text_length": 5000,
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "info": "GET /info",
            "docs": "GET / (Swagger UI)"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unexpected errors.
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# Root endpoint - redirects to docs
@app.get("/api", tags=["Info"])
async def root():
    """
    API root endpoint with quick start guide.
    """
    return {
        "message": "Welcome to Twitter Sentiment Analyzer API!",
        "documentation": "Visit / for interactive API docs",
        "quick_start": {
            "1": "Send POST request to /predict with JSON body: {'text': 'your text here'}",
            "2": "Get sentiment prediction with confidence scores",
            "3": "Check /health for API status"
        },
        "example_curl": "curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{\"text\":\"I love this!\"}'"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )
