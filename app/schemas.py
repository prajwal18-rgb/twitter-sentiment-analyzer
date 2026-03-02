"""
API Schemas - Request and Response Models
==========================================
Pydantic models for validating API inputs and outputs.
"""

from pydantic import BaseModel, Field
from typing import Dict


class PredictionRequest(BaseModel):
    """
    Request model for sentiment prediction.
    
    Example:
    {
        "text": "I love this product!"
    }
    """
    text: str = Field(
        ..., 
        min_length=1,
        max_length=5000,
        description="Text to analyze for sentiment",
        example="I love this product! It's amazing!"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this product! It's amazing!"
            }
        }


class PredictionResponse(BaseModel):
    """
    Response model for sentiment prediction.
    
    Example:
    {
        "text": "I love this product!",
        "sentiment": "positive",
        "confidence": 99.87,
        "probabilities": {
            "positive": 99.87,
            "negative": 0.08,
            "neutral": 0.05
        }
    }
    """
    text: str = Field(..., description="Original input text")
    sentiment: str = Field(..., description="Predicted sentiment (positive/negative/neutral)")
    confidence: float = Field(..., description="Confidence score (0-100)")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution for all classes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this product!",
                "sentiment": "positive",
                "confidence": 99.87,
                "probabilities": {
                    "positive": 99.87,
                    "negative": 0.08,
                    "neutral": 0.05
                }
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    """
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    version: str = Field(..., description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """
    Response model for errors.
    """
    error: str = Field(..., description="Error message")
    detail: str = Field(None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid input",
                "detail": "Text field cannot be empty"
            }
        }
