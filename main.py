# main.py - Production-ready Iris Classification API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import joblib
import numpy as np
import os
import logging
import time
from typing import Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load the trained model
MODEL_PATH = os.getenv("MODEL_PATH", "iris_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    class_names = ['setosa', 'versicolor', 'virginica']
    logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"❌ Model file not found: {MODEL_PATH}")
    raise Exception(f"Model file not found: {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    raise Exception(f"Failed to load model: {e}")

# Create FastAPI app
app = FastAPI(
    title="Iris Flower Classifier API",
    description="Predict iris species from flower measurements - Production Ready",
    version="1.0.0"
)

# Input data model with validation
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    # Validate that all measurements are positive
    @validator('*', pre=True)
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('All measurements must be positive')
        return v
    
    # Validate reasonable ranges (iris flowers aren't huge!)
    @validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    def validate_reasonable_range(cls, v):
        if v > 20:  # No iris flower has 20cm measurements
            raise ValueError('Measurement seems too large for an iris flower')
        return v

# Output data model
class IrisOutput(BaseModel):
    species: str
    confidence: float
    probabilities: dict
    processing_time_ms: Optional[float] = None

# Root endpoint - shows API info
@app.get("/")
def root():
    return {
        "message": "Iris Flower Classifier API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict - Make predictions",
            "health": "/health - Health check", 
            "docs": "/docs - Interactive documentation"
        },
        "example_request": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "timestamp": time.time()
    }

# Main prediction endpoint
@app.post("/predict", response_model=IrisOutput)
def predict_species(input_data: IrisInput):
    """
    Predict iris species from flower measurements
    
    Send measurements for sepal length, sepal width, petal length, and petal width.
    Returns predicted species with confidence score.
    """
    start_time = time.time()
    
    logger.info(f"Prediction request: {input_data}")
    
    try:
        # Convert input to numpy array in the correct order
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])
        
        # Make prediction
        prediction_index = model.predict(features)[0]
        prediction_probabilities = model.predict_proba(features)[0]
        
        # Get species name and confidence
        predicted_species = class_names[prediction_index]
        confidence = float(np.max(prediction_probabilities))
        
        # Create probabilities dictionary
        probabilities_dict = {
            class_names[i]: float(prob) 
            for i, prob in enumerate(prediction_probabilities)
        }
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        result = IrisOutput(
            species=predicted_species,
            confidence=confidence,
            probabilities=probabilities_dict,
            processing_time_ms=round(processing_time, 2)
        )
        
        logger.info(f"Prediction result: {predicted_species} (confidence: {confidence:.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

# Example endpoint to show users what data to send
@app.get("/example")
def get_example():
    return {
        "description": "Example data you can send to /predict endpoint",
        "example_data": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        "expected_result": {
            "species": "setosa",
            "confidence": 0.99,
            "probabilities": {
                "setosa": 0.99,
                "versicolor": 0.01,
                "virginica": 0.00
            }
        }
    }

# This is what makes it work on Render and other cloud platforms
if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment variables
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting Iris Classification API on {HOST}:{PORT}")
    logger.info(f"📊 Model: {MODEL_PATH}")
    logger.info(f"📝 Documentation: http://{HOST}:{PORT}/docs")
    
    # Run the app
    uvicorn.run(
        "main:app",    # app location
        host=HOST,     # where to run
        port=PORT,     # what port
        log_level=LOG_LEVEL.lower()  # logging level
    )