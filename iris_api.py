# iris_api.py

# Import FastAPI components
from fastapi import FastAPI, HTTPException    # FastAPI and error handling
from pydantic import BaseModel, validator    # For data validation
import joblib                               # To load our saved model
import numpy as np                          # For numerical operations
from typing import List                     # For type hints

print("Loading the trained model...")
# Try to load the model we saved earlier
try:
    model = joblib.load("iris_model.pkl")   # Load the trained model from file
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    # If the model file doesn't exist, show an error
    print("❌ Model file not found! Run train_iris_model.py first!")
    raise Exception("Model file not found. Run train_iris_model.py first!")

# The three iris species our model can predict
class_names = ['setosa', 'versicolor', 'virginica']

# Create our FastAPI application
app = FastAPI(
    title="Iris Flower Classifier API",           # Name shown in docs
    description="Predict iris species from flower measurements",  # Description
    version="1.0.0"                              # Version number
)

# Define what data we expect people to send us
class IrisFeatures(BaseModel):
    """
    This class defines the structure of data we expect.
    Think of it like a form with required fields.
    """
    sepal_length: float    # Length of the sepal in cm (must be a decimal number)
    sepal_width: float     # Width of the sepal in cm
    petal_length: float    # Length of the petal in cm  
    petal_width: float     # Width of the petal in cm
    
    # Custom validation: make sure all measurements are positive
    @validator('*', pre=True)    # The '*' means "apply to all fields above"
    def validate_positive(cls, v):
        """Make sure people don't send negative measurements"""
        if v <= 0:
            # If someone sends a negative number, reject it
            raise ValueError('All measurements must be positive numbers')
        return v    # If it's positive, accept it

# Define what we'll send back to people
class IrisPrediction(BaseModel):
    """
    This defines the structure of our response.
    Like a receipt showing what we determined.
    """
    species: str           # The predicted species name
    confidence: float      # How confident we are (0.0 to 1.0)
    probabilities: dict    # Probability for each species

# Main page endpoint - just says hello
@app.get("/")
def read_root():
    """
    When someone visits the main page, show them what this API does
    """
    return {
        "message": "Iris Flower Classifier API",
        "description": "Send flower measurements to /predict to get species prediction",
        "endpoints": [
            "/predict - Send measurements, get species prediction", 
            "/health - Check if API is working",
            "/docs - Interactive documentation",
            "/example - See example input data"
        ]
    }

# Health check endpoint - confirms everything is working
@app.get("/health")
def health_check():
    """
    Other systems can call this to check if our API is healthy
    """
    return {
        "status": "healthy",
        "model_loaded": True,        # Confirms our ML model is ready
        "message": "API is running and ready for predictions"
    }

# THE BIG ONE: The prediction endpoint where the magic happens
@app.post("/predict", response_model=IrisPrediction)  # This will be a POST endpoint
def predict_iris(features: IrisFeatures):
    """
    This is where people send flower measurements and get back predictions.
    It's like the main counter at our restaurant where orders are processed.
    """
    print(f"Received prediction request: {features}")  # Log what we received
    
    try:
        # Convert the input data into the format our model expects
        input_data = np.array([[
            features.sepal_length,    # Put sepal length in first position
            features.sepal_width,     # Put sepal width in second position  
            features.petal_length,    # Put petal length in third position
            features.petal_width      # Put petal width in fourth position
        ]])
        
        print(f"Formatted input for model: {input_data}")
        
        # Ask our model to make a prediction
        prediction = model.predict(input_data)[0]           # Get the predicted class (0, 1, or 2)
        probabilities = model.predict_proba(input_data)[0]   # Get probability for each class
        
        print(f"Model prediction: {prediction}")
        print(f"Model probabilities: {probabilities}")
        
        # Convert the numeric prediction to a species name
        species = class_names[prediction]                   # Convert 0->setosa, 1->versicolor, etc.
        confidence = float(np.max(probabilities))           # Get the highest probability
        
        # Create a dictionary showing probability for each species
        prob_dict = {
            class_names[i]: float(prob)                     # Convert each probability to regular float
            for i, prob in enumerate(probabilities)         # For each species and its probability
        }
        
        print(f"Final result: {species} with {confidence:.3f} confidence")
        
        # Send back our prediction in the format we promised
        return IrisPrediction(
            species=species,         # The predicted species name
            confidence=confidence,   # How confident we are
            probabilities=prob_dict  # All probabilities
        )
        
    except Exception as e:
        # If something goes wrong, send back an error
        print(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,         # HTTP error code for "server error"
            detail=f"Prediction failed: {str(e)}"  # Error message to send back
        )

# Helper endpoint - shows people example data they can send
@app.get("/example")
def get_example():
    """
    Shows people what kind of data they should send to /predict
    """
    return {
        "message": "Here's an example of data you can send to /predict",
        "example_request": {
            "sepal_length": 5.1,     # Example measurements for a setosa flower
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        "how_to_use": "Send a POST request to /predict with this data format",
        "expected_result": "setosa"
    }

# This runs when we start the server
if __name__ == "__main__":
    print("🌸 Iris Classifier API is ready!")
    print("📝 Visit http://127.0.0.1:8000/docs to test it interactively")