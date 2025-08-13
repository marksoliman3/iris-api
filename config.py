# config.py
import os
from typing import Optional

class Settings:
    # API Configuration
    API_TITLE: str = "Iris Classifier API"              # What shows up in docs
    API_VERSION: str = "1.0.0"                          # Version number
    API_DESCRIPTION: str = "Production-ready ML API"     # Description in docs
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")            # Where to run server
    PORT: int = int(os.getenv("PORT", "8000"))           # What port to use
    
    # Model Configuration  
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./iris_model.pkl")  # Where model is
    
    # Performance Settings
    WORKERS: int = int(os.getenv("WORKERS", "1"))        # How many worker processes
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "100"))  # Max items per batch
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")      # How much logging

settings = Settings()  # Create one instance to use everywhere