"""Configuration management for the application.

Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env.local if it exists
env_path = Path(".env.local")
if env_path.exists():
    load_dotenv(env_path)


class Config:
    """Application configuration loaded from environment variables."""
    
    # Google Cloud Configuration
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS", 
        ""
    )
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    # Vertex AI Model Configuration
    VERTEX_AI_MODEL_NAME: str = os.getenv("VERTEX_AI_MODEL_NAME", "gemini-2.5-flash-lite")
    VERTEX_AI_EMBEDDING_MODEL: str = os.getenv(
        "VERTEX_AI_EMBEDDING_MODEL", 
        "text-embedding-004"
    )
    
    # Vector DB Configuration
    VECTOR_DB_COLLECTION: str = os.getenv("VECTOR_DB_COLLECTION", "ng12_guidelines")
    
    # Application Configuration
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
    VECTOR_DB_PATH: Path = Path(os.getenv("VECTOR_DB_PATH", "data/vector_db"))
    PATIENTS_JSON_PATH: Path = Path(os.getenv("PATIENTS_JSON_PATH", "data/patients.json"))


# Global config instance
config = Config()
