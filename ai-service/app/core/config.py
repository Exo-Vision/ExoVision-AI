"""
Configuration settings for ExoVision AI Service
"""
import os
from pathlib import Path


class Settings:
    """Application settings"""
    
    # API Settings
    API_TITLE: str = "ExoVision AI Service"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Exoplanet Detection API using CatBoost and Voting Classifier"
    
    # CORS Settings
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # Development fallback
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: list = ["*"]
    
    # Model Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent
    MODEL_DIR: Path = BASE_DIR / "models"
    
    MODEL1_PATH: Path = MODEL_DIR / "model1_binary_catboost_20251005_195122.pkl"
    SCALER1_PATH: Path = MODEL_DIR / "scaler1_20251005_195122.pkl"
    MODEL2_PATH: Path = MODEL_DIR / "model2_candidate_voting_20251005_195122.pkl"
    SCALER2_PATH: Path = MODEL_DIR / "scaler2_20251005_195122.pkl"
    CONFIG_PATH: Path = MODEL_DIR / "config_20251005_195122.pkl"
    
    # Default values for beginner mode
    BEGINNER_DEFAULTS = {
        "ra": 286.591220,
        "koi_period": 5.825805,
        "koi_eccen": 0.000000,
        "koi_longp": 90.000000,
        "koi_incl": 89.000000,
        "koi_impact": 0.264301,
        "koi_sma": 0.063769,
        "koi_duration": 3.755000,
        "koi_depth": 1344.016115,
        "koi_insol": 236.318239,
        "koi_teq": 1091.122930,
        "koi_srad": 1.028000,
        "koi_smass": 0.989137,
        "koi_sage": 5.000000,
        "koi_steff": 5720.000000,
        "koi_slogg": 4.415000,
        "orbital_energy": 15.681531,
        "transit_signal": 9269.356704,
        "stellar_density": 0.910190,
        "log_period": 1.920710,
        "log_depth": 7.204161,
        "log_insol": 5.469402,
        "orbit_stability": 0.000000,
        "transit_snr": 291.805114,
    }


settings = Settings()
