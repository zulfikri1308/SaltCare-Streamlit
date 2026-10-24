import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "saltcare-dev-key")
    MODEL_PATH = os.path.join(
        BASE_DIR, "models", "random_forest_best.joblib"
    )
    DATABASE_PATH = os.path.join(
        BASE_DIR, "instance", "saltcare.db"
    )