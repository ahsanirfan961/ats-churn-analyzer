import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ARTIFACTS_DIR / "churn_model.joblib"))
DATASET_PATH = Path(os.environ.get("DATASET_PATH", ARTIFACTS_DIR / "customer_churn_clean.csv"))

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

RECENT_TURNS_WINDOW = int(os.environ.get("RECENT_TURNS_WINDOW", 3))

CHECKPOINT_DB_PATH = os.environ.get("CHECKPOINT_DB_PATH", "./data/checkpoints.sqlite")
SESSIONS_DB_PATH = os.environ.get("SESSIONS_DB_PATH", "./data/sessions.sqlite")

CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
