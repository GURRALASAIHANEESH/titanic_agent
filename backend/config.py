# ==============================================================
# FILE: titanic_agent/backend/config.py
# ==============================================================

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Groq configuration
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    TEMPERATURE: float = float(os.environ.get("TEMPERATURE", "0"))

    # Dataset source
    TITANIC_CSV_URL: str = os.environ.get(
        "TITANIC_CSV_URL",
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    )
    TITANIC_CSV_PATH: str = os.environ.get("TITANIC_CSV_PATH", "")

    # FastAPI server
    BACKEND_HOST: str = os.environ.get("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT: int = int(os.environ.get("BACKEND_PORT", "8000"))
    ALLOWED_ORIGINS: list[str] = os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")

    # LangChain agent tuning
    AGENT_MAX_ITERATIONS: int = int(os.environ.get("AGENT_MAX_ITERATIONS", "10"))
    AGENT_VERBOSE: bool = os.environ.get("AGENT_VERBOSE", "false").lower() == "true"


settings = Settings()
