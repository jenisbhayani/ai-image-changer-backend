from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Image Changer - Cloud Background Removal API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Hugging Face Access Token
    HF_TOKEN: str = ""

    # Specific Hugging Face model
    HF_MODEL_ID: str = "briaai/RMBG-1.4"

    # Maximum upload size in bytes (15 MB default)
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024

    # Allowed CORS origins
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Server configuration
    PORT: int = 7860
    HOST: str = "0.0.0.0"

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH) if ENV_PATH.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
