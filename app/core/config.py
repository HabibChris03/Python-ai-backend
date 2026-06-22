from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_TITLE: str = "DocFinder AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""  # kept for backward compatibility
    TESSERACT_PATH: str = ""

    DEVICE: str = "cpu"

    # Google Cloud Vision (optional)
    GOOGLE_VISION_ENABLED: bool = False
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    # Scan performance
    SCAN_MAX_EDGE: int = 1600
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:4000",
        "http://10.152.88.66:8001",
        "http://10.152.88.66:4000",
        "*",  # Allow all origins for React Native / Expo mobile clients
    ]
    API_KEY: str = ""  # internal service API key for auth between Node and Python
    
    # Node.js backend URL for API calls
    NODEJS_BACKEND_URL: str = "https://docfinder-backend.vercel.app/api"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
