from typing import Annotated
from pydantic import SecretStr, BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

def check_api_key(v: str | None) -> str | None:
    if v is None or len(v.strip()) == 0:
        raise ValueError("API key cannot be empty")
    return v

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
    )

    # API Configuration
    GROQ_API_KEY: Annotated[SecretStr, BeforeValidator(check_api_key)]
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Auth Configuration
    AUTH_SECRET: SecretStr | None = None
    
    # Model Configuration
    DEFAULT_MODEL: str = "mixtral-8x7b-32768"
    MODEL_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2048
    
    # Database Configuration
    DB_URL: str = "sqlite:///./chatbot.db"
    
    # Development Mode
    DEBUG: bool = False

    # Research Configuration
    TAVILY_API_KEY: Annotated[SecretStr, BeforeValidator(check_api_key)]
    
    # Research specific settings
    MAX_SEARCH_RESULTS: int = 3
    SEARCH_TIMEOUT: int = 30

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()