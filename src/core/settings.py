from pydantic_settings import BaseSettings
from pydantic import SecretStr

class Settings(BaseSettings):
    GROQ_API_KEY: SecretStr
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEFAULT_MODEL: str = "mixtral-8x7b-32768"
    
    class Config:
        env_file = ".env"

settings = Settings()
