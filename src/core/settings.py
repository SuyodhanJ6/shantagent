# from typing import Annotated
# from pydantic import SecretStr, BeforeValidator
# from pydantic_settings import BaseSettings, SettingsConfigDict
# from functools import lru_cache

# def check_api_key(v: str | None) -> str | None:
#     if v is None or len(v.strip()) == 0:
#         raise ValueError("API key cannot be empty")
#     return v

# class Settings(BaseSettings):
#     model_config = SettingsConfigDict(
#         env_file=".env",
#         env_file_encoding="utf-8",
#         env_ignore_empty=True,
#     )

#     # API Configuration
#     GROQ_API_KEY: Annotated[SecretStr, BeforeValidator(check_api_key)]
#     HOST: str = "0.0.0.0"
#     PORT: int = 8000
    
#     # Auth Configuration
#     AUTH_SECRET: SecretStr | None = None
    
#     # Model Configuration
#     DEFAULT_MODEL: str = "mixtral-8x7b-32768"
#     MODEL_TEMPERATURE: float = 0.7
#     MAX_TOKENS: int = 2048
    
#     # Database Configuration
#     DB_URL: str = "sqlite:///./chatbot.db"
    
#     # Development Mode
#     DEBUG: bool = False

#     # Research Configuration
#     TAVILY_API_KEY: Annotated[SecretStr, BeforeValidator(check_api_key)]
    
#     # Research specific settings
#     MAX_SEARCH_RESULTS: int = 3
#     SEARCH_TIMEOUT: int = 30

# @lru_cache
# def get_settings() -> Settings:
#     return Settings()

# settings = get_settings()
from typing import Annotated
from pydantic import SecretStr, BeforeValidator, Field
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

    # Opik Configuration
    OPIK_URL_OVERRIDE: str = Field(
        default="http://localhost:5173/api",
        description="URL of the Opik server"
    )
    OPIK_WORKSPACE: str = Field(
        default="default",
        description="Opik workspace name"
    )
    OPIK_PROJECT_NAME: str = Field(
        default="chat_agent",
        description="Opik project name"
    )
    OPIK_TRACK_DISABLE: bool = Field(
        default=False,
        description="Disable Opik tracking"
    )
    OPIK_CHECK_TLS_CERTIFICATE: bool = Field(
        default=False,
        description="Check TLS certificate for Opik server"
    )
    OPIK_DEFAULT_FLUSH_TIMEOUT: int = Field(
        default=30,
        description="Default timeout for flushing traces"
    )

    def get_opik_config(self) -> dict:
        """Get Opik configuration as a dictionary."""
        return {
            "use_local": True,
            "server_url": self.OPIK_URL_OVERRIDE,
            "project": self.OPIK_PROJECT_NAME,
            "workspace": self.OPIK_WORKSPACE,
            "disable_tracking": self.OPIK_TRACK_DISABLE,
            "verify_ssl": self.OPIK_CHECK_TLS_CERTIFICATE,
            "flush_timeout": self.OPIK_DEFAULT_FLUSH_TIMEOUT
        }

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()