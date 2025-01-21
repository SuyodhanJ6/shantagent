from functools import cache
from langchain_groq import ChatGroq
from src.core.settings import settings

@cache
def get_llm(model_name: str | None = None) -> ChatGroq:
    """Get a cached LLM instance."""
    model = model_name or settings.DEFAULT_MODEL
    return ChatGroq(
        model=model,
        api_key=settings.GROQ_API_KEY.get_secret_value(),
        temperature=0.7,
        streaming=True
    )