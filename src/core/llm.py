from typing import AsyncGenerator, Dict, Any
from functools import cache
from langchain_groq import ChatGroq
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage

from src.core.settings import settings

class TokenStreamHandler(BaseCallbackHandler):
    """Handler for streaming tokens."""
    
    def __init__(self):
        self.tokens = []
        
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Collects tokens as they're generated."""
        self.tokens.append(token)

@cache
def get_llm(model_name: str | None = None, streaming: bool = False) -> ChatGroq:
    """Get a cached LLM instance."""
    model = model_name or settings.DEFAULT_MODEL
    
    return ChatGroq(
        model=model,
        api_key=settings.GROQ_API_KEY.get_secret_value(),
        temperature=settings.MODEL_TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        streaming=streaming,
    )

async def generate_stream(
    messages: list[HumanMessage | AIMessage], 
    model_name: str | None = None
) -> AsyncGenerator[str, None]:
    """Generate streaming response."""
    
    llm = get_llm(model_name, streaming=True)
    handler = TokenStreamHandler()
    
    async for chunk in llm.astream(messages, callbacks=[handler]):
        if chunk.content:
            yield chunk.content