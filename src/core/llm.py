from langchain_core.callbacks.manager import CallbackManager
from langchain_core.outputs import ChatGenerationChunk
from typing import AsyncGenerator, Dict, Any, List
from functools import cache
from langchain_groq import ChatGroq
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from groq import AsyncGroq
import uuid
from src.core.settings import settings
from opik.integrations.langchain import OpikTracer
class StreamingCallbackHandler(BaseCallbackHandler):
    """Handler for streaming tokens."""
    
    def __init__(self):
        self.tokens = []
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Collect tokens as they're generated."""
        self.tokens.append(token)

@cache
def get_llm(model_name: str | None = None, streaming: bool = False) -> ChatGroq:
    """Get a cached LLM instance."""
    model = model_name or settings.DEFAULT_MODEL
    
    callback_manager = CallbackManager([StreamingCallbackHandler()]) if streaming else None
    
    return ChatGroq(
        model=model,
        api_key=settings.GROQ_API_KEY.get_secret_value(),
        temperature=settings.MODEL_TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        streaming=streaming,
        callback_manager=callback_manager if streaming else None,
    )


class StreamOpikTracer(OpikTracer):
    """Custom Opik tracer for streaming."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_run_id = None
        
    async def on_llm_start(self, serialized, messages, **kwargs):
        self.current_run_id = str(uuid.uuid4())
        return {"run_id": self.current_run_id}
        
    async def on_llm_new_token(self, token: str, **kwargs):
        pass
        
    async def on_llm_end(self, response, **kwargs):
        pass

async def generate_stream(
    messages: List[HumanMessage | AIMessage], 
    model_name: str | None = None,
    callbacks: List[Any] = None
) -> AsyncGenerator[str, None]:
    client = AsyncGroq(api_key=settings.GROQ_API_KEY.get_secret_value())
    tracer = StreamOpikTracer() if callbacks else None
    
    try:
        formatted_messages = [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in messages
        ]

        if tracer:
            await tracer.on_llm_start({"name": model_name or settings.DEFAULT_MODEL}, messages)
        
        async for chunk in await client.chat.completions.create(
            messages=formatted_messages,
            model=model_name or settings.DEFAULT_MODEL,
            temperature=settings.MODEL_TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            stream=True
        ):
            content = chunk.choices[0].delta.content
            if content:
                if tracer:
                    await tracer.on_llm_new_token(content)
                yield content
                
    except Exception as e:
        yield f"Error: {str(e)}"
    finally:
        if tracer:
            await tracer.on_llm_end(AIMessage(content=""))
        await client.close()