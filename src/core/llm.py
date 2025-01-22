# src/core/llm.py
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.outputs import ChatGenerationChunk
from typing import AsyncGenerator, Dict, Any, List, Optional
from functools import cache
from langchain_groq import ChatGroq
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from groq import AsyncGroq
from opik.integrations.langchain import OpikTracer
from src.core.settings import settings
from src.core.tracer import get_tracer, configure_tracer

class StreamingCallbackHandler(BaseCallbackHandler):
    """Handler for streaming tokens."""
    
    def __init__(self):
        self.tokens = []
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Collect tokens as they're generated."""
        self.tokens.append(token)

@cache
def get_llm(
    model_name: str | None = None, 
    streaming: bool = False, 
    trace: Optional[OpikTracer] = None
) -> ChatGroq:
    """Get a cached LLM instance with optional Opik tracing."""
    model = model_name or settings.DEFAULT_MODEL
    
    callback_manager = CallbackManager([StreamingCallbackHandler()]) if streaming else None
    
    llm = ChatGroq(
        model=model,
        api_key=settings.GROQ_API_KEY.get_secret_value(),
        temperature=settings.MODEL_TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        streaming=streaming,
        callback_manager=callback_manager if streaming else None,
    )
    
    # Wrap LLM with Opik tracer if provided
    if trace:
        trace = configure_tracer(trace)  # Configure tracer with metadata
        llm = trace.trace_llm(llm)
    
    return llm

async def generate_stream(
    messages: List[HumanMessage | AIMessage], 
    model_name: str | None = None
) -> AsyncGenerator[str, None]:
    """Generate streaming response."""
    client = AsyncGroq(api_key=settings.GROQ_API_KEY.get_secret_value())
    tracer = get_tracer()
    
    try:
        if tracer:
            tracer = configure_tracer(tracer)
            with tracer.start_span("generate_stream"):
                # Convert messages to the format expected by Groq
                formatted_messages = [
                    {
                        "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                        "content": msg.content
                    }
                    for msg in messages
                ]
                
                async for chunk in await client.chat.completions.create(
                    messages=formatted_messages,
                    model=model_name or settings.DEFAULT_MODEL,
                    temperature=settings.MODEL_TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS,
                    stream=True
                ):
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
        else:
            # If no tracer, proceed without tracing
            formatted_messages = [
                {
                    "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content
                }
                for msg in messages
            ]
            
            async for chunk in await client.chat.completions.create(
                messages=formatted_messages,
                model=model_name or settings.DEFAULT_MODEL,
                temperature=settings.MODEL_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
                stream=True
            ):
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
    except Exception as e:
        if tracer:
            tracer.log_error(str(e))
        yield f"Error: {str(e)}"
    finally:
        await client.close()  # Clean up client resources