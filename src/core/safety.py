from functools import wraps
from typing import Callable, TypeVar, ParamSpec, Awaitable, Any, Dict, AsyncGenerator
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
import json
from datetime import datetime



from src.core.llama_guard import llama_guard, SafetyAssessment
from src.schema.models import UserInput, ChatMessage

P = ParamSpec('P')
T = TypeVar('T')

def create_safety_response(categories: list[str], stage: str = "input") -> ChatMessage:
    """Create a standardized safety violation response."""
    return ChatMessage(
        type="ai",
        content=f"Content was flagged as unsafe for the following categories: {', '.join(categories)}",
        metadata={
            "safety": "unsafe",
            "stage": stage,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def check_content_safety(content: str, role: str = "human") -> tuple[bool, list[str]]:
    """Check content safety and return status and categories if unsafe."""
    safety_check = await llama_guard.ainvoke(
        role, 
        [HumanMessage(content=content) if role == "human" else AIMessage(content=content)]
    )
    
    if safety_check.safety_assessment == SafetyAssessment.UNSAFE:
        return False, safety_check.unsafe_categories
    return True, []

def check_safety(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """Decorator to check both input and output safety using LlamaGuard."""
    
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # Find UserInput in kwargs or args
        user_input = next(
            (arg for arg in args if isinstance(arg, UserInput)),
            next((v for v in kwargs.values() if isinstance(v, UserInput)), None)
        )
        
        if not user_input:
            raise ValueError("No UserInput found in function arguments")

        # Check input safety
        is_safe, categories = await check_content_safety(user_input.message, "human")
        if not is_safe:
            return create_safety_response(categories, "input")

        # Call the original function
        result = await func(*args, **kwargs)

        # For streaming endpoints or non-ChatMessage responses, return directly
        if isinstance(result, StreamingResponse) or not isinstance(result, ChatMessage):
            return result

        # Check output safety
        is_safe, categories = await check_content_safety(result.content, "ai")
        if not is_safe:
            return create_safety_response(categories, "output")

        return result

    return wrapper

async def safety_stream_wrapper(
    user_input: UserInput,
    stream_func: Callable[[UserInput], AsyncGenerator[str, None]]
) -> AsyncGenerator[str, None]:
    """Wrapper for streaming responses with safety checks."""
    
    # Check input safety first
    is_safe, categories = await check_content_safety(user_input.message, "human")
    if not is_safe:
        unsafe_response = create_safety_response(categories, "input")
        yield f"data: {json.dumps({'type': 'message', 'content': unsafe_response.content})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Initialize content accumulator for output safety checks
    accumulated_content = ""
    
    async for chunk in stream_func(user_input):
        if chunk.startswith("data: "):
            try:
                data = json.loads(chunk[6:])
                if data.get("type") == "token":
                    accumulated_content += data["content"]
                    # Check safety periodically (e.g., after complete sentences)
                    if any(end in data["content"] for end in [". ", "! ", "? ", "\n"]):
                        is_safe, categories = await check_content_safety(accumulated_content, "ai")
                        if not is_safe:
                            unsafe_response = create_safety_response(categories, "output")
                            yield f"data: {json.dumps({'type': 'message', 'content': unsafe_response.content})}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                yield chunk
            except json.JSONDecodeError:
                yield chunk
        else:
            yield chunk