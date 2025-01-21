
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
from typing import AsyncGenerator

from src.schema.models import ChatMessage, UserInput
from src.agents.react_agent import react_agent

router = APIRouter(prefix="/react", tags=["react"])

@router.post("")
async def react_chat(user_input: UserInput) -> ChatMessage:
    """
    ReAct agent endpoint that uses tools for enhanced responses.
    """
    try:
        result = await react_agent.handle_message(
            message=user_input.message,
            thread_id=user_input.thread_id,
            model=user_input.model,
            metadata=user_input.metadata
        )
        
        return ChatMessage(
            type="ai",
            content=result["response"],
            metadata={
                "thread_id": result["thread_id"] if result["thread_id"] else None,
                "tools_used": result.get("tools_used", [])
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing ReAct request: {str(e)}"
        )

async def _stream_generator(user_input: UserInput) -> AsyncGenerator[str, None]:
    """Generate streaming response for ReAct agent."""
    try:
        # Process with agent
        result = await react_agent.handle_message(
            message=user_input.message,
            thread_id=user_input.thread_id,
            model=user_input.model,
            metadata=user_input.metadata
        )
        
        # Stream response in chunks
        response = result["response"]
        chunk_size = 100
        
        # Stream main content
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i + chunk_size]
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            
        # Stream tool usage info if available
        if tools_used := result.get("tools_used"):
            yield f"data: {json.dumps({'type': 'tools_used', 'content': tools_used})}\n\n"
            
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

@router.post("/stream")
async def stream_react_chat(user_input: UserInput) -> StreamingResponse:
    """Stream ReAct agent responses."""
    return StreamingResponse(
        _stream_generator(user_input),
        media_type="text/event-stream"
    )