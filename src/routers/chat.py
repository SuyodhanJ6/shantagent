

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import json
from typing import AsyncGenerator, Optional, Dict, List
from uuid import uuid4

from src.schema.models import ChatMessage, UserInput, ChatHistory
from src.agents.chatbot import chat_agent
from src.core.llm import generate_stream
from langchain_core.messages import HumanMessage

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("")
async def chat(user_input: UserInput) -> ChatMessage:
    """
    Basic chat endpoint that handles message history and state management.
    """
    try:
        result = await chat_agent.handle_message(
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
                "model": user_input.model
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

async def _stream_generator(user_input: UserInput) -> AsyncGenerator[str, None]:
    """Generate streaming response for chat messages."""
    try:
        if user_input.stream:
            # Use streaming LLM
            async for chunk in generate_stream(
                [HumanMessage(content=user_input.message)],
                model_name=user_input.model
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
        else:
            # Get full response
            result = await chat_agent.handle_message(
                message=user_input.message,
                thread_id=user_input.thread_id,
                model=user_input.model,
                metadata=user_input.metadata
            )
            
            yield f"data: {json.dumps({'type': 'message', 'content': result['response']})}\n\n"
            
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

@router.post("/stream")
async def stream_chat(user_input: UserInput) -> StreamingResponse:
    """Stream chat responses."""
    return StreamingResponse(
        _stream_generator(user_input),
        media_type="text/event-stream"
    )

@router.get("/history/{thread_id}")
async def get_chat_history(thread_id: str) -> ChatHistory:
    """Get chat history for a specific thread."""
    try:
        messages = await chat_agent.state_manager.get_thread_messages(thread_id)
        
        return ChatHistory(
            messages=[
                ChatMessage(
                    type=msg["role"],
                    content=msg["content"],
                    metadata=msg["metadata"]
                )
                for msg in messages
            ],
            thread_id=thread_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Chat history not found: {str(e)}"
        )

@router.post("/new")
async def create_chat() -> Dict[str, str]:
    """Create a new chat thread."""
    thread_id = str(uuid4())
    return {"thread_id": thread_id}

@router.delete("/history/{thread_id}")
async def delete_chat_history(thread_id: str) -> Dict[str, str]:
    """Delete chat history for a specific thread."""
    try:
        # Note: You'll need to implement this in your StateManager
        await chat_agent.state_manager.delete_thread(thread_id)
        return {"status": "success", "message": f"Chat history {thread_id} deleted"}
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Error deleting chat history: {str(e)}"
        )