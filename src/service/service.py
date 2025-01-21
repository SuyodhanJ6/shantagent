# src/service/service.py
from typing import AsyncGenerator
import json
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from uuid import uuid4

from src.core.settings import settings
from src.schema.models import ChatMessage, UserInput
from src.core.llm import generate_stream
from src.agents.chatbot import chat_agent

app = FastAPI(title="Production Groq Chat Service")

# Security
security = HTTPBearer(auto_error=False)

async def verify_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> None:
    """Verify auth token if configured."""
    if not settings.AUTH_SECRET:
        return

    if not credentials or credentials.credentials != settings.AUTH_SECRET.get_secret_value():
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token"
        )

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],  # Configure this appropriately for production
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(user_input: UserInput) -> ChatMessage:
    """
    Chat endpoint that returns a response without requiring authentication.
    """
    try:
        # Handle the message using the chat agent
        result = await chat_agent.handle_message(
            message=user_input.message,
            thread_id=user_input.thread_id,
            model=user_input.model,
            metadata=user_input.metadata
        )
        
        return ChatMessage(
            type="ai",
            content=result["response"],
            metadata={"thread_id": result["thread_id"]} if result["thread_id"] else {}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}"
        )

async def _stream_generator(user_input: UserInput) -> AsyncGenerator[str, None]:
    """Generate streaming response."""
    try:
        if user_input.stream:
            # Use streaming LLM
            async for chunk in generate_stream(
                [HumanMessage(content=user_input.message)],
                model_name=user_input.model
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
        else:
            # Get full response and stream it as one message
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

@app.post("/stream")
async def stream_chat(user_input: UserInput) -> StreamingResponse:
    """Stream chat responses."""
    return StreamingResponse(
        _stream_generator(user_input),
        media_type="text/event-stream"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": settings.DEFAULT_MODEL,
        "version": "0.1.0"
    }