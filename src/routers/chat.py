from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, AsyncGenerator, List
import json
import uuid

from src.schema.models import ChatMessage, UserInput, ChatHistory
from src.agents.chatbot import chat_agent
from src.core.llm import generate_stream
from langchain_core.messages import HumanMessage, AIMessage
from src.core.safety import  SafetyAssessment
from src.core.llama_guard import LlamaGuard
from src.core.llama_guard import llama_guard, SafetyAssessment

import opik
from opik.integrations.langchain import OpikTracer

router = APIRouter(prefix="/chat", tags=["chat"])

async def check_message_safety(message: str) -> Dict[str, any]:
    """Check message safety using LlamaGuard."""
    llama_guard = LlamaGuard()
    safety_result = await llama_guard.ainvoke(
        "human", 
        [HumanMessage(content=message)]
    )
    
    if safety_result.safety_assessment == SafetyAssessment.UNSAFE:
        return {
            "is_safe": False,
            "response": safety_result.response_message,
            "categories": safety_result.unsafe_categories
        }
    return {"is_safe": True}

@router.post("")
async def chat(user_input: UserInput) -> ChatMessage:
    """Chat endpoint with enhanced safety checks."""
    try:
        # First check input safety
        safety_check = await check_message_safety(user_input.message)
        if not safety_check["is_safe"]:
            return ChatMessage(
                type="ai",
                content=safety_check["response"],
                metadata={
                    "safety_blocked": True,
                    "unsafe_categories": safety_check["categories"]
                }
            )

        # Process message if safe
        result = await chat_agent.handle_message(
            message=user_input.message,
            thread_id=user_input.thread_id,
            model=user_input.model,
            metadata=user_input.metadata
        )
        
        # Check output safety 
        output_safety = await check_message_safety(result["response"])
        if not output_safety["is_safe"]:
            return ChatMessage(
                type="ai", 
                content=output_safety["response"],
                metadata={
                    "safety_blocked": True,
                    "unsafe_categories": output_safety["categories"]
                }
            )

        return ChatMessage(
            type="ai",
            content=result["response"],
            metadata={
                "thread_id": result.get("thread_id"),
                "model": user_input.model,
                "safety_checked": True
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

async def _stream_generator(user_input: UserInput) -> AsyncGenerator[str, None]:
    try:
        # Get OpikTracer from chat agent
        callbacks = []
        if chat_agent.opik_enabled:  # Changed condition
            opik_tracer = OpikTracer(graph=chat_agent.agent.get_graph(xray=True))
            callbacks.append(opik_tracer)

        safety_result = await llama_guard.ainvoke(
            "human", 
            [HumanMessage(content=user_input.message)]
        )
        
        if safety_result.safety_assessment == SafetyAssessment.UNSAFE:
            yield f"data: {json.dumps({
                'type': 'message',
                'content': "I apologize, but I cannot provide information about that topic as it may be inappropriate.",
                'metadata': {
                    'safety_blocked': True,
                    'unsafe_categories': safety_result.unsafe_categories
                }
            })}\n\n"
            yield "data: [DONE]\n\n"
            return

        async for chunk in generate_stream(
            [HumanMessage(content=user_input.message)],
            model_name=user_input.model,
            callbacks=callbacks
        ):
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

@router.post("/stream")
async def stream_chat(user_input: UserInput) -> StreamingResponse:
    return StreamingResponse(
        _stream_generator(user_input),
        media_type="text/event-stream"
    )

@router.get("/history/{thread_id}")
async def get_chat_history(thread_id: str) -> ChatHistory:
    """Get chat history with safety metadata."""
    try:
        messages = await chat_agent.state_manager.get_thread_messages(thread_id)
        
        return ChatHistory(
            messages=[
                ChatMessage(
                    type=msg["role"],
                    content=msg["content"],
                    metadata={
                        **msg.get("metadata", {}),
                        "safety_checked": True  # Indicate message was safety checked
                    }
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
    thread_id = str(uuid.uuid4())
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