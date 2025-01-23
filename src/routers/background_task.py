from typing import AsyncGenerator
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.schema.models import ChatMessage, UserInput
from src.agents.bg_tasks.bg_task_agent import bg_task_agent
from src.core.safety import check_safety

router = APIRouter(prefix="/background-task", tags=["background-task"])

@router.post("")
@check_safety
async def background_task(user_input: UserInput) -> ChatMessage:
    """
    Background task agent endpoint that processes long-running tasks.
    """
    try:
        result = await bg_task_agent.handle_message(
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
            detail=f"Error processing background task request: {str(e)}"
        )

async def _background_task_stream_generator(user_input: UserInput) -> AsyncGenerator[str, None]:
    """Generate streaming response for background task agent."""
    try:
        result = await bg_task_agent.handle_message(
            message=user_input.message,
            thread_id=user_input.thread_id,
            model=user_input.model,
            metadata=user_input.metadata
        )
        
        # Stream the response in chunks
        response = result["response"]
        chunk_size = 100
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i + chunk_size]
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

@router.post("/stream")
async def stream_background_task(user_input: UserInput) -> StreamingResponse:
    """Stream background task agent responses."""
    # For streaming endpoints, we check safety before starting the stream
    input_safety = await llama_guard.ainvoke("human", [HumanMessage(content=user_input.message)])
    if input_safety.safety_assessment == SafetyAssessment.UNSAFE:
        unsafe_msg = f"Input was flagged as unsafe for following categories: {', '.join(input_safety.unsafe_categories)}"
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'message', 'content': unsafe_msg})}\n\ndata: [DONE]\n\n"]),
            media_type="text/event-stream"
        )
    
    return StreamingResponse(
        _background_task_stream_generator(user_input),
        media_type="text/event-stream"
    )