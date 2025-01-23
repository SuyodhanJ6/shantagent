from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import StreamingResponse, JSONResponse
import json
from typing import AsyncGenerator, Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

from src.schema.models import ChatMessage, UserInput, ChatHistory
from src.agents.react_agent import research_agent
from src.core.llama_guard import LlamaGuard, llama_guard, SafetyAssessment

router = APIRouter(prefix="/research", tags=["research"])

async def check_message_safety(message: str, check_type: str = "human") -> Dict[str, any]:
    """
    Enhanced safety check using LlamaGuard with additional content filtering.
    
    Args:
        message: Content to check
        check_type: Type of content ("human" for input, "ai" for output)
    """
    # List of sensitive topics that require strict filtering
    sensitive_topics = [
        "porn", "pornography", "explicit content", "adult content",
        "nsfw", "xxx", "adult material", "adult entertainment"
    ]
    
    # Check for sensitive topics
    lower_msg = message.lower()
    if any(topic in lower_msg for topic in sensitive_topics):
        return {
            "is_safe": False,
            "response": "I apologize, but I cannot provide information about adult or explicit content. Please ask about something else.",
            "categories": ["adult_content"]
        }
    
    # Run LlamaGuard check
    safety_result = await llama_guard.ainvoke(
        check_type,
        [HumanMessage(content=message)]
    )
    
    if safety_result.safety_assessment == SafetyAssessment.UNSAFE:
        return {
            "is_safe": False,
            "response": "I apologize, but I cannot assist with that request as it may be inappropriate. Please ask something else.",
            "categories": safety_result.unsafe_categories
        }
    return {"is_safe": True}

def format_research_response(content: str, sources: list = None) -> Dict[str, Any]:
    """Format the research response to separate content from sources."""
    if "\n\nSources:" in content:
        content = content.split("\n\nSources:")[0]

    if "Source" in content:
        content = content.split("\n* Source")[0]

    content = content.replace("<", "").replace(">", "")
    
    return {
        "content": content.strip(),
        "sources": sources or []
    }

@router.post("")
async def research_chat(user_input: UserInput) -> ChatMessage:
    """Research agent endpoint with enhanced safety checks."""
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
        result = await research_agent.handle_message(
            message=user_input.message,
            thread_id=user_input.thread_id,
            model=user_input.model,
            metadata=user_input.metadata
        )
        
        formatted = format_research_response(
            result["response"],
            result.get("search_results", [])
        )
        
        # Check output safety
        output_safety = await check_message_safety(formatted["content"])
        if not output_safety["is_safe"]:
            return ChatMessage(
                type="ai",
                content=output_safety["response"],
                metadata={
                    "safety_blocked": True,
                    "unsafe_categories": output_safety["categories"]
                }
            )
        
        metadata = {
            "thread_id": result.get("thread_id"),
            "tools_used": result.get("tools_used", []),
            "sources": result.get("search_results", []),
            "timestamp": datetime.utcnow().isoformat(),
            "model": user_input.model,
            "safety_checked": True
        }
        
        return ChatMessage(
            type="ai",
            content=formatted["content"],
            metadata=metadata
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Error processing research request",
                "message": str(e),
                "type": type(e).__name__
            }
        )

async def _stream_generator(user_input: UserInput) -> AsyncGenerator[str, None]:
    """Generate streaming response with safety checks."""
    try:
        buffer = ""
        
        async for chunk in research_agent.stream_response(
            message=user_input.message,
            thread_id=user_input.thread_id,
            model=user_input.model,
            metadata=user_input.metadata
        ):
            if isinstance(chunk, dict):
                yield f"data: {json.dumps(chunk)}\n\n"
            else:
                buffer += chunk
                # Check safety every 100 characters
                if len(buffer) >= 100:
                    safety_result = await llama_guard.ainvoke("ai", [HumanMessage(content=buffer)])
                    if safety_result.safety_assessment == SafetyAssessment.UNSAFE:
                        yield f"data: {json.dumps({
                            'type': 'message',
                            'content': "Response contained inappropriate content and was blocked.",
                            'metadata': {
                                'safety_blocked': True,
                                'unsafe_categories': safety_result.unsafe_categories
                            }
                        })}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    formatted = format_research_response(buffer)
                    yield f"data: {json.dumps({'type': 'token', 'content': formatted['content']})}\n\n"
                    buffer = ""
        
        if buffer:
            safety_result = await llama_guard.ainvoke("ai", [HumanMessage(content=buffer)])
            if safety_result.safety_assessment == SafetyAssessment.UNSAFE:
                yield f"data: {json.dumps({
                    'type': 'message',
                    'content': "Response contained inappropriate content and was blocked.",
                    'metadata': {
                        'safety_blocked': True,
                        'unsafe_categories': safety_result.unsafe_categories
                    }
                })}\n\n"
            else:
                formatted = format_research_response(buffer)
                yield f"data: {json.dumps({'type': 'token', 'content': formatted['content']})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        error_data = {
            "type": "error",
            "content": str(e),
            "error_type": type(e).__name__
        }
        yield f"data: {json.dumps(error_data)}\n\n"

@router.post("/stream")
async def stream_research_chat(user_input: UserInput) -> StreamingResponse:
    """Stream research responses with safety checks."""
    # Initial safety check
    safety_result = await llama_guard.ainvoke(
        "human", 
        [HumanMessage(content=user_input.message)]
    )
    
    if safety_result.safety_assessment == SafetyAssessment.UNSAFE:
        return StreamingResponse(
            iter([f"data: {json.dumps({
                'type': 'message',
                'content': "I apologize, but I cannot provide information about that topic as it may be inappropriate. Please ask something else.",
                'metadata': {
                    'safety_blocked': True,
                    'unsafe_categories': safety_result.unsafe_categories
                }
            })}\n\ndata: [DONE]\n\n"]),
            media_type="text/event-stream"
        )
    
    return StreamingResponse(
        _stream_generator(user_input),
        media_type="text/event-stream"
    )



def parse_timestamp(timestamp) -> str:
    """Convert timestamp to ISO format string."""
    if isinstance(timestamp, datetime):
        return timestamp.isoformat()
    elif isinstance(timestamp, str):
        try:
            return datetime.fromisoformat(timestamp).isoformat()
        except ValueError:
            try:
                # Try parsing as UTC timestamp
                return datetime.utcfromtimestamp(float(timestamp)).isoformat()
            except:
                return datetime.utcnow().isoformat()
    return datetime.utcnow().isoformat()

@router.get("/history/{thread_id}")
async def get_research_history(thread_id: str) -> ChatHistory:
    """Get research chat history with sources and tool usage."""
    try:
        messages = await research_agent.state_manager.get_thread_messages(thread_id)
        if not messages:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No messages found for thread {thread_id}"
            )
        
        formatted_messages = []
        for msg in messages:
            # Parse metadata
            metadata = msg.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            
            # Create ChatMessage with properly formatted timestamp
            formatted_messages.append(
                ChatMessage(
                    type=msg["role"],
                    content=msg["content"],
                    metadata={
                        **metadata,
                        "timestamp": parse_timestamp(msg.get("created_at")),
                        "message_id": msg.get("id"),
                    }
                )
            )
        
        return ChatHistory(
            messages=formatted_messages,
            thread_id=thread_id,
            metadata={
                "last_updated": datetime.utcnow().isoformat(),
                "message_count": len(messages),
                "has_ai_response": any(msg["role"] == "ai" for msg in messages)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving research history: {str(e)}"
        )


@router.get("/status/{thread_id}")
async def get_research_status(thread_id: str) -> Dict[str, Any]:
    """Get status of ongoing research for a thread."""
    try:
        messages = await research_agent.state_manager.get_thread_messages(thread_id)
        last_message = messages[-1] if messages else None
        
        return {
            "thread_id": thread_id,
            "status": "completed" if last_message and last_message["role"] == "ai" else "in_progress",
            "last_update": last_message["created_at"].isoformat() if last_message else None,
            "message_count": len(messages),
            "tool_usage": [
                msg.get("metadata", {}).get("tools_used", [])
                for msg in messages
                if msg["role"] == "ai"
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread status not found: {str(e)}"
        )

@router.delete("/history/{thread_id}")
async def delete_research_history(thread_id: str) -> Dict[str, str]:
    """Delete research history for a specific thread."""
    try:
        await research_agent.state_manager.delete_thread(thread_id)
        return {
            "status": "success",
            "message": f"Research history {thread_id} deleted successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting research history: {str(e)}"
        )