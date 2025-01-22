from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import StreamingResponse, JSONResponse
import json
from typing import AsyncGenerator, Dict, Any
from datetime import datetime

from src.schema.models import ChatMessage, UserInput, ChatHistory
from src.agents.react_agent import research_agent

router = APIRouter(prefix="/research", tags=["research"])

def format_research_response(content: str, sources: list = None) -> Dict[str, Any]:
    """
    Format the research response to separate content from sources.
    
    Args:
        content: The raw response content
        sources: List of source URLs
    
    Returns:
        Formatted response dictionary
    """
    # Remove source citations from content if they exist
    if "\n\nSources:" in content:
        content = content.split("\n\nSources:")[0]

    if "Source" in content:
        content = content.split("\n* Source")[0]

    # Clean up any remaining URLs or citations
    content = content.replace("<", "").replace(">", "")
    
    return {
        "content": content.strip(),
        "sources": sources or []
    }

@router.post("", response_model=ChatMessage)
async def research_chat(user_input: UserInput) -> ChatMessage:
    """
    Research agent endpoint that provides enhanced responses with clean formatting.
    """
    try:
        result = await research_agent.handle_message(
            message=user_input.message,
            thread_id=user_input.thread_id,
            model=user_input.model,
            metadata=user_input.metadata
        )
        
        # Format the response
        formatted = format_research_response(
            result["response"],
            result.get("search_results", [])
        )
        
        # Prepare metadata with sources and tool usage
        metadata = {
            "thread_id": result.get("thread_id"),
            "tools_used": result.get("tools_used", []),
            "sources": result.get("search_results", []),
            "timestamp": datetime.utcnow().isoformat(),
            "model": user_input.model
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
    """Generate streaming response with clean research results."""
    try:
        current_chunk = ""
        
        async for chunk in research_agent.stream_response(
            message=user_input.message,
            thread_id=user_input.thread_id,
            model=user_input.model,
            metadata=user_input.metadata
        ):
            if isinstance(chunk, dict):
                # Handle structured updates (e.g., tool usage, sources)
                yield f"data: {json.dumps(chunk)}\n\n"
            else:
                # Accumulate content until we have a complete sentence
                current_chunk += chunk
                if any(end in chunk for end in [". ", "! ", "? ", "\n"]):
                    # Clean any source citations from the chunk
                    formatted = format_research_response(current_chunk)
                    yield f"data: {json.dumps({'type': 'token', 'content': formatted['content']})}\n\n"
                    current_chunk = ""
        
        # Send any remaining content
        if current_chunk:
            formatted = format_research_response(current_chunk)
            yield f"data: {json.dumps({'type': 'token', 'content': formatted['content']})}\n\n"
        
        # Send completion signal
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
    """Stream research agent responses with progress updates."""
    return StreamingResponse(
        _stream_generator(user_input),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked"
        }
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