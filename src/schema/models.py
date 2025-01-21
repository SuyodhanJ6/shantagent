from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class MessageRole(str, Enum):
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    TOOL = "tool"

class Message(BaseModel):
    """Base message model."""
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatMessage(BaseModel):
    """API chat message."""
    type: str = Field(description="Role of the message", examples=["human", "ai"])
    content: str = Field(description="Content of the message")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class UserInput(BaseModel):
    """User input for the agent."""
    message: str = Field(description="User input to the agent")
    model: Optional[str] = Field(
        default="mixtral-8x7b-32768",
        description="Model to use for generation"
    )
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response")
    thread_id: Optional[str] = Field(default=None, description="Thread ID for conversation")
    metadata: Optional[Dict] = Field(default_factory=dict, description="Additional metadata")


class ChatHistory(BaseModel):
    """Full chat history."""
    messages: List[ChatMessage]
    thread_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
