from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class MessageRole(str, Enum):
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"

class Message(BaseModel):
    """A chat message."""
    role: MessageRole
    content: str


class ChatMessage(BaseModel):
    """Message in a chat."""
    type: str = Field(description="Role of the message", examples=["human", "ai"])
    content: str = Field(description="Content of the message")

class UserInput(BaseModel):
    """User input for the agent."""
    message: str = Field(description="User input to the agent")
    model: Optional[str] = Field(
        default="llama-3.1-8b-instant",
        description="Model to use for generation"
    )

class ChatHistory(BaseModel):
    """Full chat history."""
    messages: List[ChatMessage]