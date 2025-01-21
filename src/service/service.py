from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from uuid import uuid4
import json

from src.core.settings import settings
from src.schema.models import ChatMessage, UserInput
from src.agents.chatbot import chatbot

app = FastAPI(title="Groq Chat Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/chat")
async def chat(user_input: UserInput) -> ChatMessage:
    """
    Chat endpoint that returns a final response.
    """
    try:
        # Generate a new thread_id if not provided
        thread_id = str(uuid4())
        
        kwargs = {
            "input": {"messages": [HumanMessage(content=user_input.message)]},
            "config": RunnableConfig(
                configurable={
                    "thread_id": thread_id,
                    "model": user_input.model
                }
            )
        }
        
        response = await chatbot.ainvoke(**kwargs)
        final_message = response["messages"][-1]
        
        return ChatMessage(
            type="ai",
            content=final_message.content
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}