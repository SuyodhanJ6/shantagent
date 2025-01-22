from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, FunctionMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

from src.core.llm import get_llm
from src.agents.state import StateManager
from src.core.tracer import get_tracer

class ReactAgent:
    def __init__(self):
        self.state_manager = StateManager()
        self.opik_tracer = get_tracer()
        self.tools = self._get_tools()
        self.agent = self._build_agent()
        self.checkpointer = MemorySaver()


    def _get_tools(self):
        """Initialize available tools."""
        @tool
        def search(query: str) -> str:
            """Search the web for current information."""
            # This is a placeholder implementation
            if "weather" in query.lower():
                return "The current weather is sunny with a temperature of 72°F"
            return "Search results for: " + query

        @tool
        def calculator(expression: str) -> str:
            """Perform mathematical calculations."""
            try:
                return str(eval(expression))
            except:
                return "Error performing calculation"

        return [search, calculator]

    def _build_agent(self):
        """Build the ReAct agent."""
        # Get LLM instance
        model = get_llm(streaming=False)
        
        # Create ReAct agent
        agent = create_react_agent(
            model=model,
            tools=self.tools,
        )
        
        return agent

    async def handle_message(
        self, 
        message: str,
        thread_id: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle a new message with ReAct agent."""
        if thread_id:
            # Save user message
            await self.state_manager.save_message(
                thread_id,
                {
                    "role": "human",
                    "content": message,
                    "metadata": metadata or {}
                }
            )
            
            # Get conversation history
            history = await self.state_manager.get_thread_messages(thread_id)
            messages = [
                {"role": "user" if msg["role"] == "human" else "assistant",
                 "content": msg["content"]}
                for msg in history
            ]
        else:
            messages = [{"role": "user", "content": message}]

        # Process with ReAct agenta
        final_state = self.agent.invoke(
            {"messages": messages},
            config={"configurable": {
                "thread_id": thread_id,
                "model": model,
                "metadata": metadata
            }}
        )
        
        # Extract response
        response = final_state["messages"][-1].content
        
        # Save AI response if thread exists
        if thread_id:
            await self.state_manager.save_message(
                thread_id,
                {
                    "role": "ai",
                    "content": response,
                    "metadata": metadata or {}
                }
            )

        return {
            "thread_id": thread_id,
            "response": response
        }

# Create singleton instance
react_agent = ReactAgent()