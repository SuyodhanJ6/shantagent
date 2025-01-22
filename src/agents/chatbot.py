from typing import Dict, Any, Optional
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph

from src.core.llm import get_llm
from src.agents.state import StateManager

from opik.integrations.langchain import OpikTracer

class AgentState(MessagesState, total=False):
    """Agent state using MessagesState."""
    thread_id: str
    metadata: Dict[str, Any]

class ChatAgent:
    def __init__(self, tracer: Optional[OpikTracer] = None):
        self.state_manager = StateManager()
        self.opik_tracer = tracer
        self.agent = self._build_agent()

    def _build_agent(self) -> StateGraph:
        """Build the agent graph with optional tracing."""
        agent = StateGraph(AgentState)
        agent.add_node("model", self._call_model)
        agent.set_entry_point("model")
        agent.add_edge("model", END)
        
        # If tracer is available, wrap the graph
        if self.opik_tracer:
            agent = self.opik_tracer.trace_graph(agent)
        
        return agent.compile()

    async def _call_model(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Call the LLM model."""
        model = get_llm(
            config["configurable"].get("model"),
            trace=self.opik_tracer
        )
        messages = state["messages"]
        response = await model.ainvoke(messages)
        
        # Save to state manager if thread_id is provided
        thread_id = config["configurable"].get("thread_id")
        if thread_id:
            await self.state_manager.save_message(
                thread_id,
                {
                    "role": "ai",
                    "content": response.content,
                    "metadata": config["configurable"].get("metadata", {})
                }
            )
        
        return {"messages": [response]}

    async def handle_message(
        self, 
        message: str,
        thread_id: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle a new message with conversation history."""
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
                HumanMessage(content=msg["content"]) if msg["role"] == "human"
                else AIMessage(content=msg["content"])
                for msg in history
            ]
        else:
            messages = [HumanMessage(content=message)]
            
        # Process with agent
        response = await self.agent.ainvoke(
            {
                "messages": messages,
            },
            config=RunnableConfig(
                configurable={
                    "thread_id": thread_id,
                    "model": model,
                    "metadata": metadata
                }
            )
        )
        
        return {
            "thread_id": thread_id,
            "response": response["messages"][-1].content
        }
    
opik_tracer = OpikTracer(
    project_name="agent-service",  # Your project name
    environment="production",  # or "development"
    # Optional: add more configuration
)
# Create singleton instance
chat_agent = ChatAgent(tracer=opik_tracer)
