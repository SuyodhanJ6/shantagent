
from typing import Dict, Any, Optional, TypedDict, Annotated, List
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage, FunctionMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from src.core.llm import get_llm
from src.agents.state import StateManager

class AgentState(MessagesState, total=False):
    """Agent state using MessagesState."""
    thread_id: str
    metadata: Dict[str, Any]
    context: List[str]  # For storing search results

class ResearchAgent:
    def __init__(self):
        self.state_manager = StateManager()
        self.tools = self._get_tools()
        self.agent = self._build_agent()

    def _get_tools(self):
        """Initialize research tools."""
        tools = [
            TavilySearchResults(max_results=3),
        ]
        return tools

    def _build_agent(self) -> StateGraph:
        """Build the agent graph with research capabilities."""
        # Create the graph
        agent = StateGraph(AgentState)

        # Add the researcher node
        agent.add_node("researcher", self._call_research_model)
        
        # Set the entrypoint
        agent.set_entry_point("researcher")
        
        # Add edges
        agent.add_edge("researcher", END)

        return agent.compile()

    async def _call_research_model(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Call the LLM model with research capabilities."""
        model = get_llm(config["configurable"].get("model"))
        messages = state["messages"]

        # Check if we need to do research
        last_message = messages[-1].content if messages else ""
        if any(keyword in last_message.lower() for keyword in ["search", "find", "research", "look up"]):
            # Perform search
            search_tool = TavilySearchResults(max_results=3)
            search_results = await search_tool.ainvoke(last_message)
            
            # Format search results
            context = "\n".join([f"Source {i+1}: {result['content']}" for i, result in enumerate(search_results)])
            
            # Add context to messages
            messages.append(FunctionMessage(
                content=f"Here are the search results:\n{context}",
                name="search"
            ))

        # Generate response using all messages including search results
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
        """Handle a new message with research capabilities."""
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
                else AIMessage(content=msg["content"]) if msg["role"] == "ai"
                else FunctionMessage(content=msg["content"], name=msg.get("name", "function"))
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

# Create singleton instance
research_agent = ResearchAgent()