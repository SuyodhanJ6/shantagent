# from typing import Dict, Any, Optional, List
# from langchain_core.messages import AIMessage, HumanMessage, FunctionMessage
# from langchain_core.runnables import RunnableConfig
# from langgraph.graph import END, MessagesState, StateGraph
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.tools import tool
# from typing import AsyncGenerator, Optional


# from src.core.llm import get_llm
# from src.agents.state import StateManager

# class AgentState(MessagesState, total=False):
#     """Agent state using MessagesState."""
#     thread_id: str
#     metadata: Dict[str, Any]
#     search_results: List[str]  # Store search results
#     tools_used: List[str]  # Track tools used

# class ResearchAgent:
#     def __init__(self):
#         self.state_manager = StateManager()
#         self.memory = MemorySaver()
#         self.tools = self._get_tools()
#         self.agent = self._build_agent()

#     def _get_tools(self) -> List[Any]:
#         """Initialize research and utility tools."""
#         tools = [
#             TavilySearchResults(max_results=2),
#         ]

#         @tool
#         def summarize_findings(text: str) -> str:
#             """Summarize research findings in a concise format."""
#             model = get_llm()
#             summary_prompt = f"Please summarize these findings concisely:\n{text}"
#             summary = model.invoke([HumanMessage(content=summary_prompt)])
#             return summary.content

#         tools.append(summarize_findings)
#         return tools

#     def _build_agent(self) -> StateGraph:
#         """Build the research agent workflow."""
#         workflow = StateGraph(AgentState)
        
#         # Add the research node
#         workflow.add_node("researcher", self._research_step)
#         workflow.add_node("synthesizer", self._synthesize_results)
        
#         # Set the entry point
#         workflow.set_entry_point("researcher")
        
#         # Add edges
#         workflow.add_edge("researcher", "synthesizer")
#         workflow.add_edge("synthesizer", END)

#         return workflow.compile()

#     async def _research_step(self, state: AgentState, config: RunnableConfig) -> AgentState:
#         """Perform research using search tools."""
#         model = get_llm(config["configurable"].get("model"))
#         messages = state["messages"]
#         last_message = messages[-1].content if messages else ""

#         # Perform search based on the query
#         search_tool = TavilySearchResults(max_results=2)
#         search_results = await search_tool.ainvoke(last_message)
        
#         # Format and store search results
#         formatted_results = "\n".join([
#             f"Source {i+1}: {result['content']}\nURL: {result['url']}"
#             for i, result in enumerate(search_results)
#         ])
        
#         # Add search results to state
#         state["search_results"] = search_results
#         state["tools_used"] = ["tavily_search"]

#         # Add results to messages
#         messages.append(FunctionMessage(
#             content=formatted_results,
#             name="search"
#         ))
        
#         return state

#     async def _synthesize_results(self, state: AgentState, config: RunnableConfig) -> AgentState:
#         """Synthesize research results into a coherent response."""
#         model = get_llm(config["configurable"].get("model"))
#         messages = state["messages"]
#         search_results = state.get("search_results", [])

#         # Create synthesis prompt
#         synthesis_prompt = [
#             HumanMessage(content=(
#                 "Based on the search results provided, please synthesize a comprehensive "
#                 "and informative response. Include relevant facts and cite sources when appropriate."
#             ))
#         ]
#         synthesis_prompt.extend(messages)

#         # Generate response
#         response = await model.ainvoke(synthesis_prompt)
        
#         # Save to state manager if thread_id is provided
#         thread_id = config["configurable"].get("thread_id")
#         if thread_id:
#             await self.state_manager.save_message(
#                 thread_id,
#                 {
#                     "role": "ai",
#                     "content": response.content,
#                     "metadata": {
#                         **(config["configurable"].get("metadata", {})),
#                         "tools_used": state.get("tools_used", []),
#                         "sources": [r.get("url") for r in search_results if r.get("url")]
#                     }
#                 }
#             )
        
#         messages.append(response)
#         return state

#     async def handle_message(
#         self, 
#         message: str,
#         thread_id: Optional[str] = None,
#         model: Optional[str] = None,
#         metadata: Optional[Dict[str, Any]] = None
#     ) -> Dict[str, Any]:
#         """Handle a new message with research capabilities."""
#         if thread_id:
#             # Save user message
#             await self.state_manager.save_message(
#                 thread_id,
#                 {
#                     "role": "human",
#                     "content": message,
#                     "metadata": metadata or {}
#                 }
#             )
            
#             # Get conversation history
#             history = await self.state_manager.get_thread_messages(thread_id)
#             messages = [
#                 HumanMessage(content=msg["content"]) if msg["role"] == "human"
#                 else AIMessage(content=msg["content"]) if msg["role"] == "ai"
#                 else FunctionMessage(content=msg["content"], name=msg.get("name", "function"))
#                 for msg in history
#             ]
#         else:
#             messages = [HumanMessage(content=message)]

#         # Process with agent
#         try:
#             result = await self.agent.ainvoke(
#                 {
#                     "messages": messages,
#                     "search_results": [],
#                     "tools_used": []
#                 },
#                 config=RunnableConfig(
#                     configurable={
#                         "thread_id": thread_id,
#                         "model": model,
#                         "metadata": metadata
#                     }
#                 )
#             )
            
#             # Extract final response
#             final_response = result["messages"][-1].content
            
#             return {
#                 "thread_id": thread_id,
#                 "response": final_response,
#                 "tools_used": result.get("tools_used", []),
#                 "search_results": result.get("search_results", [])
#             }
#         except Exception as e:
#             error_response = f"Error during research: {str(e)}"
#             if thread_id:
#                 await self.state_manager.save_message(
#                     thread_id,
#                     {
#                         "role": "ai",
#                         "content": error_response,
#                         "metadata": metadata or {}
#                     }
#                 )
#             return {
#                 "thread_id": thread_id,
#                 "response": error_response,
#                 "tools_used": [],
#                 "search_results": []
#             }

#     async def stream_response(
#         self,
#         message: str,
#         thread_id: Optional[str] = None,
#         model: Optional[str] = None,
#         metadata: Optional[Dict[str, Any]] = None
#     ) -> AsyncGenerator[str, None]:
#         """Stream the agent's response."""
#         try:
#             result = await self.handle_message(message, thread_id, model, metadata)
#             response = result["response"]
#             chunk_size = 100
            
#             # Stream the response in chunks
#             for i in range(0, len(response), chunk_size):
#                 yield response[i:i + chunk_size]
            
#             # If there were search results, send source information
#             if result.get("search_results"):
#                 sources = "\n\nSources:\n" + "\n".join([
#                     f"- {r.get('url')}"
#                     for r in result["search_results"]
#                     if r.get("url")
#                 ])
#                 yield sources
                
#         except Exception as e:
#             yield f"Error streaming response: {str(e)}"

# # Create singleton instance
# research_agent = ResearchAgent()


from typing import Dict, Any, Optional, List, AsyncGenerator
from langchain_core.messages import AIMessage, HumanMessage, FunctionMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import opik
from opik.integrations.langchain import OpikTracer

from src.core.llm import get_llm
from src.agents.state import StateManager
from src.core.settings import settings

# Simple Opik configuration for local development
opik.configure(use_local=True)

class AgentState(MessagesState, total=False):
    """Agent state using MessagesState."""
    thread_id: str
    metadata: Dict[str, Any]
    search_results: List[str]  # Store search results
    tools_used: List[str]  # Track tools used

class ResearchAgent:
    def __init__(self):
        self.state_manager = StateManager()
        self.memory = MemorySaver()
        self.tools = self._get_tools()
        self.agent = self._build_agent()
        self.opik_enabled = not settings.OPIK_TRACK_DISABLE

    def _get_tools(self) -> List[Any]:
        """Initialize research and utility tools."""
        tools = [
            TavilySearchResults(max_results=2),
        ]

        @tool
        def summarize_findings(text: str) -> str:
            """Summarize research findings in a concise format."""
            model = get_llm()
            summary_prompt = f"Please summarize these findings concisely:\n{text}"
            summary = model.invoke([HumanMessage(content=summary_prompt)])
            return summary.content

        tools.append(summarize_findings)
        return tools

    def _build_agent(self) -> StateGraph:
        """Build the research agent workflow."""
        workflow = StateGraph(AgentState)
        
        # Add the research node
        workflow.add_node("researcher", self._research_step)
        workflow.add_node("synthesizer", self._synthesize_results)
        
        # Set the entry point
        workflow.set_entry_point("researcher")
        
        # Add edges
        workflow.add_edge("researcher", "synthesizer")
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    async def _research_step(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Perform research using search tools."""
        model = get_llm(config["configurable"].get("model"))
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""

        # Perform search based on the query
        search_tool = TavilySearchResults(max_results=2)
        search_results = await search_tool.ainvoke(last_message)
        
        # Format and store search results
        formatted_results = "\n".join([
            f"Source {i+1}: {result['content']}\nURL: {result['url']}"
            for i, result in enumerate(search_results)
        ])
        
        # Add search results to state
        state["search_results"] = search_results
        state["tools_used"] = ["tavily_search"]

        # Add results to messages
        messages.append(FunctionMessage(
            content=formatted_results,
            name="search"
        ))
        
        return state

    async def _synthesize_results(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Synthesize research results into a coherent response."""
        model = get_llm(config["configurable"].get("model"))
        messages = state["messages"]
        search_results = state.get("search_results", [])

        # Create synthesis prompt
        synthesis_prompt = [
            HumanMessage(content=(
                "Based on the search results provided, please synthesize a comprehensive "
                "and informative response. Include relevant facts and cite sources when appropriate."
            ))
        ]
        synthesis_prompt.extend(messages)

        # Generate response
        response = await model.ainvoke(synthesis_prompt)
        
        # Save to state manager if thread_id is provided
        thread_id = config["configurable"].get("thread_id")
        if thread_id:
            await self.state_manager.save_message(
                thread_id,
                {
                    "role": "ai",
                    "content": response.content,
                    "metadata": {
                        **(config["configurable"].get("metadata", {})),
                        "tools_used": state.get("tools_used", []),
                        "sources": [r.get("url") for r in search_results if r.get("url")]
                    }
                }
            )
        
        messages.append(response)
        return state

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

        # Initialize Opik tracer if enabled
        callbacks = []
        if self.opik_enabled:
            opik_tracer = OpikTracer(graph=self.agent.get_graph(xray=True))
            callbacks.append(opik_tracer)

        # Process with agent
        try:
            result = await self.agent.ainvoke(
                {
                    "messages": messages,
                    "search_results": [],
                    "tools_used": []
                },
                config=RunnableConfig(
                    configurable={
                        "thread_id": thread_id,
                        "model": model,
                        "metadata": metadata
                    },
                    callbacks=callbacks
                )
            )
            
            # Extract final response
            final_response = result["messages"][-1].content
            
            return {
                "thread_id": thread_id,
                "response": final_response,
                "tools_used": result.get("tools_used", []),
                "search_results": result.get("search_results", [])
            }
        except Exception as e:
            error_response = f"Error during research: {str(e)}"
            if thread_id:
                await self.state_manager.save_message(
                    thread_id,
                    {
                        "role": "ai",
                        "content": error_response,
                        "metadata": metadata or {}
                    }
                )
            return {
                "thread_id": thread_id,
                "response": error_response,
                "tools_used": [],
                "search_results": []
            }

    async def stream_response(
        self,
        message: str,
        thread_id: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream the agent's response."""
        try:
            # Initialize Opik tracer for streaming if enabled
            callbacks = []
            if self.opik_enabled:
                opik_tracer = OpikTracer(graph=self.agent.get_graph(xray=True))
                callbacks.append(opik_tracer)

            result = await self.handle_message(
                message, 
                thread_id, 
                model, 
                metadata
            )
            response = result["response"]
            chunk_size = 100
            
            # Stream the response in chunks
            for i in range(0, len(response), chunk_size):
                yield response[i:i + chunk_size]
            
            # If there were search results, send source information
            if result.get("search_results"):
                sources = "\n\nSources:\n" + "\n".join([
                    f"- {r.get('url')}"
                    for r in result["search_results"]
                    if r.get("url")
                ])
                yield sources
                
        except Exception as e:
            yield f"Error streaming response: {str(e)}"

# Create singleton instance
research_agent = ResearchAgent()