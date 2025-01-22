from typing import Dict, Any, Optional
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph

from src.core.llm import get_llm
from src.agents.state import StateManager

class AgentState(MessagesState, total=False):
    """Agent state using MessagesState."""
    thread_id: str
    metadata: Dict[str, Any]

class ChatAgent:
    def __init__(self):
        self.state_manager = StateManager()
        self.agent = self._build_agent()

    def _build_agent(self) -> StateGraph:
        """Build the agent graph."""
        agent = StateGraph(AgentState)
        agent.add_node("model", self._call_model)
        agent.set_entry_point("model")
        agent.add_edge("model", END)
        return agent.compile()

    async def _call_model(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Call the LLM model."""
        try:
            print("Starting model call with state:", state)
            print("Config:", config)
            
            model = get_llm(config["configurable"].get("model"))
            messages = state["messages"]
            print("Sending messages to model:", messages)
            
            response = await model.ainvoke(messages)
            print("Raw response from model:", response)
            
            # Create new state with response
            new_state = state.copy()
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                response_content = str(response)
            print("Processed response content:", response_content)
            
            # Create AIMessage
            new_message = AIMessage(content=response_content)
            new_state["messages"] = messages + [new_message]
            
            # Save to state manager if thread_id provided
            thread_id = config["configurable"].get("thread_id")
            if thread_id:
                await self.state_manager.save_message(
                    thread_id,
                    {
                        "role": "ai",
                        "content": response_content,
                        "metadata": config["configurable"].get("metadata", {})
                    }
                )
                print(f"Saved message to thread {thread_id}")
            
            print("Returning new state:", new_state)
            return new_state
            
        except Exception as e:
            print(f"Error in _call_model: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

    async def handle_message(
        self, 
        message: str,
        thread_id: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle a new message with conversation history."""
        try:
            print(f"Processing message: {message} for thread: {thread_id}")
            
            if thread_id:
                print("Getting thread history")
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
                print("Thread history:", history)
                messages = [
                    HumanMessage(content=msg["content"]) if msg["role"] == "human"
                    else AIMessage(content=msg["content"])
                    for msg in history
                ]
            else:
                messages = [HumanMessage(content=message)]
                
            print(f"Calling agent with messages:", messages)
            
            # Process with agent
            agent_response = await self.agent.ainvoke(
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
            print("Agent response:", agent_response)
            
            # Extract response content
            if isinstance(agent_response, dict) and "messages" in agent_response:
                response = agent_response["messages"][-1].content
            else:
                response = str(agent_response)
                
            print("Final extracted response:", response)
            
            return {
                "thread_id": thread_id,
                "response": response
            }
        except Exception as e:
            print(f"Error in handle_message: {e}")
            import traceback
            print(traceback.format_exc())
            raise

# Create singleton instance
chat_agent = ChatAgent()