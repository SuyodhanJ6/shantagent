from typing import Dict, Any, Optional, List, Tuple
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.callbacks import CallbackManager
from src.agents.state import StateManager
from src.core.tracer import get_tracer
from src.core.llm import get_llm

def freeze_messages(messages: List[BaseMessage]) -> Tuple[BaseMessage, ...]:
    return tuple(messages)

def unfreeze_messages(messages: Tuple[BaseMessage, ...]) -> List[BaseMessage]:
    return list(messages)

class AgentState(MessagesState, total=False):
    """Agent state using MessagesState."""
    thread_id: str
    metadata: Dict[str, Any]

class ChatAgent:
    def __init__(self):
        self.state_manager = StateManager()
        self.tracer = get_tracer()
        self.callback_manager = CallbackManager([self.tracer]) if self.tracer else None
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
            model = get_llm(
                config["configurable"].get("model"),
                callbacks=self.callback_manager.handlers if self.callback_manager else None
            )
            
            messages = unfreeze_messages(tuple(state["messages"]))
            response = await model.ainvoke(
                messages,
                config={"callbacks": self.callback_manager.handlers if self.callback_manager else None}
            )
            
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
            
            return {"messages": freeze_messages([response])}
            
        except Exception as e:
            if self.tracer:
                self.tracer.on_llm_error(error=e)
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
            messages: List[BaseMessage] = []
            
            if thread_id:
                await self.state_manager.save_message(
                    thread_id,
                    {
                        "role": "human",
                        "content": message,
                        "metadata": metadata or {}
                    }
                )
                
                history = await self.state_manager.get_thread_messages(thread_id)
                messages = [
                    HumanMessage(content=msg["content"]) if msg["role"] == "human"
                    else AIMessage(content=msg["content"])
                    for msg in history
                ]
            else:
                messages = [HumanMessage(content=message)]
                
            response = await self.agent.ainvoke(
                {
                    "messages": freeze_messages(messages),
                },
                config=RunnableConfig(
                    configurable={
                        "thread_id": thread_id,
                        "model": model,
                        "metadata": metadata
                    },
                    callbacks=self.callback_manager.handlers if self.callback_manager else None
                )
            )
            
            return {
                "thread_id": thread_id,
                "response": response["messages"][-1].content
            }
        except Exception as e:
            if self.tracer:
                self.tracer.on_chain_error(error=e)
            raise

try:
    chat_agent = ChatAgent()
except Exception as e:
    print(f"Error creating ChatAgent: {e}")
    raise