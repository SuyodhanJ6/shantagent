from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from src.core.llm import get_llm

class AgentState(MessagesState, total=False):
    """Agent state using MessagesState."""
    pass

async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Call the LLM model."""
    model = get_llm(config["configurable"].get("model"))
    messages = state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": [response]}

# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.set_entry_point("model")
agent.add_edge("model", END)

chatbot = agent.compile(checkpointer=MemorySaver())
