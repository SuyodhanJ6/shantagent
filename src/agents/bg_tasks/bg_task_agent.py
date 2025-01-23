import asyncio
from typing import Dict, Any, Optional, Literal
from uuid import uuid4
from datetime import datetime

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph

from src.core.llm import get_llm
from src.agents.state import StateManager

class TaskState:
    def __init__(self, name: str) -> None:
        self.name = name
        self.id = str(uuid4())
        self.state: Literal["new", "running", "complete"] = "new"
        self.result: Optional[Literal["success", "error"]] = None
        self.data: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "id": self.id,
            "state": self.state,
            "result": self.result,
            "data": self.data,
            "timestamp": datetime.utcnow().isoformat()
        }

class BackgroundTaskAgent:
    def __init__(self):
        self.state_manager = StateManager()
        self.agent = self._build_agent()
        self.active_tasks: Dict[str, TaskState] = {}

    def _build_agent(self) -> StateGraph:
        """Build the agent graph."""
        agent = StateGraph(MessagesState)
        agent.add_node("process", self._process_tasks)
        agent.add_node("respond", self._generate_response)
        agent.set_entry_point("process")
        agent.add_edge("process", "respond")
        agent.add_edge("respond", END)
        return agent.compile()

    async def _process_tasks(self, state: MessagesState, config: RunnableConfig) -> MessagesState:
        """Process background tasks."""
        thread_id = config["configurable"].get("thread_id")
        last_message = state["messages"][-1].content
        
        # Example task creation based on message content
        if "start task" in last_message.lower():
            task = TaskState("Example Task")
            self.active_tasks[task.id] = task
            
            # Save task state
            if thread_id:
                await self.state_manager.save_message(
                    thread_id,
                    {
                        "role": "task",
                        "content": f"Task {task.name} created",
                        "metadata": task.to_dict()
                    }
                )

            # Simulate task progress
            asyncio.create_task(self._run_background_task(task, thread_id))
            
        return state

    async def _run_background_task(self, task: TaskState, thread_id: Optional[str]) -> None:
        """Run a background task."""
        try:
            # Update state to running
            task.state = "running"
            task.data = {"progress": "25%"}
            if thread_id:
                await self.state_manager.save_message(
                    thread_id,
                    {
                        "role": "task",
                        "content": f"Task {task.name} running",
                        "metadata": task.to_dict()
                    }
                )

            # Simulate work
            await asyncio.sleep(5)
            
            # Update progress
            task.data = {"progress": "75%"}
            if thread_id:
                await self.state_manager.save_message(
                    thread_id,
                    {
                        "role": "task",
                        "content": f"Task {task.name} progressing",
                        "metadata": task.to_dict()
                    }
                )

            await asyncio.sleep(5)

            # Complete task
            task.state = "complete"
            task.result = "success"
            task.data = {"progress": "100%", "result": "Task completed successfully"}
            
            if thread_id:
                await self.state_manager.save_message(
                    thread_id,
                    {
                        "role": "task",
                        "content": f"Task {task.name} completed",
                        "metadata": task.to_dict()
                    }
                )

        except Exception as e:
            task.state = "complete"
            task.result = "error"
            task.data = {"error": str(e)}
            
            if thread_id:
                await self.state_manager.save_message(
                    thread_id,
                    {
                        "role": "task",
                        "content": f"Task {task.name} failed",
                        "metadata": task.to_dict()
                    }
                )

    async def _generate_response(self, state: MessagesState, config: RunnableConfig) -> MessagesState:
        """Generate response about task status."""
        model = get_llm(config["configurable"].get("model"))
        messages = state["messages"]

        # Add task status context
        task_statuses = "\n".join([
            f"Task {task.name}: {task.state} - {task.data.get('progress', 'N/A')}"
            for task in self.active_tasks.values()
        ])
        
        context = f"\nCurrent task statuses:\n{task_statuses}"
        messages.append(AIMessage(content=context))

        response = await model.ainvoke(messages)
        
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
        """Handle a new message for background task processing."""
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
            messages = [AIMessage(content=msg["content"]) for msg in history]
        else:
            messages = [AIMessage(content=message)]
            
        response = await self.agent.ainvoke(
            {"messages": messages},
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
bg_task_agent = BackgroundTaskAgent()