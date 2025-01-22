import asyncio
from typing import Optional
from src.agents.bg_tasks.tasks import TaskManager, TaskStatus

class BackgroundAgent:
    """Agent for processing background tasks."""
    
    def __init__(self, task_manager: TaskManager):
        self.task_manager = task_manager
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the background task processor."""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._process_tasks())
        
    async def stop(self):
        """Stop the background task processor."""
        if not self._running:
            return
            
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            
    async def _process_tasks(self):
        """Main task processing loop."""
        while self._running:
            try:
                # Get pending tasks
                tasks = await self.task_manager.list_tasks(
                    status=TaskStatus.PENDING,
                    limit=5  # Process in small batches
                )
                
                for task in tasks:
                    if not self._running:
                        break
                        
                    try:
                        # Update task status
                        task.status = TaskStatus.RUNNING
                        await self.task_manager.update_task(task)
                        
                        # Process based on task type
                        if task.type == "analysis":
                            result = await self._run_analysis(task)
                        elif task.type == "report":
                            result = await self._run_report(task)
                        else:
                            raise ValueError(f"Unknown task type: {task.type}")
                            
                        # Update with success
                        task.status = TaskStatus.COMPLETED
                        task.result = result
                        await self.task_manager.update_task(task)
                        
                    except Exception as e:
                        # Update with failure
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        await self.task_manager.update_task(task)
                        
                # Sleep if no tasks found
                if not tasks:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                print(f"Error in task processor: {e}")
                await asyncio.sleep(1)
                
    async def _run_analysis(self, task):
        """Run analysis task."""
        return {"status": "completed", "type": "analysis"}
        
    async def _run_report(self, task):
        """Run report generation task."""
        return {"status": "completed", "type": "report"}