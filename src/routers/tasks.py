from fastapi import APIRouter, HTTPException, Query
from uuid import UUID
from typing import List, Optional
from datetime import datetime, timedelta

from src.agents.bg_tasks.tasks import TaskCreate, Task, TaskManager, TaskStatus
from src.core.settings import settings
from src.agents.bg_tasks.tasks import task_manager
router = APIRouter(prefix="/tasks", tags=["tasks"])

@router.post("", response_model=Task)
async def create_task(task: TaskCreate):
    """Create a new background task."""
    try:
        return await task_manager.create_task(task)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create task: {str(e)}"
        )

@router.get("/{task_id}", response_model=Task)
async def get_task(task_id: UUID):
    """Get task status and result."""
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    return task

@router.get("", response_model=List[Task])
async def list_tasks(
    status: Optional[TaskStatus] = None,
    since: Optional[datetime] = Query(
        default=datetime.utcnow() - timedelta(days=1),
        description="Get tasks since this time"
    ),
    limit: int = Query(default=50, le=100)
):
    """List tasks with optional filters."""
    try:
        return await task_manager.list_tasks(
            status=status,
            since=since,
            limit=limit
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving tasks: {str(e)}"
        )
        
@router.delete("/{task_id}", response_model=dict)
async def cancel_task(task_id: UUID):
    """Cancel a pending or running task."""
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
        
    if task.status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task in {task.status} state"
        )
        
    await task_manager.cancel_task(task_id)
    return {"status": "success", "message": f"Task {task_id} cancelled"}

@router.post("/{task_id}/retry", response_model=Task)
async def retry_task(task_id: UUID):
    """Retry a failed task."""
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
        
    if task.status != TaskStatus.FAILED:
        raise HTTPException(
            status_code=400,
            detail=f"Can only retry failed tasks, current status: {task.status}"
        )
        
    return await task_manager.retry_task(task_id)