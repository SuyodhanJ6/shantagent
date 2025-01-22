import json
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import UUID, uuid4
import aiosqlite
from pydantic import BaseModel

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskCreate(BaseModel):
    type: str
    params: Dict[str, Any]
    priority: int = 0
    
class Task(TaskCreate):
    id: UUID
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[float] = None

class TaskManager:
    def __init__(self, db_path: str = "chatbot.db"):
        self.db_path = db_path
        self.tasks: Dict[UUID, asyncio.Task] = {}
        
    async def init_db(self):
        """Initialize the database tables."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    params TEXT NOT NULL,
                    priority INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    result TEXT,
                    error TEXT,
                    progress REAL
                )
            """)
            await db.commit()
            
    async def create_task(self, task: TaskCreate) -> Task:
        """Create a new task."""
        task_id = uuid4()
        new_task = Task(
            id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            **task.dict()
        )
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO tasks (id, type, params, priority, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(new_task.id),
                    new_task.type,
                    json.dumps(new_task.params),
                    new_task.priority,
                    new_task.status
                )
            )
            await db.commit()
            
        return new_task
    
    async def get_task(self, task_id: UUID) -> Optional[Task]:
        """Get task by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (str(task_id),)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return Task(
                        id=UUID(row[0]),
                        type=row[1],
                        params=json.loads(row[2]),
                        priority=row[3],
                        status=TaskStatus(row[4]),
                        created_at=datetime.fromisoformat(row[5]),
                        updated_at=datetime.fromisoformat(row[6]),
                        result=json.loads(row[7]) if row[7] else None,
                        error=row[8],
                        progress=row[9]
                    )
        return None
        
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        since: Optional[datetime] = None,
        limit: int = 50
    ) -> List[Task]:
        """List tasks with optional filters."""
        query = "SELECT * FROM tasks WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status)
            
        if since:
            query += " AND created_at >= ?"
            params.append(since.isoformat())
            
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [
                    Task(
                        id=UUID(row[0]),
                        type=row[1],
                        params=json.loads(row[2]),
                        priority=row[3],
                        status=TaskStatus(row[4]),
                        created_at=datetime.fromisoformat(row[5]),
                        updated_at=datetime.fromisoformat(row[6]),
                        result=json.loads(row[7]) if row[7] else None,
                        error=row[8],
                        progress=row[9]
                    )
                    for row in rows
                ]
                
    async def update_task(self, task: Task):
        """Update task status and result."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE tasks 
                SET status = ?, updated_at = ?, result = ?, error = ?, progress = ?
                WHERE id = ?
                """,
                (
                    task.status,
                    datetime.utcnow(),
                    json.dumps(task.result) if task.result else None,
                    task.error,
                    task.progress,
                    str(task.id)
                )
            )
            await db.commit()
            
    async def cancel_task(self, task_id: UUID):
        """Cancel a task."""
        task = await self.get_task(task_id)
        if task:
            task.status = TaskStatus.CANCELLED
            await self.update_task(task)
            
            # Cancel any running asyncio task
            if task_id in self.tasks:
                self.tasks[task_id].cancel()
                del self.tasks[task_id]
                
    async def retry_task(self, task_id: UUID) -> Task:
        """Retry a failed task."""
        old_task = await self.get_task(task_id)
        if not old_task:
            raise ValueError(f"Task {task_id} not found")
            
        # Create new task with same parameters
        new_task = await self.create_task(
            TaskCreate(
                type=old_task.type,
                params=old_task.params,
                priority=old_task.priority
            )
        )
        return new_task

    async def update_progress(self, task_id: UUID, progress: float):
        """Update task progress."""
        task = await self.get_task(task_id)
        if task:
            task.progress = progress
            await self.update_task(task)


task_manager=TaskManager()