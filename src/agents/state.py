from typing import Dict, Any, Optional
from datetime import datetime
import sqlite3
from contextlib import asynccontextmanager
import json
import aiosqlite
from typing import List

class StateManager:
    def __init__(self, db_path: str = "chatbot.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    thread_id TEXT PRIMARY KEY,
                    last_updated TIMESTAMP,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT,
                    role TEXT,
                    content TEXT,
                    created_at TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (thread_id) REFERENCES conversations (thread_id)
                )
            """)

    @asynccontextmanager
    async def get_db(self):
        """Get async database connection."""
        async with aiosqlite.connect(self.db_path) as db:
            yield db

    async def save_message(self, thread_id: str, message: Dict[str, Any]) -> None:
        """Save a message to the database."""
        async with self.get_db() as db:
            # Update conversation last_updated
            await db.execute(
                """
                INSERT OR REPLACE INTO conversations (thread_id, last_updated, metadata)
                VALUES (?, ?, ?)
                """,
                (thread_id, datetime.utcnow(), json.dumps({}))
            )
            
            # Save message
            await db.execute(
                """
                INSERT INTO messages (id, thread_id, role, content, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message.get("id", str(datetime.utcnow().timestamp())),
                    thread_id,
                    message["role"],
                    message["content"],
                    message.get("created_at", datetime.utcnow()),
                    json.dumps(message.get("metadata", {}))
                )
            )
            await db.commit()

    async def get_thread_messages(self, thread_id: str) -> List[Dict[str, Any]]:
            """Get all messages for a thread."""
            async with self.get_db() as db:
                async with db.execute(
                    """
                    SELECT role, content, created_at, metadata
                    FROM messages
                    WHERE thread_id = ?
                    ORDER BY created_at
                    """,
                    (thread_id,)
                ) as cursor:
                    messages = await cursor.fetchall()
                    
                    return [
                        {
                            "role": msg[0],
                            "content": msg[1],
                            "created_at": msg[2],
                            "metadata": json.loads(msg[3])
                        }
                        for msg in messages
                    ]

    async def delete_thread(self, thread_id: str) -> None:
        """Delete a thread and all its messages."""
        async with self.get_db() as db:
            # Delete all messages for the thread
            await db.execute(
                """
                DELETE FROM messages
                WHERE thread_id = ?
                """,
                (thread_id,)
            )
            
            # Delete the conversation entry
            await db.execute(
                """
                DELETE FROM conversations
                WHERE thread_id = ?
                """,
                (thread_id,)
            )
            
            await db.commit()