# src/service/service.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Dict, Any

# Import routers
from ..routers import chat, react, tasks

# Import middleware
from ..middleware.logging import LoggingMiddleware
from ..middleware.metrics import MetricsMiddleware, get_metrics
from ..middleware.safety import SafetyMiddleware

# Import settings and core components
from ..core.settings import settings
from ..core.llm import get_llm
from ..agents.bg_tasks.tasks import TaskManager
from ..agents.bg_tasks.agent import BackgroundAgent
from src.agents.bg_tasks.tasks import BackgroundAgent

# Create FastAPI app
app = FastAPI(
    title="Agent Service",
    description="Multi-agent service supporting chat, ReAct capabilities, and background tasks",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer(auto_error=False)

async def verify_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    """Verify auth token if configured."""
    if settings.AUTH_SECRET:
        if not credentials or credentials.credentials != settings.AUTH_SECRET.get_secret_value():
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing authentication token"
            )

# Configure middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(SafetyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize task management
task_manager = TaskManager()
bg_agent = BackgroundAgent(task_manager)

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    # Initialize task database
    await task_manager.init_db()
    # Start background task processor
    await bg_agent.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await bg_agent.stop()

# Include routers with auth
app.include_router(
    chat.router,
    prefix="/v1",
    dependencies=[Depends(verify_token)]
)
app.include_router(
    react.router,
    prefix="/v1",
    dependencies=[Depends(verify_token)]
)
app.include_router(
    tasks.router,
    prefix="/v1",
    dependencies=[Depends(verify_token)]
)

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Test LLM connection
        llm = get_llm()
        
        # Check task processor status
        task_status = "ok" if bg_agent._running else "not running"
        
        return {
            "status": "healthy",
            "components": {
                "api": "ok",
                "llm": "ok",
                "tasks": task_status,
                "database": "ok"
            },
            "version": "0.1.0"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return PlainTextResponse(
        get_metrics(),
        media_type="text/plain"
    )

@app.get("/info")
async def get_info() -> Dict[str, Any]:
    """Get service information and capabilities."""
    return {
        "version": "0.1.0",
        "models": {
            "default": settings.DEFAULT_MODEL,
            "available": [
                "mixtral-8x7b-32768",
                "llama-guard-2"
            ]
        },
        "endpoints": {
            "chat": {
                "description": "Basic chat functionality with history management",
                "streaming": True,
                "paths": [
                    "/v1/chat",
                    "/v1/chat/stream",
                    "/v1/chat/history/{thread_id}",
                    "/v1/chat/new"
                ]
            },
            "react": {
                "description": "Agent with tool usage capabilities",
                "streaming": True,
                "paths": [
                    "/v1/react",
                    "/v1/react/stream"
                ]
            },
            "tasks": {
                "description": "Background task management",
                "paths": [
                    "/v1/tasks",
                    "/v1/tasks/{task_id}",
                    "/v1/tasks/{task_id}/retry"
                ]
            }
        },
        "features": {
            "streaming": True,
            "history": True,
            "tools": True,
            "background_tasks": True,
            "safety_checks": True,
            "auth_required": bool(settings.AUTH_SECRET)
        },
        "metrics_available": True
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
            "type": type(exc).__name__
        }
    )