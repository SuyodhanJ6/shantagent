from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Dict, Any

# Import routers
from src.routers import chat
from src.routers import research
# Import middleware
from src.middleware.logging import LoggingMiddleware
from src.middleware.metrics import MetricsMiddleware, get_metrics
# Import settings and core components
from src.core.settings import settings
from src.core.llm import get_llm

# Create FastAPI app
app = FastAPI(
    title="Agent Service",
    description="Multi-agent service supporting chat and ReAct capabilities",
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with auth
app.include_router(
    chat.router,
    prefix="/v1",
    dependencies=[Depends(verify_token)]
)
app.include_router(
    research.router,
    prefix="/v1",
    dependencies=[Depends(verify_token)]
)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Test LLM connection
        llm = get_llm()
        return {
            "status": "healthy",
            "components": {
                "api": "ok",
                "llm": "ok",
                "database": "ok"  # Add proper DB health check
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
                # Add other available models
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
            "research": {
                "description": "Research agent with citation support",
                "streaming": True,
                "paths": [
                    "/v1/research",
                    "/v1/research/stream"
                ]
            }
        },
        "features": {
            "streaming": True,
            "history": True,
            "tools": True,
            "citations": True,
            "auth_required": bool(settings.AUTH_SECRET)
        },
        "metrics_available": True
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    # Log the error here
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
            "type": type(exc).__name__
        }
    )