import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from fastapi.responses import JSONResponse

from src.agents.bg_tasks.safety import LlamaSafety
from src.core.settings import settings

class SafetyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.safety = LlamaSafety(settings.GROQ_API_KEY.get_secret_value())
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check request content for safety."""
        # Only check POST requests with JSON content
        if request.method == "POST" and request.headers.get("content-type") == "application/json":
            try:
                # Get raw body
                body = await request.body()
                data = json.loads(body)
                
                # Check message content if present
                if "message" in data:
                    is_safe, reason = await self.safety.check_input(data["message"])
                    if not is_safe:
                        return JSONResponse(
                            status_code=400,
                            content={
                                "error": "Content safety check failed",
                                "detail": reason
                            }
                        )
                
                # Store raw body for re-reading
                await request._receive()
                
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Error processing request",
                        "detail": str(e)
                    }
                )
        
        # Process request with downstream handlers
        response = await call_next(request)
        
        # Check response content if it's JSON
        if response.headers.get("content-type") == "application/json":
            try:
                body = response.body
                data = json.loads(body)
                
                # Check response content if present
                if "response" in data:
                    is_safe, reason = await self.safety.check_output(data["response"])
                    if not is_safe:
                        return JSONResponse(
                            status_code=400,
                            content={
                                "error": "Response safety check failed",
                                "detail": reason
                            }
                        )
            except:
                pass  # Skip response checking on error
                
        return response