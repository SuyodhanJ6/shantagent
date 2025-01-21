import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry
from prometheus_client.openmetrics.exposition import generate_latest
import psutil

# Create a custom registry
REGISTRY = CollectorRegistry()

class MetricsManager:
    """Singleton class to manage Prometheus metrics."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize_metrics()
        return cls._instance
    
    @classmethod
    def _initialize_metrics(cls):
        """Initialize Prometheus metrics."""
        # Request metrics
        cls.request_count = Counter(
            'http_request_count',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=REGISTRY
        )
        
        cls.request_latency = Histogram(
            'http_request_latency_seconds',
            'HTTP request latency in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=REGISTRY
        )
        
        cls.response_size = Histogram(
            'http_response_size_bytes',
            'HTTP response size in bytes',
            ['endpoint'],
            buckets=[100, 1000, 10000, 100000, 1000000],
            registry=REGISTRY
        )
        
        cls.error_count = Counter(
            'http_error_count',
            'Total number of HTTP errors',
            ['method', 'endpoint', 'error_type'],
            registry=REGISTRY
        )
        
        # System metrics
        cls.active_requests = Gauge(
            'http_active_requests',
            'Number of currently active HTTP requests',
            ['method'],
            registry=REGISTRY
        )
        
        cls.system_memory = Gauge(
            'system_memory_usage_bytes',
            'Current system memory usage',
            registry=REGISTRY
        )
        
        cls.system_cpu = Gauge(
            'system_cpu_usage_percent',
            'Current system CPU usage',
            registry=REGISTRY
        )
        
        # API info
        cls.api_info = Info(
            'api_info',
            'API version information',
            registry=REGISTRY
        )
        cls.api_info.info({'version': '0.1.0', 'name': 'Agent Service'})

class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.app = app
        self.metrics = MetricsManager()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Track active requests
        self.metrics.active_requests.labels(method=request.method).inc()
        
        # Start timing
        start_time = time.time()
        
        try:
            # Update system metrics
            self._update_system_metrics()
            
            # Process request
            response = await call_next(request)
            
            # Record metrics
            self._record_metrics(request, response, start_time)
            
            return response
            
        except Exception as e:
            # Record error metrics
            self.metrics.error_count.labels(
                method=request.method,
                endpoint=request.url.path,
                error_type=type(e).__name__
            ).inc()
            raise
        
        finally:
            # Decrease active requests count
            self.metrics.active_requests.labels(method=request.method).dec()
    
    def _record_metrics(self, request: Request, response: Response, start_time: float):
        """Record various metrics about the request/response."""
        duration = time.time() - start_time
        
        # Record request count
        self.metrics.request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        # Record latency
        self.metrics.request_latency.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Record response size if available
        if hasattr(response, 'body'):
            self.metrics.response_size.labels(
                endpoint=request.url.path
            ).observe(len(response.body))
    
    def _update_system_metrics(self):
        """Update system resource metrics."""
        try:
            memory = psutil.virtual_memory()
            self.metrics.system_memory.set(memory.used)
            self.metrics.system_cpu.set(psutil.cpu_percent())
        except Exception:
            pass

def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    return generate_latest(REGISTRY)