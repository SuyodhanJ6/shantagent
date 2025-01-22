
from opik.integrations.langchain import OpikTracer
from functools import lru_cache
import os
import opik

# Configure Opik to use local instance
opik.configure(use_local=True)

@lru_cache()
def get_tracer() -> OpikTracer:
    """Get a singleton instance of OpikTracer."""
    try:
        tracer = OpikTracer(
            project_name="agent-service",
            environment="development",
        )
        return tracer
    except Exception as e:
        print(f"Error initializing OpikTracer: {e}")
        return None

# If you need to set additional configuration after initialization
def configure_tracer(tracer: OpikTracer) -> OpikTracer:
    if tracer:
        tracer.set_metadata({
            "service_name": "agent-service",
            "version": "0.1.0",
            "model": "mixtral-8x7b-32768"
        })
    return tracer