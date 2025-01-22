# src/core/tracer.py

from functools import lru_cache
import opik
from typing import Union
from opik.integrations.langchain import OpikTracer
from typing import Optional, Dict, Any
from langchain_core.callbacks.base import BaseCallbackHandler

# Configure Opik to use local instance only
opik.configure(use_local=True)

class LocalOpikTracer(BaseCallbackHandler):
    """Extended OpikTracer that implements required callback methods"""
    
    def __init__(self):
        super().__init__()
        self._tracer = OpikTracer()
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        """Called when LLM starts processing."""
        try:
            self._tracer.on_llm_start(serialized=serialized, prompts=prompts, **kwargs)
        except Exception as e:
            print(f"Warning: Failed to trace LLM start: {e}")

    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when LLM ends processing."""
        try:
            self._tracer.on_llm_end(response=response, **kwargs)
        except Exception as e:
            print(f"Warning: Failed to trace LLM end: {e}")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Called when LLM errors."""
        try:
            self._tracer.on_llm_error(error=str(error), **kwargs)
        except Exception as e:
            print(f"Warning: Failed to trace LLM error: {e}")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when chain starts."""
        try:
            self._tracer.on_chain_start(serialized=serialized, inputs=inputs, **kwargs)
        except Exception as e:
            print(f"Warning: Failed to trace chain start: {e}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when chain ends."""
        try:
            self._tracer.on_chain_end(outputs=outputs, **kwargs)
        except Exception as e:
            print(f"Warning: Failed to trace chain end: {e}")

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Called when chain errors."""
        try:
            self._tracer.on_chain_error(error=str(error), **kwargs)
        except Exception as e:
            print(f"Warning: Failed to trace chain error: {e}")

    def is_active(self) -> bool:
        """Check if tracer is active"""
        return True

@lru_cache()
def get_tracer() -> LocalOpikTracer:
    """Get a singleton instance of LocalOpikTracer"""
    return LocalOpikTracer()

def configure_tracer(tracer: LocalOpikTracer) -> LocalOpikTracer:
    """Configure the tracer"""
    return tracer