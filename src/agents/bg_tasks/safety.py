from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseLLM
from langchain_groq import ChatGroq

class LlamaSafety:
    """LlamaGuard-based safety layer for input/output filtering."""
    
    def __init__(self, api_key: str):
        # Use Groq's hosted LlamaGuard model
        self.llm = ChatGroq(
            model="llama-guard-2",
            api_key=api_key,
            temperature=0
        )
        
    async def check_input(self, message: str) -> tuple[bool, str]:
        """Check if user input is safe."""
        prompt = f"""You are a content moderation AI. Check if this message is appropriate and safe:
        
        Message: {message}
        
        Respond with just TRUE if safe, or FALSE followed by reason if unsafe."""
        
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        result = response.content.strip().upper()
        
        if result.startswith("TRUE"):
            return True, ""
        else:
            reason = result.replace("FALSE", "").strip()
            return False, reason

    async def check_output(self, message: str) -> tuple[bool, str]:
        """Check if AI output is safe."""
        prompt = f"""You are a content moderation AI. Check if this AI response is appropriate and safe:
        
        Response: {message}
        
        Respond with just TRUE if safe, or FALSE followed by reason if unsafe."""
        
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        result = response.content.strip().upper()
        
        if result.startswith("TRUE"):
            return True, ""
        else:
            reason = result.replace("FALSE", "").strip()
            return False, reason

class SafetyWrapper:
    """Wrapper to add safety checks to any agent."""
    
    def __init__(self, agent: Any, safety: LlamaSafety):
        self.agent = agent
        self.safety = safety
        
    async def handle_message(
        self,
        message: str,
        thread_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle message with safety checks."""
        # Check input
        is_safe, reason = await self.safety.check_input(message)
        if not is_safe:
            return {
                "error": "Input rejected by safety filter",
                "reason": reason
            }
            
        # Process with agent
        result = await self.agent.handle_message(
            message=message,
            thread_id=thread_id,
            **kwargs
        )
        
        # Check output
        if "response" in result:
            is_safe, reason = await self.safety.check_output(result["response"])
            if not is_safe:
                return {
                    "error": "Output rejected by safety filter",
                    "reason": reason
                }
                
        return result