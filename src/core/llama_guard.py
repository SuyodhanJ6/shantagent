from enum import Enum
from typing import List, Optional, Dict, Any
from functools import wraps
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.core.settings import settings

class SafetyAssessment(Enum):
    """Enum for safety assessment results."""
    SAFE = "safe"
    UNSAFE = "unsafe"
    ERROR = "error"

class LlamaGuardOutput(BaseModel):
    """Output from Llama Guard assessment."""
    safety_assessment: SafetyAssessment = Field(description="Safety assessment result")
    unsafe_categories: List[str] = Field(
        description="List of unsafe categories if content is unsafe",
        default_factory=list
    )
    response_message: str = Field(
        description="Custom response message for unsafe content",
        default=""
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata about the safety check",
        default_factory=dict
    )

class LlamaGuard:
    """LlamaGuard safety checker implementation."""
    
    def __init__(self) -> None:
        """Initialize LlamaGuard with safety model."""
        if not settings.GROQ_API_KEY:
            print("GROQ_API_KEY not set, LlamaGuard will be disabled")
            self.model = None
            return
            
        self.model = ChatGroq(
            model="llama-guard-3-8b",
            temperature=0.0,
            api_key=settings.GROQ_API_KEY.get_secret_value(),
            tags=["llama_guard"]
        )
        self.prompt = PromptTemplate.from_template(llama_guard_instructions)


# Comprehensive unsafe content categories
unsafe_content_categories = {
    "S1": "Violence and Gore",
    "S2": "Criminal Activities and Instructions",
    "S3": "Sexual Content and Adult Themes",
    "S4": "Child Exploitation and Endangerment",
    "S5": "Hate Speech and Discrimination",
    "S6": "Self-Harm and Suicide",
    "S7": "Personal Information and Privacy",
    "S8": "Copyright and IP Violations",
    "S9": "Weapons and Explosives",
    "S10": "Extremism and Radicalization",
    "S11": "Harassment and Bullying",
    "S12": "Graphic Medical Content",
    "S13": "Misinformation and Manipulation",
    "S14": "Drug Production and Usage"
}

# Detailed prompt for LlamaGuard
llama_guard_instructions = """
Task: Strictly evaluate if the message contains unsafe or inappropriate content.

<SAFETY GUIDELINES>
1. Content Assessment Rules:
   - Flag adult/NSFW content and anatomical discussions
   - Block harmful activity instructions or illegal content
   - Identify hate speech or discriminatory language
   - Block inappropriate content for young audiences
   - Flag discussions of weapons, violence, self-harm
   - Block explicit medical/anatomical content
   - Flag extreme political/ideological content
   - Block drug-related instructions or promotion
   
2. High Priority Concerns:
   - Any adult/NSFW themes, even if academic
   - Personal information requests
   - Content involving minors
   - Instructions for harmful activities
   - Hate speech and discrimination
   - Self-harm or suicide discussion
   - Weapon creation or usage details
   - Extreme political content
</SAFETY GUIDELINES>

<BEGIN CONVERSATION CONTEXT>
{conversation_history}
</BEGIN CONVERSATION CONTEXT>

Safety assessment for the last {role} message:
First line: Write only 'safe' or 'unsafe'
Second line (if unsafe): List violated category codes (S1-S14)"""

class LlamaGuard:
    """LlamaGuard safety checker implementation."""
    
    def __init__(self) -> None:
        """Initialize LlamaGuard with safety model."""
        if not settings.GROQ_API_KEY:
            print("GROQ_API_KEY not set, LlamaGuard will be disabled")
            self.model = None
            return
            
        self.model = ChatGroq(
            model="llama-guard-3-8b",
            temperature=0.0,
            api_key=settings.GROQ_API_KEY.get_secret_value(),
            tags=["llama_guard"]
        )
        self.prompt = PromptTemplate.from_template(llama_guard_instructions)
        
    def _compile_prompt(self, role: str, messages: List[AnyMessage]) -> str:
        """Compile the safety check prompt."""
        if not messages:
            raise ValueError("Messages list cannot be empty")
        if role not in ["ai", "human"]:
            raise ValueError("Role must be either 'ai' or 'human'")
            
        role_mapping = {"ai": "Agent", "human": "User"}
        messages_str = [
            f"{role_mapping[m.type]}: {m.content}" 
            for m in messages 
            if m.type in ["ai", "human"]
        ]
        
        return self.prompt.format(
            role=role,
            conversation_history="\n\n".join(messages_str)
        )

    def _get_safety_response(self, categories: List[str]) -> str:
        """Generate appropriate response for unsafe content."""
        templates = {
            "Sexual Content and Adult Themes": "I cannot discuss adult or NSFW content. Please keep our conversation appropriate.",
            "Violence and Gore": "I cannot assist with violent or graphic content. Please ask something else.",
            "Criminal Activities": "I cannot assist with anything illegal or harmful. Please ask about legal activities only.",
            "Hate Speech and Discrimination": "I do not engage with discriminatory or hateful content. Please be respectful.",
            "Self-Harm and Suicide": "I cannot assist with self-harm related content. If you need help, please contact appropriate support services.",
            "DEFAULT": "I apologize, but I cannot provide information about that topic as it may be inappropriate or unsafe. Please ask something else."
        }
        
        # Match category to template or use default
        for category in categories:
            for key, response in templates.items():
                if key in category:
                    return response
        return templates["DEFAULT"]

    def parse_output(self, output: str) -> LlamaGuardOutput:
        """Parse LlamaGuard model output with enhanced error handling."""
        output = output.strip()
        
        if output == "safe":
            return LlamaGuardOutput(
                safety_assessment=SafetyAssessment.SAFE,
                response_message="",
                metadata={"raw_output": output}
            )
            
        parsed_output = output.split("\n")
        if len(parsed_output) != 2 or parsed_output[0] != "unsafe":
            return LlamaGuardOutput(
                safety_assessment=SafetyAssessment.ERROR,
                response_message="Safety check error: Invalid format",
                metadata={"raw_output": output, "error": "Invalid format"}
            )
            
        try:
            categories = [c.strip() for c in parsed_output[1].split(",")]
            readable_categories = [
                unsafe_content_categories[c].strip(".")
                for c in categories
            ]
            
            return LlamaGuardOutput(
                safety_assessment=SafetyAssessment.UNSAFE,
                unsafe_categories=readable_categories,
                response_message=self._get_safety_response(readable_categories),
                metadata={
                    "raw_output": output,
                    "category_codes": categories
                }
            )
        except KeyError as e:
            return LlamaGuardOutput(
                safety_assessment=SafetyAssessment.ERROR,
                response_message="Safety check error: Invalid category",
                metadata={
                    "raw_output": output,
                    "error": f"Invalid category: {str(e)}"
                }
            )

    async def ainvoke(self, role: str, messages: List[AnyMessage]) -> LlamaGuardOutput:
        """Async safety check."""
        if self.model is None:
            return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
            
        try:
            compiled_prompt = self._compile_prompt(role, messages)
            result = await self.model.ainvoke(
                [HumanMessage(content=compiled_prompt)]
            )
            return self.parse_output(result.content)
        except Exception as e:
            return LlamaGuardOutput(
                safety_assessment=SafetyAssessment.ERROR,
                response_message="Safety check error: System error",
                metadata={"error": str(e)}
            )

    def invoke(self, role: str, messages: List[AnyMessage]) -> LlamaGuardOutput:
        """Sync safety check."""
        if self.model is None:
            return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
            
        try:
            compiled_prompt = self._compile_prompt(role, messages)
            result = self.model.invoke(
                [HumanMessage(content=compiled_prompt)]
            )
            return self.parse_output(result.content)
        except Exception as e:
            return LlamaGuardOutput(
                safety_assessment=SafetyAssessment.ERROR,
                response_message="Safety check error: System error",
                metadata={"error": str(e)}
            )

# Create singleton instance
llama_guard = LlamaGuard()

def check_safety(func):
    """Decorator for adding safety checks to endpoints."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract user input
        user_input = next((arg for arg in args if hasattr(arg, 'message')), None)
        if not user_input:
            return await func(*args, **kwargs)
            
        # Check input safety
        safety_result = await llama_guard.ainvoke(
            "human", 
            [HumanMessage(content=user_input.message)]
        )
        
        if safety_result.safety_assessment == SafetyAssessment.UNSAFE:
            return {
                "type": "ai",
                "content": safety_result.response_message,
                "metadata": {
                    "safety_blocked": True,
                    "unsafe_categories": safety_result.unsafe_categories
                }
            }
            
        # Process original request if safe
        return await func(*args, **kwargs)
        
    return wrapper