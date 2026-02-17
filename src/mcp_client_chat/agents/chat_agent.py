"""Chat agent for interactive conversations."""

from pathlib import Path
from typing import Optional, Union
from machine_core.core.agent_base import BaseAgent
from machine_core.core.config import AgentConfig

agent_config = AgentConfig(
    max_iterations=3,
    timeout=60000,
    max_tool_retries=3,
    allow_sampling=True
)

SYSTEM_PROMPT = """You are a helpful AI assistant with access to various tools. 
<Instructions>
You use your tools to assist the use in an autonomous manner. if you want to try another way to answer the query, do it without asking for permission.
</Instructions>

<Instructions>
Do not ask the user for permission to use your tools.
</Instructions>

<Instructions>
Do not ask for the user feedback or input to answer the query. 
</Instructions>

<Instructions>
Just do whatever is needed to answer the query.
</Instructions>

<Instructions>
You like to always make things visual when you can. Especially when you are responding to the user.
</Instructions>
"""

class ChatAgent(BaseAgent):
    """Chat agent for interactive conversations.
    
    Uses streaming for real-time responses with thinking display.
    Perfect for: Streamlit UI, web chat, real-time interfaces
    """
    
    def __init__(self, model_name: Optional[str] = None, mcp_config_path: str = "mcp.json", agent_config: Optional[AgentConfig] = agent_config):
        super().__init__(
            model_name=model_name,
            system_prompt=SYSTEM_PROMPT,
            mcp_config_path=mcp_config_path,
            agent_config=agent_config
        )
    
    async def run(self, query: str, image_paths: Optional[Union[str, Path, list]] = None):
        """Run a streaming chat query.
        
        Yields streaming events for real-time UI updates.
        """
        async for event in self.run_query_stream(query, image_paths):
            yield event
