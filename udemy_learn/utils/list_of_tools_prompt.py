import json
from collections.abc import Sequence
from mcp.types import Tool

def build_tool_descriptions(tools: Sequence[Tool]) -> str:
    """Serialize tool names, descriptions, and argument schemas for the LLM."""
    descriptions = [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema,
        }
        for tool in tools
    ]
    return json.dumps(descriptions, indent=2)