from collections.abc import Sequence,Mapping
from mcp.types import Tool, CallToolResult
import json
from typing import Any

def parse_tool_selection(
    response: str,
    tools: Sequence[Tool],
) -> tuple[str | None, dict[str, Any]]:
    """Validate the model's JSON tool selection against discovered tools."""
    try:
        selection = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("LLM tool selection must be valid JSON.") from error

    if not isinstance(selection, Mapping):
        raise ValueError("LLM tool selection must be a JSON object.")

    tool_name = selection.get("tool_name")
    arguments = selection.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("LLM tool arguments must be a JSON object.")
    if tool_name is None:
        if arguments:
            raise ValueError("Tool arguments must be empty when no tool is selected.")
        return None, {}
    if not isinstance(tool_name, str):
        raise ValueError("LLM tool_name must be a string or null.")

    available_names = {tool.name for tool in tools}
    if tool_name not in available_names:
        raise ValueError(f"LLM selected an unknown MCP tool: {tool_name!r}")

    return tool_name, arguments


def format_tool_result(result: CallToolResult, tool_name: str) -> str:
    """Return readable text from a successful MCP tool result."""
    content = "\n".join(
        text
        for item in result.content
        if (text := getattr(item, "text", None)) is not None
    )
    if result.isError:
        raise RuntimeError(f"MCP tool {tool_name!r} failed: {content}")
    if content:
        return content
    if result.structuredContent is not None:
        return json.dumps(result.structuredContent)
    return "The tool returned no text content."