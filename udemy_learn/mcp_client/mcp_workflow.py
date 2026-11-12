"""Use Gemini to choose a tool discovered from the MCP server."""

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from google.genai import Client
from mcp.types import CallToolResult, Tool

from udemy_learn.assistant import generate_assistant_reply
from udemy_learn.mcp_client.client import connect, discover_tools, execute_tool


@dataclass(frozen=True)
class ToolExecution:
    """A selected MCP tool and its text result."""

    tool_name: str
    content: str


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


async def run_mcp_workflow(
    client: Client,
    user_input: str,
    model: str,
) -> ToolExecution | None:
    """Discover remote tools, ask Gemini to select one, and call it."""
    async with connect() as mcp_session:
        tools = await discover_tools(mcp_session)
        tool_prompt_description = build_tool_descriptions(tools)
        response = await asyncio.to_thread(
            generate_assistant_reply,
            client=client,
            model=model,
            messages=[{"role": "user", "content": user_input}],
            system_instruction=f"""Choose a tool only if it is useful for the user's request.
Available tools and their input schemas:
{tool_prompt_description}

Return exactly one JSON object with this shape:
{{"tool_name": "exact_tool_name", "arguments": {{}}}}
Use the tool's input schema to provide all required arguments.
If no tool applies, return {{"tool_name": null, "arguments": {{}}}}.
Do not include markdown or any text outside the JSON object.""",
        )
        tool_name, arguments = parse_tool_selection(response, tools)
        if tool_name is None:
            return None

        result = await execute_tool(mcp_session, tool_name, arguments)
        return ToolExecution(
            tool_name=tool_name,
            content=format_tool_result(result, tool_name),
        )
