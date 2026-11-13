"""Discover and execute MCP tools during an agent run."""

from collections.abc import Sequence
from typing import Any

from mcp import ClientSession
from mcp.types import CallToolResult, Tool

from udemy_learn.mcp_client.client import discover_tools, execute_tool
from udemy_learn.utils.tool_selection_parser import (
    format_tool_result,
    parse_tool_selection,
)


async def tool_discovery(session: ClientSession) -> list[Tool]:
    """Discover tools from an initialized MCP session."""
    return await discover_tools(session)


async def parse_planner_response_tool_res(
    response: str,
    tools: Sequence[Tool],
    session: ClientSession,
) -> tuple[str, dict[str, Any], CallToolResult]:
    """Validate a planner tool choice and execute it through the MCP session."""
    tool_name, arguments = parse_tool_selection(response, tools)
    if tool_name is None:
        raise ValueError("The planner must select a tool before it can be executed.")
    result = await execute_tool(session, tool_name, arguments)
    return tool_name, arguments, result


def format_planner_tool_result(result: CallToolResult, tool_name: str) -> str:
    """Format the result of a successful planner-selected tool call."""
    return format_tool_result(result, tool_name)
