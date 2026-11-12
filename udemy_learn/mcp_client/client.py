"""Client helpers for connecting to the learning tools MCP server."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, Tool

DEFAULT_SERVER_URL = "http://127.0.0.1:8000/mcp"


@asynccontextmanager
async def connect(
    server_url: str = DEFAULT_SERVER_URL,
) -> AsyncIterator[ClientSession]:
    """Connect to the HTTP MCP server and close resources when done."""
    async with streamablehttp_client(server_url) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


async def discover_tools(session: ClientSession) -> list[Tool]:
    """Return tools advertised by an initialized MCP session."""
    response = await session.list_tools()
    return response.tools


async def execute_tool(
    session: ClientSession,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> CallToolResult:
    """Call a remote tool by name and return its MCP result."""
    if arguments is None:
        arguments = {}
    return await session.call_tool(tool_name, arguments)
