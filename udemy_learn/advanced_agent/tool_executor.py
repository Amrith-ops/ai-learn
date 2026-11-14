import asyncio
from typing import Any

import anyio
import httpx
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, Tool

from udemy_learn.mcp_client.client import discover_tools, execute_tool
from udemy_learn.utils.tool_selection_parser import format_tool_result

MAX_TOOL_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 0.25
_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.TransportError,
    httpx.HTTPStatusError,
    anyio.BrokenResourceError,
    anyio.ClosedResourceError,
    anyio.EndOfStream,
)


class ToolExecutionError(RuntimeError):
    """A tool call failed after applying the retry policy."""

    def __init__(
        self,
        tool_name: str,
        attempts: int,
        message: str,
        *,
        retries_exhausted: bool,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.attempts = attempts
        self.retries_exhausted = retries_exhausted


async def tool_discovery(session: ClientSession) -> list[Tool]:
    """Discover tools from an initialized MCP session."""
    return await discover_tools(session)


async def execute_tool_with_retry(
    session: ClientSession,
    tool_name: str,
    arguments: dict[str, Any],
    max_attempts: int = MAX_TOOL_ATTEMPTS,
) -> CallToolResult:
    """Call a tool up to three times for transient MCP or transport failures."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one.")

    attempt = 1
    while True:
        try:
            return await execute_tool(session, tool_name, arguments)
        except McpError as exc:
            if exc.error.code != httpx.codes.REQUEST_TIMEOUT:
                raise ToolExecutionError(
                    tool_name,
                    attempt,
                    f"MCP rejected the call: {exc}",
                    retries_exhausted=False,
                ) from exc
            error = exc
        except _RETRYABLE_TRANSPORT_ERRORS as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in {
                408,
                425,
                429,
                500,
                502,
                503,
                504,
            }:
                raise ToolExecutionError(
                    tool_name,
                    attempt,
                    f"HTTP request failed: {exc}",
                    retries_exhausted=False,
                ) from exc
            error = exc
        except RuntimeError as exc:
            raise ToolExecutionError(
                tool_name,
                attempt,
                f"Tool returned an invalid result: {exc}",
                retries_exhausted=False,
            ) from exc

        if attempt >= max_attempts:
            raise ToolExecutionError(
                tool_name,
                attempt,
                f"Automatic retry limit exhausted: {error}",
                retries_exhausted=True,
            ) from error

        await asyncio.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
        attempt += 1


def parse_tool_response(result: CallToolResult, tool_name: str) -> str:
    """Extract readable content from a successful MCP tool result."""
    try:
        return format_tool_result(result, tool_name)
    except RuntimeError as exc:
        raise ToolExecutionError(
            tool_name,
            1,
            f"Tool returned an error result: {exc}",
            retries_exhausted=False,
        ) from exc


async def execute_and_parse_tool_response(
    session: ClientSession,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """Execute a tool with retries and parse its successful result."""
    result = await execute_tool_with_retry(session, tool_name, arguments)
    return parse_tool_response(result, tool_name)