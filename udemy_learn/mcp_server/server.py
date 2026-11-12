"""Streamable HTTP MCP server for the project's utility tools."""

from mcp.server.fastmcp import FastMCP

from udemy_learn.tools.basics import (
    generate_password as make_password,
    get_current_time as read_current_time,
    roll_dice as roll,
)

mcp = FastMCP("learning-tools")


@mcp.tool()
def get_current_time() -> str:
    """Return the current local date and time."""
    return read_current_time()


@mcp.tool()
def roll_dice() -> int:
    """Roll a six-sided die and return the result."""
    return roll()


@mcp.tool()
def generate_password(length: int = 12) -> str:
    """Generate a secure password; length must be between 1 and 128."""
    if not 1 <= length <= 128:
        raise ValueError("Password length must be between 1 and 128.")
    return make_password(length)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
