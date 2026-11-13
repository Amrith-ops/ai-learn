"""Use Gemini to choose a tool discovered from the MCP server."""

import asyncio
from dataclasses import dataclass
from google.genai import Client
from udemy_learn.assistant import generate_assistant_reply
from udemy_learn.mcp_client.client import connect, discover_tools, execute_tool
from udemy_learn.utils.list_of_tools_prompt import build_tool_descriptions
from udemy_learn.utils.tool_selection_parser import parse_tool_selection, format_tool_result


@dataclass(frozen=True)
class ToolExecution:
    """A selected MCP tool and its text result."""

    tool_name: str
    content: str


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
