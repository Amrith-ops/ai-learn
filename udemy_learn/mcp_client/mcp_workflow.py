"""Use Gemini to choose a tool discovered from the MCP server."""

import asyncio
from collections.abc import Sequence

from google.genai import Client
from mcp.types import CallToolResult
from udemy_learn.mcp_client.client import connect,discover_tools,execute_tool

from udemy_learn.assistant import generate_assistant_reply
from google.genai import Client

"""
First the python code calls the MCP client to call MCP server to get the list of tools available
Then it prepares a prompt for the LLM model to select a tool from the list of tools available
LLM model looks at the prompt and user input and select a tool from the list of tools available and returns the tool name to 
the python code

Then the python code calls the MCP client to call MCP server to execute the tool selected by the LLM model and returns the result to the user

"""

def build_tool_descriptions(tools):
    descriptions = ""
    for tool in tools:
        descriptions += f"""
                        Tool: {tool.name}
                        Description : {tool.description}
                        ----------------------------------------
                        """
    return descriptions

async def run_mcp_workflow(
    client: Client,
    user_input: str,
    model: str,
) -> CallToolResult | None:
    """Discover remote tools, ask Gemini to select one, and call it."""
    async with connect() as mcp_session:
        tools = await discover_tools(mcp_session)
        tool_prompt_description = build_tool_descriptions(tools)
        response = await asyncio.to_thread(
            generate_assistant_reply,
            client=client,
            model=model,
            messages=[{"role": "user", "content": user_input}],
            system_instruction=f"""You select a tool for the user's request.
Available tools and their input schemas:
{tool_prompt_description}

Return only the exact tool name. If no tool applies, return only None.""",
        )
        tool_name = response.strip()
        if tool_name.casefold() == "none":
            return None

        available_names = {tool.name for tool in tools}
        if tool_name not in available_names:
            raise ValueError(f"LLM selected an unknown MCP tool: {tool_name!r}")
        tool_response = await execute_tool(mcp_session, tool_name)
        return f"""
                The user asked : {user_input}
                The tool returned : {tool_response}
                 Answer the user naturally.don't add anything from your side in the answer. if tool is none, then answer it from llm model.
                """
