"""Ask Gemini to choose the next action for an agent run."""

import asyncio
from collections.abc import Sequence

from google.genai import Client
from mcp.types import Tool

from udemy_learn.utils.state import AgentState
from udemy_learn.utils.assistant import generate_assistant_reply
from udemy_learn.utils.list_of_tools_prompt import build_tool_descriptions


def agent_loop_planner_prompt(
    tools: Sequence[Tool],
    agent_state: AgentState,
) -> str:
    """Build a planner prompt from the current request and accumulated state."""
    tool_descriptions = build_tool_descriptions(tools)
    return f"""You are an AI agent. Choose the next single tool action needed to
complete the user's request, based on the previous tool observations.

User request:
{agent_state.user_input}

Completed actions:
{agent_state.actions}

Previous observations:
{agent_state.observations}

Available tools and input schemas:
{tool_descriptions}

Return exactly one JSON object. To call a tool, use:
{{"tool_name":"exact_tool_name","arguments":{{}}}}
Use the tool schema to provide all required arguments. Do not repeat completed
actions unless necessary. When no more tools are needed, use:
{{"tool_name":null,"arguments":{{}},"final_answer":"answer for the user"}}
Do not include markdown or text outside the JSON object."""


async def fetch_next_action_recommendation_planner(
    tools: Sequence[Tool],
    agent_state: AgentState,
    client: Client,
    model: str,
) -> str:
    """Return Gemini's next action decision for the current agent state."""
    planner_prompt = agent_loop_planner_prompt(tools, agent_state)
    return await asyncio.to_thread(
        generate_assistant_reply,
        client=client,
        model=model,
        messages=[{"role": "user", "content": planner_prompt}],
        system_instruction="You are an AI planner. Follow the requested JSON format exactly.",
    )
