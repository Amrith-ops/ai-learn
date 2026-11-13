"""Run a multi-step plan against tools discovered through MCP."""

import json
from collections.abc import Mapping
from typing import Any

from google.genai import Client
from mcp.types import Tool

from udemy_learn.agent_loop.planner import fetch_next_action_recommendation_planner
from udemy_learn.agent_loop.state import (
    AgentState,
    create_agent_state,
    finish_agent_state,
    update_action_taken,
    update_current_step,
    update_observations,
)
from udemy_learn.agent_loop.tool_executor import (
    format_planner_tool_result,
    parse_planner_response_tool_res,
    tool_discovery,
)
from udemy_learn.mcp_client.client import connect
from udemy_learn.utils.tool_selection_parser import parse_tool_selection

DEFAULT_MAX_STEPS = 5


def _read_planner_decision(
    response: str,
    tools: list[Tool],
) -> tuple[str | None, dict[str, Any], str | None]:
    """Validate a planner decision and return its tool call or final answer."""
    selection = json.loads(response)
    if not isinstance(selection, Mapping):
        raise ValueError("Agent planner response must be a JSON object.")

    tool_name, arguments = parse_tool_selection(response, tools)
    if tool_name is not None:
        if selection.get("final_answer") is not None:
            raise ValueError("Planner must not return a final answer with a tool call.")
        return tool_name, arguments, None

    final_answer = selection.get("final_answer")
    if not isinstance(final_answer, str) or not final_answer.strip():
        raise ValueError("Planner must provide a final answer when it selects no tool.")
    return None, {}, final_answer


async def agent_loop_run_workflow(
    user_input: str,
    client: Client,
    model: str,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> AgentState:
    """Plan, execute, and observe tool calls until the planner returns an answer."""
    agent_state = create_agent_state(user_input, max_steps)

    async with connect() as mcp_session:
        tools = await tool_discovery(mcp_session)
        while not agent_state.finished:
            response = await fetch_next_action_recommendation_planner(
                tools,
                agent_state,
                client,
                model,
            )
            tool_name, arguments, final_answer = _read_planner_decision(response, tools)
            if tool_name is None:
                if final_answer is None:
                    raise RuntimeError("Planner stopped without a final answer.")
                finish_agent_state(agent_state, final_answer)
                break

            update_current_step(agent_state)
            action = f"{tool_name}({json.dumps(arguments, sort_keys=True)})"
            update_action_taken(agent_state, action)
            executed_name, _, result = await parse_planner_response_tool_res(
                response,
                tools,
                mcp_session,
            )
            observation = format_planner_tool_result(result, executed_name)
            update_observations(agent_state, action, observation)

    return agent_state
