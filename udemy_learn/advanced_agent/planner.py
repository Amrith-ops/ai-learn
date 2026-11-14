"""Build prompts for the task-aware advanced agent planner."""

import json
import asyncio
from collections.abc import Sequence
from typing import Any, Literal, TypedDict

from google.genai import Client
from mcp.types import Tool

from udemy_learn.utils.assistant import generate_assistant_reply
from udemy_learn.utils.list_of_tools_prompt import build_tool_descriptions
from udemy_learn.utils.state import AgentState


class PlannerAgentDecision(TypedDict):
    task: str
    tool_name: str | None
    arguments: dict[str, Any]
    task_state: Literal["pending", "completed", "failed"]
    reason: str


def advanced_agent_planner_prompt(
    tools: Sequence[Tool],
    agent_state: AgentState,
) -> str:
    """Build a planner prompt from the current task and accumulated state."""
    tool_descriptions = build_tool_descriptions(tools)
    tasks = json.dumps(agent_state.tasks, indent=2)
    actions = json.dumps(agent_state.actions, indent=2)
    observations = json.dumps(agent_state.observations, indent=2)
    results = json.dumps(agent_state.results, indent=2)

    return f"""
    You are a task-aware AI planner. Choose one pending task and the
    next single tool action that will make progress toward it.

    User request:
    {agent_state.user_input}

    Task list:
    {tasks}

    Current task:
    {agent_state.current_task or "No task selected"}

    Actions already performed:
    {actions}

    Observations from those actions:
    {observations}

    Intermediate results:
    {results}

    Available tools and input schemas:
    {tool_descriptions}

    Use the current task if it is still pending. Otherwise, choose the first
    pending task in the task list. Select only a tool listed above and provide
    arguments that satisfy its input schema. Use the observations and intermediate
    results to avoid repeating actions unless necessary. The tool executor
    automatically retries transient failures up to three total attempts. If an
    observation says the retry limit was exhausted or the failure was
    non-retryable, do not request another trial of that call; mark the task
    failed with the attempt count and error as the reason. Continue with other independent pending tasks.
    Do not mark a failed task completed or imply the overall request was fully
    completed; ensure the final answer discloses failed tasks and their reasons.

    Return only one valid JSON object with exactly these keys:
    For a tool action:
    {{"task": "exact pending task text", "tool_name": "exact_tool_name", "arguments": {{}}, "task_state": "pending", "reason": ""}}
    For an achieved task, use "tool_name": null, "task_state": "completed", and
    put the evidence in "reason":
    {{"task": "exact pending task text", "tool_name": null, "arguments": {{}}, "task_state": "completed", "reason": "evidence"}}
    For an unrecoverable task, use "tool_name": null, "task_state": "failed", and
    explain the failure in "reason":
    {{"task": "exact pending task text", "tool_name": null, "arguments": {{}}, "task_state": "failed", "reason": "failure reason"}}
    Do not select completed or failed tasks or include markdown or text outside
    the JSON object."""

async def planner_agent(
    client: Client,
    model: str,
    tools: Sequence[Tool],
    agent_state: AgentState,
) -> PlannerAgentDecision:
    user_prompt = advanced_agent_planner_prompt(tools, agent_state)
    tasks_list = await asyncio.to_thread(
        generate_assistant_reply,
        client,
        model,
        messages=[{"role": "user", "content": user_prompt}],
        system_instruction="You are task decomposition agent",
    )
    return parse_planner_agent_response(tasks_list, tools, agent_state)

def parse_planner_agent_response(
    response: str,
    tools: Sequence[Tool],
    agent_state: AgentState,
) -> PlannerAgentDecision:
    """Parse and validate a planner decision against pending tasks and tools."""
    try:
        decision = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("Planner response must be valid JSON.") from error

    required_keys = {"task", "tool_name", "arguments", "task_state", "reason"}
    if not isinstance(decision, dict) or set(decision) != required_keys:
        raise ValueError(
            "Planner response must be an object with exactly "
            "'task', 'tool_name', 'arguments', 'task_state', and 'reason'."
        )

    task = decision["task"]
    tool_name = decision["tool_name"]
    arguments = decision["arguments"]
    task_state = decision["task_state"]
    reason = decision["reason"]

    if not isinstance(task, str) or not task.strip():
        raise ValueError("Planner task must be a non-empty string.")
    if not isinstance(arguments, dict):
        raise ValueError("Planner arguments must be a JSON object.")
    if not isinstance(reason, str):
        raise ValueError("Planner reason must be a string.")
    if not isinstance(task_state, str) or task_state not in {
        "pending",
        "completed",
        "failed",
    }:
        raise ValueError(
            "Planner task_state must be 'pending', 'completed', or 'failed'."
        )
    if not any(
        isinstance(pending_task, dict)
        and pending_task.get("task") == task
        and pending_task.get("state") == "pending"
        for pending_task in agent_state.tasks
    ):
        raise ValueError(f"Planner selected a task that is not pending: {task!r}.")

    if task_state == "pending":
        if not isinstance(tool_name, str):
            raise ValueError("A pending task must select a tool.")
        if reason.strip():
            raise ValueError("A pending tool action must have an empty reason.")
        available_names = {tool.name for tool in tools}
        if tool_name not in available_names:
            raise ValueError(f"Planner selected an unknown tool: {tool_name!r}.")
    else:
        if tool_name is not None:
            raise ValueError("Completed or failed tasks must not select a tool.")
        if arguments:
            raise ValueError("Completed or failed tasks must have empty arguments.")
        if not reason.strip():
            raise ValueError(
                f"A {task_state} task must include a reason."
            )

    return {
        "task": task,
        "tool_name": tool_name,
        "arguments": arguments,
        "task_state": task_state,
        "reason": reason,
    }
