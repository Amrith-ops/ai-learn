import asyncio
import json

from udemy_learn.utils.assistant import generate_assistant_reply
from google.genai import Client


def task_decomposition_prompt(user_input: str) -> str:
    """Build a prompt that decomposes a user request into trackable tasks."""
    if not user_input.strip():
        raise ValueError("user_input must not be empty.")

    return f"""You are a task decomposition agent. Break the user's request into
                the smallest useful set of clear, outcome-focused tasks that together satisfy
                the request.

                Guidelines:
                - Include every distinct requirement from the request.
                - Make each task specific and independently verifiable as complete.
                - Keep tasks in a sensible execution order when one depends on another.
                - Do not turn tasks into tool calls or implementation steps; the planner will
                choose actions and tools for each task.
                - Do not add requirements or assumptions that the user did not state.
                - For a simple request, return one task rather than splitting it unnecessarily.

                Return only a valid JSON array. Each item must have exactly these string keys:
                "task" and "state". Set every new task's state to "pending".
                Example:
                [{{"task": "Get the weather for Paris", "state": "pending"}}]

                User request:
                {user_input}
            """


async def fetch_task_list_from_td_agent(
    client: Client, model: str, user_input: str
) -> list[dict[str, str]]:
    user_prompt = task_decomposition_prompt(user_input)
    tasks_list = await asyncio.to_thread(
        generate_assistant_reply,
        client,
        model,
        messages=[{"role": "user", "content": user_prompt}],
        system_instruction="You are task decomposition agent",
    )
    return parse_fetched_task_list(tasks_list)


def parse_fetched_task_list(task_list: str) -> list[dict[str, str]]:
    """Parse and validate the task-decomposition agent's JSON response."""
    parsed_task_list = json.loads(task_list)
    if not isinstance(parsed_task_list, list):
        raise ValueError("Task decomposition response must be a JSON array.")

    required_keys = {"task", "state"}
    tasks: list[dict[str, str]] = []
    for index, task in enumerate(parsed_task_list):
        if not isinstance(task, dict) or set(task) != required_keys:
            raise ValueError(
                f"Task at index {index} must have exactly the keys 'task' and 'state'."
            )
        if not isinstance(task["task"], str) or not isinstance(task["state"], str):
            raise ValueError(
                f"Task and state at index {index} must both be strings."
            )
        if task["state"] != "pending":
            raise ValueError(f"Task at index {index} must have state 'pending'.")

        tasks.append({"task": task["task"], "state": task["state"]})

    return tasks

