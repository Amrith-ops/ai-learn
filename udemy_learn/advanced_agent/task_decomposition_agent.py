import asyncio
import json
from typing import Literal, TypedDict

from google.genai import Client

from udemy_learn.utils.assistant import generate_assistant_json_reply
from udemy_learn.utils.state import AgentState


class DecomposedTask(TypedDict):
    task: str
    state: Literal["pending"]


class TaskDecompositionResult(TypedDict):
    decision: Literal["Finish", "CreateTasks"]
    answer: str
    tasks: list[DecomposedTask]


def task_decomposition_prompt(
    user_input: str,
    agent_state: AgentState,
) -> str:
    """Build a prompt that routes direct-answer requests or decomposes actions."""
    if not user_input.strip():
        raise ValueError("user_input must not be empty.")

    memory = json.dumps(agent_state.memory, indent=2, ensure_ascii=False)
    return f"""You are a task decomposition and request-routing agent.
Analyze the user's request and the relevant conversation records before deciding
whether to answer directly or create tasks.

Choose "Finish" when:
- The request is conversational, general knowledge, or otherwise answerable
  without external actions.
- Relevant memory records contain enough information to answer the request.
- The request does not need multiple distinct, actionable tasks.
For "Finish", answer the user directly using the request and relevant memory.
Do not invent facts, claim an external action occurred, or create tasks.

Choose "CreateTasks" only when the request requires multiple actionable
outcomes. When creating tasks:
- Include every distinct requirement from the request.
- Make each task specific and independently verifiable as complete.
- Keep tasks in a sensible execution order when one depends on another.
- Do not turn tasks into tool calls or implementation steps; the planner will
  choose actions and tools for each task.
- Do not add requirements or assumptions that the user did not state.
- Create at least two tasks that together satisfy the request.
Set every new task's state to "pending". Do not include a direct answer in this
case.

Relevant memory records fetched from the database:
{memory}

Return only one valid JSON object with exactly these keys. Do not use markdown
fences or add any text before or after the JSON object.
- For a direct answer: "decision", "answer", and "tasks", with an empty task list:
  {{"decision": "Finish", "answer": "Direct answer to the user", "tasks": []}}
- For a multi-task request: "decision", "answer", and "tasks", with an empty
  answer and at least two tasks:
  {{"decision": "CreateTasks", "answer": "", "tasks": [{{"task": "First outcome", "state": "pending"}}, {{"task": "Second outcome", "state": "pending"}}]}}

User request:
{user_input}
"""


async def fetch_task_list_from_td_agent(
    client: Client,
    model: str,
    user_input: str,
    agent_state: AgentState,
) -> TaskDecompositionResult:
    user_prompt = task_decomposition_prompt(user_input, agent_state)
    response = await asyncio.to_thread(
        generate_assistant_json_reply,
        client,
        model,
        messages=[{"role": "user", "content": user_prompt}],
        system_instruction="You route direct-answer requests or decompose actionable requests.",
    )
    return parse_task_decomposition_response(response)


def parse_task_decomposition_response(response: str) -> TaskDecompositionResult:
    """Parse and validate a direct-answer or task-decomposition response."""
    response_text = response.strip()
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as error:
        decoder = json.JSONDecoder()
        parsed = None
        for index, character in enumerate(response_text):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(response_text, index)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                break
            parsed = None

        if parsed is None:
            response_preview = " ".join(response_text.split())[:300]
            raise ValueError(
                "Task decomposition response must contain a valid JSON object. "
                f"Response preview: {response_preview!r}."
            ) from error

    required_keys = {"decision", "answer", "tasks"}
    if not isinstance(parsed, dict) or set(parsed) != required_keys:
        raise ValueError(
            "Task decomposition response must be an object with exactly "
            "'decision', 'answer', and 'tasks'."
        )

    decision = parsed["decision"]
    answer = parsed["answer"]
    tasks = parsed["tasks"]
    if not isinstance(decision, str) or decision not in {"Finish", "CreateTasks"}:
        raise ValueError("Decision must be 'Finish' or 'CreateTasks'.")
    if not isinstance(answer, str):
        raise ValueError("Task decomposition answer must be a string.")
    if not isinstance(tasks, list):
        raise ValueError("Task decomposition tasks must be a JSON array.")

    validated_tasks: list[DecomposedTask] = []
    required_task_keys = {"task", "state"}
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or set(task) != required_task_keys:
            raise ValueError(
                f"Task at index {index} must have exactly the keys 'task' and 'state'."
            )
        if not isinstance(task["task"], str) or not task["task"].strip():
            raise ValueError(f"Task at index {index} must have non-empty text.")
        if task["state"] != "pending":
            raise ValueError(f"Task at index {index} must have state 'pending'.")
        validated_tasks.append(
            {"task": task["task"], "state": "pending"}
        )

    if decision == "Finish":
        if not answer.strip():
            raise ValueError("A Finish decision must include a direct answer.")
        if validated_tasks:
            raise ValueError("A Finish decision must not include tasks.")
    else:
        if answer.strip():
            raise ValueError("A CreateTasks decision must not include an answer.")
        if len(validated_tasks) < 2:
            raise ValueError(
                "A CreateTasks decision must include at least two tasks."
            )

    return {
        "decision": decision,
        "answer": answer,
        "tasks": validated_tasks,
    }
