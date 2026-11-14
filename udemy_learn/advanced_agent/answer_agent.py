"""Build prompts and generate user-facing answers for advanced agent runs."""

import asyncio
import json

from google.genai import Client

from udemy_learn.utils.assistant import generate_assistant_reply
from udemy_learn.utils.state import AgentState


def advanced_agent_answer_prompt(agent_state: AgentState) -> str:
    """Build an evidence-grounded answer prompt from the completed agent state."""
    completed_tasks = [
        task for task in agent_state.tasks if task.get("state") == "completed"
    ]
    failed_tasks = [
        task for task in agent_state.tasks if task.get("state") == "failed"
    ]

    return f"""You are an answer agent. Answer the user's original request in
clear, natural language using only the supplied state and evidence.

User request:
{agent_state.user_input}

Completed tasks:
{json.dumps(completed_tasks, indent=2)}

Failed or incomplete tasks:
{json.dumps(failed_tasks, indent=2)}

Actions performed:
{json.dumps(agent_state.actions, indent=2)}

Observations:
{json.dumps(agent_state.observations, indent=2)}

Intermediate results:
{json.dumps(agent_state.results, indent=2)}

Answer the user's request directly. Report useful findings from the observations
and results, not merely a list of tool calls. Do not invent facts or claim a task
was completed without supporting evidence. If any task failed, clearly disclose
what could not be completed and the recorded reason. If evidence is insufficient,
say so. Do not output JSON or discuss internal agent implementation."""


async def answer_agent(
    client: Client,
    model: str,
    agent_state: AgentState,
) -> str:
    """Generate a natural-language answer from the final agent state."""
    prompt = advanced_agent_answer_prompt(agent_state)
    return await asyncio.to_thread(
        generate_assistant_reply,
        client,
        model,
        messages=[{"role": "user", "content": prompt}],
        system_instruction=(
            "You are a concise, evidence-grounded assistant. Answer the user's "
            "request naturally and disclose any incomplete tasks."
        ),
    )