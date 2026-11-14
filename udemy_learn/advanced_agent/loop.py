import json

from google.genai import Client

from udemy_learn.advanced_agent.planner import planner_agent
from udemy_learn.advanced_agent.task_decomposition import (
    fetch_task_list_from_td_agent,
)
from udemy_learn.advanced_agent.tool_executor import (
    ToolExecutionError,
    execute_and_parse_tool_response,
    tool_discovery,
)
from udemy_learn.mcp_client.client import connect
from udemy_learn.utils.state import (
    AgentState,
    create_agent_state,
    get_pending_tasks,
    update_action_taken,
    update_current_step,
    update_observations,
)


async def run_advanced_agent_loop(
    user_input: str,
    client: Client,
    model: str,
) -> AgentState:
    """Plan and execute tool actions until every task is completed or failed."""
    # fetch task list from task decomposition agent
    task_list = await fetch_task_list_from_td_agent(client, model, user_input)
    agent_state = create_agent_state(user_input, max_steps=5)
    agent_state.tasks.extend(task_list)

    async with connect() as mcp_session:
        tool_list = await tool_discovery(mcp_session)
        while get_pending_tasks(agent_state):
            """based on pending task, observations, actions input given planner agent gives 
            which tool or action it thinks is suited to solve task
            """ 
            decision = await planner_agent(
                client,
                model,
                tool_list,
                agent_state,
            )
            task = decision["task"]
            agent_state.current_task = task
            """
            Planner agent based on the evidence from observations, it 
            updates whether the given task is completed or not
            So planner agent is using observations to decide whether a task is
            completed or not.
            Planner agent is also using tasks sections to decide which next
            only one pending task it need to respond and what tool is appropriate
            for it
            """
            if decision["task_state"] in {"completed", "failed"}:
                for task_item in agent_state.tasks:
                    if (
                        task_item["task"] == task
                        and task_item["state"] == "pending"
                    ):
                        task_item["state"] = decision["task_state"]
                        break
                agent_state.results.append(
                    f"{decision['task_state']}: {task} - {decision['reason']}"
                )
                continue

            tool_name = decision["tool_name"]
            if tool_name is None:
                raise RuntimeError("Planner selected a pending task without a tool.")

            arguments = decision["arguments"]
            action = f"{tool_name}({json.dumps(arguments, sort_keys=True)})"
            update_current_step(agent_state)
            update_action_taken(agent_state, action)

            try:
                observation = await execute_and_parse_tool_response(
                    mcp_session,
                    tool_name,
                    arguments,
                )
            except ToolExecutionError as exc:
                if exc.retries_exhausted:
                    observation = (
                        f"TOOL_ATTEMPTS_EXHAUSTED: {tool_name!r} failed after "
                        f"{exc.attempts} attempts; do not retry this call again. "
                        f"Last error: {exc}"
                    )
                else:
                    observation = (
                        f"TOOL_FAILED_NON_RETRYABLE: {tool_name!r} failed on "
                        f"attempt {exc.attempts}; do not retry this call again. "
                        f"Error: {exc}"
                    )
            except RuntimeError as exc:
                observation = (
                    f"TOOL_FAILED_NON_RETRYABLE: {tool_name!r} failed on "
                    f"attempt 1; do not retry this call again. Error: {exc}"
                )

            update_observations(agent_state, action, observation)
            agent_state.results.append(observation)

    return agent_state
