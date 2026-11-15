import asyncio
import json

from google.genai import Client

from udemy_learn.advanced_agent.answer_agent import answer_agent
from udemy_learn.advanced_agent.planner import planner_agent
from udemy_learn.advanced_agent.task_decomposition_agent import (
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
    finish_agent_state,
    get_pending_tasks,
    update_action_taken,
    update_current_step,
    update_observations,
    add_memory_from_db,
)
from udemy_learn.sql_persistence.sqllitedb import pick_high_similar_records,save_new_responses  

TASK_SEPARATOR = "=" * 72


async def run_advanced_agent_loop(
    user_input: str,
    client: Client,
    model: str,
) -> AgentState:
    """Answer directly when possible; otherwise run the task and tool workflow."""
    agent_state = create_agent_state(user_input, max_steps=5)
    fetched_records_db = await asyncio.to_thread(
        pick_high_similar_records,
        user_input,
    )
    add_memory_from_db(fetched_records_db, agent_state)

    decomposition = await fetch_task_list_from_td_agent(
        client,
        model,
        user_input,
        agent_state,
    )
    if decomposition["decision"] == "Finish":
        finish_agent_state(agent_state, decomposition["answer"])
        print(
            f"\n{TASK_SEPARATOR}\n"
            "REQUEST ANSWERED WITHOUT TASKS OR TOOL CALLS\n"
            f"{TASK_SEPARATOR}"
        )
        await save_new_responses(user_input,decomposition["answer"])
        return agent_state

    task_list = decomposition["tasks"]

    agent_state.tasks.extend(task_list)
    print(f"\n{TASK_SEPARATOR}\nTASKS CREATED")
    if task_list:
        for index, task in enumerate(task_list, start=1):
            print(f"{index}. [{task['state']}] {task['task']}")
    else:
        print("No tasks were created.")
    print(TASK_SEPARATOR)

    async with connect() as mcp_session:
        tool_list = await tool_discovery(mcp_session)
        logged_task: str | None = None
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
            if task != logged_task:
                print(f"\n{TASK_SEPARATOR}\nCURRENT TASK: {task}\n{TASK_SEPARATOR}")
                logged_task = task
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
                print(
                    f"TASK {decision['task_state'].upper()}: {task}\n"
                    f"Reason: {decision['reason']}\n{TASK_SEPARATOR}"
                )
                continue

            tool_name = decision["tool_name"]
            if tool_name is None:
                raise RuntimeError("Planner selected a pending task without a tool.")

            arguments = decision["arguments"]
            action = f"{tool_name}({json.dumps(arguments, sort_keys=True)})"
            print(f"ACTION: {action}")
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

            print(f"TOOL RESPONSE for {action}:\n{observation}")
            update_observations(agent_state, action, observation)
            agent_state.results.append(observation)

    final_answer = await answer_agent(client, model, agent_state)
    await save_new_responses(user_input,final_answer)
    finish_agent_state(agent_state, final_answer)
    print(f"\n{TASK_SEPARATOR}\nADVANCED AGENT WORKFLOW COMPLETE\n{TASK_SEPARATOR}")
    return agent_state
