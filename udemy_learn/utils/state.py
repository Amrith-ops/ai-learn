"""State and transitions for a multi-step agent run."""

from dataclasses import dataclass, field


@dataclass
class AgentState:
    """Track one request and the actions and observations it accumulates."""

    user_input: str
    current_step: int = 0
    max_steps: int = 5
    actions: list[str] = field(default_factory=list)
    observations: list[dict[str, str | int]] = field(default_factory=list)
    tasks: list[dict[str, str]] = field(default_factory=list)
    current_task: str = ""
    results: list[str] = field(default_factory=list)
    finished: bool = False
    final_answer: str | None = None

    def __post_init__(self) -> None:
        if not self.user_input.strip():
            raise ValueError("user_input must not be empty.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be greater than zero.")
        if not 0 <= self.current_step <= self.max_steps:
            raise ValueError("current_step must be between zero and max_steps.")
        print(f"Agent state initialized for request: {self.user_input}")


def create_agent_state(user_input: str, max_steps: int = 5) -> AgentState:
    """Create fresh state for a user request."""
    state = AgentState(user_input=user_input, max_steps=max_steps)
    print(f"Created agent state with a maximum of {max_steps} steps.")
    return state


def update_current_step(state: AgentState) -> int:
    """Advance to the next step, or reject a finished/over-limit run."""
    if state.finished:
        raise RuntimeError("Cannot advance a finished agent run.")
    if state.current_step >= state.max_steps:
        raise RuntimeError(f"Agent exceeded the limit of {state.max_steps} steps.")
    if state.current_step and len(state.observations) < state.current_step:
        raise RuntimeError("Record the current step's observation before advancing.")

    state.current_step += 1
    print(f"Agent advanced to step {state.current_step}/{state.max_steps}.")
    return state.current_step


def update_action_taken(state: AgentState, action: str) -> None:
    """Record the action selected for the current step."""
    if state.finished:
        raise RuntimeError("Cannot record an action for a finished agent run.")
    if not action.strip():
        raise ValueError("action must not be empty.")
    if state.current_step == 0:
        raise RuntimeError("Advance the agent step before recording an action.")
    if len(state.actions) >= state.current_step:
        raise RuntimeError("An action is already recorded for the current step.")

    state.actions.append(action)
    print(f"Action recorded for step {state.current_step}: {action}")


def update_observations(state: AgentState, action:str,observation: str) -> None:
    """Record the result observed after the current step's action."""
    if state.finished:
        raise RuntimeError("Cannot record an observation for a finished agent run.")
    if not observation.strip():
        raise ValueError("observation must not be empty.")
    if state.current_step == 0 or len(state.actions) < state.current_step:
        raise RuntimeError("Record an action before its observation.")
    if len(state.observations) >= state.current_step:
        raise RuntimeError("An observation is already recorded for the current step.")

    state.observations.append(
        {
            "step": state.current_step,
            "action": action,
            "observation": observation,
        }
    )
    print(f"Observation recorded for step {state.current_step}: {observation}")

def add_tasks(state: AgentState, current_task: str):
    if current_task == "" or current_task is None:
        raise ValueError("current_task cannot be empty")
    state.tasks.append({
        "task": current_task,
        "state": "pending"
    })
    return state

def complete_task(agent_state: AgentState, current_task: str):
    for pending_marked_task in agent_state.tasks:
        if current_task == pending_marked_task["task"] and pending_marked_task["state"] == "pending":
            pending_marked_task["state"] = "completed"
    return agent_state
    
def get_pending_tasks(agent_state: AgentState) -> list[str]:
    return [task["task"] for task in agent_state.tasks if task["state"]=="pending"]

def all_tasks_completed(agent_state: AgentState) -> bool:
    return all(
        [task["state"]=="completed" for task in agent_state["tasks"] ]
    )

def add_intermediate_result(agent_state: AgentState, result: str):
    return agent_state.results.append(result)


def finish_agent_state(state: AgentState, final_answer: str) -> None:
    """Mark the run complete and store its final answer."""
    if state.finished:
        raise RuntimeError("Agent run is already finished.")
    if state.current_step and len(state.observations) < state.current_step:
        raise RuntimeError("Record the current step's observation before finishing.")
    if not final_answer.strip():
        raise ValueError("final_answer must not be empty.")

    state.final_answer = final_answer
    state.finished = True
    print(f"Agent run finished with answer: {final_answer}")
