"""Execute tools selected by the LLM planner."""

from collections.abc import Callable

from udemy_learn.tools.basics import generate_password, get_current_time, roll_dice


LLM_TOOL_HANDLERS: dict[str, Callable[[], str]] = {
    "get_current_time": get_current_time,
    "roll_dice": lambda: str(roll_dice()),
    "generate_password": generate_password,
}


def execute_llm_tool(tool_name: str) -> str | None:
    """Execute a named tool, or return ``None`` when the planner selects none."""
    normalized_name = tool_name.strip()
    if normalized_name.casefold() == "none":
        return None

    handler = LLM_TOOL_HANDLERS.get(normalized_name)
    if handler is None:
        raise ValueError(f"Unknown tool selected by LLM: {normalized_name!r}")

    return handler()
