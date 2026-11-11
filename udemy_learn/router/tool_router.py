"""Match user messages to registered local tool handlers."""

from dataclasses import dataclass
from collections.abc import Callable

from udemy_learn.tools.basics import generate_password, get_current_time, roll_dice
from udemy_learn.tools.files import get_summarize_file_content


@dataclass(frozen=True)
class ToolRoute:
    keywords: tuple[str, ...]
    handler: Callable[[str], str]


# This route table keeps the current tool choice explicit and Python-controlled.
TOOL_ROUTES = (
    ToolRoute(
        keywords=("time", "clock"),
        handler=lambda _user_input: f"The current date and time is: {get_current_time()}",
    ),
    ToolRoute(
        keywords=("roll", "dice"),
        handler=lambda _user_input: f"You rolled a {roll_dice()}",
    ),
    ToolRoute(
        keywords=("password", "passcode"),
        handler=lambda _user_input: f"Here is your generated password: {generate_password()}",
    ),
    ToolRoute(
        keywords=("summarize", "summary"),
        handler=get_summarize_file_content,
    ),
)


def route_tool_command(user_input: str) -> str | None:
    """Run the first registered tool whose keyword appears in the input."""
    normalized_input = user_input.lower()
    for route in TOOL_ROUTES:
        if any(keyword in normalized_input for keyword in route.keywords):
            return route.handler(user_input)
    return None
