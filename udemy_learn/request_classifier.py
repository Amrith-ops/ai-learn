"""Classify requests for single-step or multi-step tool workflows."""

from typing import Literal

from google.genai import Client

from udemy_learn.assistant import generate_assistant_reply

RequestType = Literal["single-step", "multi-step"]


def classify_request(
    user_input: str,
    client: Client,
    model: str,
) -> RequestType:
    """Ask Gemini whether a request needs one tool action or a sequence."""
    if not user_input.strip():
        raise ValueError("user_input must not be empty.")

    response = generate_assistant_reply(
        client=client,
        model=model,
        messages=[{"role": "user", "content": user_input}],
        system_instruction="""Classify the user's request by the number of tool
actions it requires. Return exactly one of these strings and nothing else:
single-step
multi-step

Use single-step when the request can be completed with at most one tool call,
including requests that do not need a tool. Use multi-step when completing the
request requires multiple dependent or sequential tool calls.""",
    )
    classification = response.strip().casefold()
    if classification == "single-step":
        return "single-step"
    if classification == "multi-step":
        return "multi-step"
    raise ValueError(
        "Request classifier must return exactly 'single-step' or 'multi-step'."
    )
