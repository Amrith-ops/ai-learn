from collections.abc import Callable

from google import genai

from udemy_learn.assistant import generate_assistant_reply
from udemy_learn.config.config import load_config_env
from udemy_learn.router.tool_router import route_tool_command

DEFAULT_PERSONALITY = (
    "You are an empathetic psychologist and life coach. "
    "Provide thoughtful and supportive guidance."
)

PERSONALITIES = {
    "1": DEFAULT_PERSONALITY,
    "2": (
        "You are an expert Python programmer and software engineer. "
        "Provide knowledgeable, patient, and clear explanations."
    ),
    "3": (
        "You are a creative chef and culinary expert. "
        "Provide practical and delicious recipes."
    ),
}


def run_conversation(
    client: genai.Client,
    model: str,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run the terminal chat while keeping history and tool routing in one place."""
    output_fn("=" * 40)
    output_fn("      My AI Assistant")
    output_fn("=" * 40)
    choice = input_fn("Choose a personality (1: Psychologist, 2: Programmer, 3: Chef): ")
    system_instruction = PERSONALITIES.get(choice, DEFAULT_PERSONALITY)

    # The conversation owns history so every Gemini turn can include prior exchanges.
    messages: list[dict[str, str]] = []
    while True:
        user_input = input_fn("\nYou : ")
        if user_input.strip().casefold() == "quit":
            output_fn("\nAI  : Goodbye! Have a great day.")
            return

        # This is intentionally manual Python routing; model-selected tool calls
        # are a later learning step, not part of this version.
        tool_response = route_tool_command(user_input)
        if tool_response is not None:
            if not tool_response.startswith("Summarize the following content:"):
                output_fn(f"\nPython  : {tool_response}")
                continue
            prompt_for_model = tool_response
        else:
            prompt_for_model = user_input

        messages.append({"role": "user", "content": prompt_for_model})
        output_fn("\nPython  : Sending the conversation to Gemini for a response...")
        ai_reply = generate_assistant_reply(
            client=client,
            model=model,
            messages=messages,
            system_instruction=system_instruction,
        )
        output_fn(f"\nAI : {ai_reply}")
        messages.append({"role": "assistant", "content": ai_reply})

        output_fn("\n---------- Conversation History ----------")
        for message in messages:
            output_fn(f"{message['role'].capitalize()} : {message['content']}")
        output_fn("------------------------------------------")


def main() -> None:
    """Load runtime configuration and start the terminal conversation."""
    settings = load_config_env()
    if not settings.api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment or .env file.")

    client = genai.Client(api_key=settings.api_key)
    run_conversation(client=client, model=settings.model)
