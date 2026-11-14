from google.genai import Client

from udemy_learn.utils.assistant import generate_assistant_reply


def plan_llm_tools_prompt() -> str:
    """
    Ask the LLM to choose a tool based on the user input.

    Returns:
        str: A prompt requesting a registered tool name or ``None``.
    """
    return """You are an AI planner that selects a tool based on the user input.
    You have access to the following tools:
    1. get_current_time: Returns the current date and time.
    2. roll_dice: Returns a random number between 1 and 6.
    3. generate_password: Generates a secure random password.
    
    Based on the user input, choose the most appropriate tool.
    Return only its exact name (get_current_time, roll_dice, or generate_password).
    If no tool applies, return only None.
    """


def fetch_llm_tool_decision(user_input: str, client: Client, model: str) -> str:
    """Ask Gemini to choose a registered tool for the user's request."""
    return generate_assistant_reply(
        client=client,
        model=model,
        messages=[{"role": "user", "content": user_input}],
        system_instruction=plan_llm_tools_prompt(),
    )