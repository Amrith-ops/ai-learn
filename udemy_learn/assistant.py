from collections.abc import Mapping, Sequence

from google.genai import Client, types


def generate_assistant_reply(
    client: Client,
    model: str,
    messages: Sequence[Mapping[str, str]],
    system_instruction: str,
) -> str:
    """Send the full user/assistant history to Gemini and return its reply."""
    # Provider boundary: keep Gemini's request format out of the conversation loop.
    conversation_history = [
        {
            "role": "user" if message["role"] == "user" else "model",
            "parts": [{"text": message["content"]}],
        }
        for message in messages
    ]
    response = client.models.generate_content(
        model=model,
        contents=conversation_history,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    if response.text is None:
        raise RuntimeError("Gemini returned no text for this response.")
    return response.text
