from collections.abc import Callable
from dataclasses import dataclass

from udemy_learn.config.config import load_config_env
from udemy_learn.router.tool_router import route_tool_command
from udemy_learn.rag.retrieval_workflow import run_retrieval_workflow
from udemy_learn.tools.files import DATA_DIRECTORY

"""
Manual tool worflow orchestration is a learning step in the Udemy course "Building AI Agents with Gemini API".
"""
DEFAULT_PERSONALITY = (
   "You are an expert Python programmer and software engineer. "
           "Provide knowledgeable, patient, and clear explanations."
)

PERSONALITIES = {
    "1":  "You are an empathetic psychologist and life coach. "
        "Provide thoughtful and supportive guidance.",
    "2": (
        DEFAULT_PERSONALITY
    )
}


@dataclass(frozen=True)
class ManualWorkflowResult:
    system_instruction: str
    model_prompt: str | None
    direct_response: str | None


def get_system_instruction(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> str:
    """Choose the conversation's system instruction once at startup."""
    if not load_config_env().enable_personality:
        return "You are a helpful AI assistant."

    output_fn("Personality feature is enabled.")
    choice = input_fn("Choose a personality (1: Psychologist, 2: Programmer, 3: Chef): ")
    output_fn(f"Selected personality: {choice}")
    return PERSONALITIES.get(choice, DEFAULT_PERSONALITY)


def llm_capability_routing(
    user_input: str,
    system_instruction: str,
    output_fn: Callable[[str], None] = print,
) -> ManualWorkflowResult:
    """
    Route the user input to the appropriate tool based on the content of the input.
    This function is a placeholder for manual routing logic.
    Here, based on my learning journey first i learned tune LLM personality 
    then I learned to route the user input to appropriate tools based on the content of the input.
    Then i learned to route the user input to the RAG retrieval tool if no other tool is applicable.

    Args:
        user_input (str): The input provided by the user.

    Returns:
        ManualWorkflowResult: The selected model prompt or direct tool response.
    """
    tool_response = route_tool_command(user_input)
    if tool_response is not None:
        if not tool_response.startswith("Summarize the following content:"):
            output_fn(
                "\nPython : User has asked a question that can be answered "
                "by a non-file-reading tool. Routing to the appropriate tool."
            )
            return ManualWorkflowResult(
                system_instruction=system_instruction,
                model_prompt=None,
                direct_response=tool_response,
            )

        output_fn(
            "\nPython : User has asked a question that requires file reading. "
            "Routing to the non-RAG retrieval tool."
        )
        return ManualWorkflowResult(
            system_instruction=system_instruction,
            model_prompt=tool_response,
            direct_response=None,
        )
        
    else:
        output_fn(
            "\nPython : User has asked a question that requires RAG retrieval. "
            "Routing to the RAG retrieval tool."
        )
        # Semantic retrieval is the next explicit learning step: retrieve
        # local context first, then provide that text and the original
        # question to Gemini (which still writes the final response).
        retrieved_document = run_retrieval_workflow(
            data_folder=DATA_DIRECTORY,
            user_query=user_input,
        )
        if retrieved_document is None:
            model_prompt = user_input
        else:
            file_name, content, similarity = retrieved_document
            model_prompt = (
                "Use the retrieved document context to answer the user's "
                "question. If the context does not contain the answer, say so.\n\n"
                f"Retrieved from {file_name} (similarity: {similarity:.3f}):\n"
                f"{content}\n\n"
                f"User question: {user_input}"
            )

        return ManualWorkflowResult(
            system_instruction=system_instruction,
            model_prompt=model_prompt,
            direct_response=None,
        )