"""Coordinate LLM tool selection and execution."""

from google.genai import Client

from udemy_learn.planner.llm_tools_planner import fetch_llm_tool_decision
from udemy_learn.planner.tool_executor import execute_llm_tool


def run_llm_tool_workflow(
    user_input: str,
    client: Client,
    model: str,
) -> str | None:
    """Ask Gemini to select a tool, execute it, and return its result.

    Returns ``None`` when the planner determines that no registered tool
    applies. Unknown tool names are rejected by the executor.
    """
    tool_name = fetch_llm_tool_decision(
        user_input=user_input,
        client=client,
        model=model,
    )
    result = execute_llm_tool(tool_name)
    return f"""
        The user asked:
        {user_input}
        The tool returned:
        {result}
        Answer the user naturally.don't add anything from your side in the answer. if tool is none, then answer it from llm model.
        """