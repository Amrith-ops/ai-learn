import asyncio
from collections.abc import Callable

from google import genai

from udemy_learn.agent_loop.agent_loop_workflow import agent_loop_run_workflow
from udemy_learn.utils.assistant import generate_assistant_reply
from udemy_learn.config.config import load_config_env
from udemy_learn.manual_workflow_tool.manual_tool_orcheastrate import (
    get_system_instruction,
    llm_capability_routing,
)
from udemy_learn.planner.llm_tool_workflow import run_llm_tool_workflow
from udemy_learn.mcp_client.mcp_workflow import run_mcp_workflow
from udemy_learn.request_classifier import classify_request


async def run_conversation(
    client: genai.Client,
    model: str,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run the terminal chat while keeping history and tool routing in one place."""
    output_fn("=" * 40)
    output_fn("      My AI Assistant")
    output_fn("=" * 40)
    system_instruction = get_system_instruction(input_fn, output_fn)

    # The conversation owns history so every Gemini turn can include prior exchanges.
    messages: list[dict[str, str]] = []
    while True:
        user_input = input_fn("\nYou : ")
        if user_input.strip().casefold() == "quit":
            output_fn("\nAI  : Goodbye! Have a great day.")
            return
        if load_config_env().enable_manual_workflow:
            output_fn("\nPython  : Routing the conversation through the manual workflow...")
            workflow_result = llm_capability_routing(
                user_input,
                system_instruction,
                output_fn,
            )
            if workflow_result.direct_response is not None:
                output_fn(f"\nPython  : {workflow_result.direct_response}")
                continue
            if workflow_result.model_prompt is None:
                raise RuntimeError("Manual workflow produced neither a model prompt nor a tool response.")

            messages.append({"role": "user", "content": workflow_result.model_prompt})
            output_fn("\nPython  : Sending the conversation to Gemini for a response...")
            ai_reply = generate_assistant_reply(
                client=client,
                model=model,
                messages=messages,
                system_instruction=workflow_result.system_instruction,
            )
            output_fn(f"\nAI : {ai_reply}")
            messages.append({"role": "assistant", "content": ai_reply})
        elif load_config_env().tool_discovery_manual:
            output_fn("\nPython  : Routing the conversation through the LLM deciding which tool to use workflow...")
            tool_result = run_llm_tool_workflow(user_input, client, model)
            messages.append({"role": "user", "content": tool_result})
            ai_reply = generate_assistant_reply(
                client=client,
                model=model,
                messages=messages,
                system_instruction= tool_result,
            )
            output_fn(f"\nAI : {ai_reply}")
            messages.append({"role": "assistant", "content": ai_reply})
        else:
            request_type = await asyncio.to_thread(
                classify_request,
                user_input,
                client,
                model,
            )
            messages.append({"role": "user", "content": user_input})
            if request_type == "multi-step":
                output_fn("\nPython: Routing the conversation through the agent loop")
                agent_state = await agent_loop_run_workflow(user_input, client, model)
                for observation in agent_state.observations:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Agent step {observation['step']} "
                                f"({observation['action']}):\n"
                                f"{observation['observation']}"
                            ),
                        }
                    )
                if agent_state.final_answer is None:
                    raise RuntimeError("Agent loop completed without a final answer.")
                ai_reply = agent_state.final_answer
            else:
                output_fn("\nPython: Routing the conversation through the LLM + MCP workflow")
                tool_execution = await run_mcp_workflow(client, user_input, model)
                if tool_execution is not None:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Tool result from {tool_execution.tool_name}:\n"
                                f"{tool_execution.content}"
                            ),
                        }
                    )
                ai_reply = await asyncio.to_thread(
                    generate_assistant_reply,
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


async def main() -> None:
    """Load runtime configuration and start the terminal conversation."""
    settings = load_config_env()
    if not settings.api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment or .env file.")

    client = genai.Client(api_key=settings.api_key)
    await run_conversation(client=client, model=settings.model)
