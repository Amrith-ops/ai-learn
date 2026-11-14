import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from udemy_learn.conversation import run_conversation


class FakeModels:
    def __init__(self) -> None:
        self.requests = []

    def generate_content(self, **request):
        self.requests.append(request)
        return SimpleNamespace(text=f"reply {len(self.requests)}")


class FakeClient:
    def __init__(self) -> None:
        self.models = FakeModels()


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_tools_stay_local_and_chat_sends_full_history(self) -> None:
        client = FakeClient()
        outputs = []
        inputs = iter(["roll dice", "first question", "second question", "quit"])

        with (
            patch(
                "udemy_learn.conversation.get_system_instruction",
                return_value="programmer",
            ) as get_system_instruction,
            patch(
                "udemy_learn.manual_workflow_tool.manual_tool_orcheastrate.run_retrieval_workflow",
                return_value=None,
            ) as retrieval,
            patch(
                "udemy_learn.conversation.load_config_env",
                return_value=SimpleNamespace(
                    enable_manual_workflow=True,
                    tool_discovery_manual=False,
                ),
            ),
        ):
            await run_conversation(
                client,
                "test-model",
                lambda _prompt: next(inputs),
                outputs.append,
            )

        requests = client.models.requests
        self.assertEqual(len(requests), 2)
        get_system_instruction.assert_called_once()
        self.assertEqual(retrieval.call_count, 2)
        self.assertEqual(retrieval.call_args_list[0].kwargs["user_query"], "first question")
        self.assertEqual(retrieval.call_args_list[1].kwargs["user_query"], "second question")
        self.assertEqual(
            requests[0]["contents"],
            [{"role": "user", "parts": [{"text": "first question"}]}],
        )
        self.assertEqual(
            requests[1]["contents"],
            [
                {"role": "user", "parts": [{"text": "first question"}]},
                {"role": "model", "parts": [{"text": "reply 1"}]},
                {"role": "user", "parts": [{"text": "second question"}]},
            ],
        )
        self.assertEqual(requests[0]["config"].system_instruction, "programmer")
        self.assertIn("You rolled a ", "\n".join(outputs))

    async def test_retrieved_document_and_original_question_are_sent_to_gemini(self) -> None:
        client = FakeClient()
        inputs = iter(["What is Python used for?", "quit"])
        retrieved = (
            "python.txt",
            "Python is used for automation and data science.",
            0.82,
        )

        with (
            patch(
                "udemy_learn.conversation.get_system_instruction",
                return_value="helpful",
            ),
            patch(
                "udemy_learn.manual_workflow_tool.manual_tool_orcheastrate.run_retrieval_workflow",
                return_value=retrieved,
            ) as retrieval,
            patch(
                "udemy_learn.conversation.load_config_env",
                return_value=SimpleNamespace(
                    enable_manual_workflow=True,
                    tool_discovery_manual=False,
                ),
            ),
        ):
            await run_conversation(
                client,
                "test-model",
                lambda _prompt: next(inputs),
                lambda _text: None,
            )

        retrieval.assert_called_once()
        request = client.models.requests[0]
        sent_prompt = request["contents"][0]["parts"][0]["text"]
        self.assertIn("Python is used for automation and data science.", sent_prompt)
        self.assertIn("User question: What is Python used for?", sent_prompt)
        self.assertIn("similarity: 0.820", sent_prompt)
        self.assertEqual(request["config"].system_instruction, "helpful")

    async def test_summary_content_is_passed_to_gemini(self) -> None:
        client = FakeClient()
        inputs = iter(["summarize python.txt", "quit"])

        with (
            patch(
                "udemy_learn.conversation.get_system_instruction",
                return_value="helpful",
            ),
            patch(
                "udemy_learn.manual_workflow_tool.manual_tool_orcheastrate.run_retrieval_workflow"
            ) as retrieval,
            patch(
                "udemy_learn.conversation.load_config_env",
                return_value=SimpleNamespace(
                    enable_manual_workflow=True,
                    tool_discovery_manual=False,
                ),
            ),
        ):
            await run_conversation(
                client,
                "test-model",
                lambda _prompt: next(inputs),
                lambda _text: None,
            )

        prompt = client.models.requests[0]["contents"][0]["parts"][0]["text"]
        self.assertTrue(prompt.startswith("Summarize the following content:\n"))
        self.assertIn("Python is a high-level programming language.", prompt)
        retrieval.assert_not_called()

    async def test_mcp_result_is_given_to_gemini_as_user_context(self) -> None:
        client = FakeClient()
        inputs = iter(["roll a die", "quit"])

        with (
            patch(
                "udemy_learn.conversation.get_system_instruction",
                return_value="helpful",
            ),
            patch(
                "udemy_learn.conversation.load_config_env",
                return_value=SimpleNamespace(
                    enable_manual_workflow=False,
                    tool_discovery_manual=False,
                ),
            ),
            patch(
                "udemy_learn.conversation.classify_request",
                return_value="single-step",
            ),
            patch(
                "udemy_learn.conversation.run_mcp_workflow",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(
                    tool_name="roll_dice",
                    content="4",
                ),
            ),
        ):
            await run_conversation(
                client,
                "test-model",
                lambda _prompt: next(inputs),
                lambda _text: None,
            )

        request = client.models.requests[0]
        self.assertEqual(
            request["contents"],
            [
                {"role": "user", "parts": [{"text": "roll a die"}]},
                {
                    "role": "user",
                    "parts": [{"text": "Tool result from roll_dice:\n4"}],
                },
            ],
        )
        self.assertEqual(request["config"].system_instruction, "helpful")

    async def test_mcp_no_tool_selection_answers_original_request(self) -> None:
        client = FakeClient()
        inputs = iter(["Explain recursion", "quit"])

        with (
            patch(
                "udemy_learn.conversation.get_system_instruction",
                return_value="helpful",
            ),
            patch(
                "udemy_learn.conversation.load_config_env",
                return_value=SimpleNamespace(
                    enable_manual_workflow=False,
                    tool_discovery_manual=False,
                ),
            ),
            patch(
                "udemy_learn.conversation.classify_request",
                return_value="single-step",
            ),
            patch(
                "udemy_learn.conversation.run_mcp_workflow",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await run_conversation(
                client,
                "test-model",
                lambda _prompt: next(inputs),
                lambda _text: None,
            )

        request = client.models.requests[0]
        self.assertEqual(
            request["contents"],
            [{"role": "user", "parts": [{"text": "Explain recursion"}]}],
        )
        self.assertEqual(request["config"].system_instruction, "helpful")

    async def test_multi_step_request_runs_advanced_agent_loop(self) -> None:
        client = FakeClient()
        inputs = iter(["Get the time, then roll a die", "quit"])
        agent_state = SimpleNamespace(
            observations=[
                {
                    "step": 1,
                    "action": "get_current_time({})",
                    "observation": "10:00",
                },
                {
                    "step": 2,
                    "action": "roll_dice({})",
                    "observation": "4",
                },
            ],
            final_answer="It is 10:00 and the die roll is 4.",
        )

        with (
            patch(
                "udemy_learn.conversation.get_system_instruction",
                return_value="helpful",
            ),
            patch(
                "udemy_learn.conversation.load_config_env",
                return_value=SimpleNamespace(
                    enable_manual_workflow=False,
                    tool_discovery_manual=False,
                ),
            ),
            patch(
                "udemy_learn.conversation.classify_request",
                return_value="multi-step",
            ),
            patch(
                "udemy_learn.conversation.run_advanced_agent_loop",
                new_callable=AsyncMock,
                return_value=agent_state,
            ) as agent_loop,
            patch(
                "udemy_learn.conversation.run_mcp_workflow",
                new_callable=AsyncMock,
            ) as mcp_workflow,
        ):
            outputs = []
            await run_conversation(
                client,
                "test-model",
                lambda _prompt: next(inputs),
                outputs.append,
            )

        agent_loop.assert_awaited_once_with(
            "Get the time, then roll a die",
            client,
            "test-model",
        )
        mcp_workflow.assert_not_awaited()
        self.assertTrue(
            any("It is 10:00 and the die roll is 4." in output for output in outputs)
        )
        history = "\n".join(outputs)
        self.assertIn("get_current_time({})", history)
        self.assertIn("roll_dice({})", history)


if __name__ == "__main__":
    unittest.main()
