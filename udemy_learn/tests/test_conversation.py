import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


class ConversationTests(unittest.TestCase):
    def test_manual_tools_stay_local_and_chat_sends_full_history(self) -> None:
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
        ):
            run_conversation(client, "test-model", lambda _prompt: next(inputs), outputs.append)

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

    def test_retrieved_document_and_original_question_are_sent_to_gemini(self) -> None:
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
        ):
            run_conversation(
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

    def test_summary_content_is_passed_to_gemini(self) -> None:
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
        ):
            run_conversation(
                client,
                "test-model",
                lambda _prompt: next(inputs),
                lambda _text: None,
            )

        prompt = client.models.requests[0]["contents"][0]["parts"][0]["text"]
        self.assertTrue(prompt.startswith("Summarize the following content:\n"))
        self.assertIn("Python is a high-level programming language.", prompt)
        retrieval.assert_not_called()


if __name__ == "__main__":
    unittest.main()
