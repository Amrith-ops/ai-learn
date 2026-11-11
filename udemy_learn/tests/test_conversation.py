import unittest
from types import SimpleNamespace

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
    def test_manual_tool_stays_local_and_chat_sends_full_history(self) -> None:
        client = FakeClient()
        outputs = []
        inputs = iter(["2", "roll dice", "first question", "second question", "quit"])

        run_conversation(client, "test-model", lambda _prompt: next(inputs), outputs.append)

        requests = client.models.requests
        self.assertEqual(len(requests), 2)
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
        self.assertIn("You rolled a ", "\n".join(outputs))
        self.assertEqual(
            requests[0]["config"].system_instruction,
            "You are an expert Python programmer and software engineer. "
            "Provide knowledgeable, patient, and clear explanations.",
        )

    def test_summary_content_is_passed_to_gemini(self) -> None:
        client = FakeClient()
        inputs = iter(["1", "summarize python.txt", "quit"])

        run_conversation(client, "test-model", lambda _prompt: next(inputs), lambda _text: None)

        prompt = client.models.requests[0]["contents"][0]["parts"][0]["text"]
        self.assertTrue(prompt.startswith("Summarize the following content:\n"))
        self.assertIn("Python is a high-level programming language.", prompt)


if __name__ == "__main__":
    unittest.main()
