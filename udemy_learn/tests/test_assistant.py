import unittest
from types import SimpleNamespace
from typing import cast

from google.genai import Client

from udemy_learn.utils.assistant import generate_assistant_json_reply


class AssistantReplyTests(unittest.TestCase):
    def test_passes_json_response_mode_to_gemini_config(self) -> None:
        requests = {}

        def generate_content(**request):
            requests.update(request)
            return SimpleNamespace(text='{"ok": true}')

        models = SimpleNamespace(generate_content=generate_content)
        client = cast(Client, SimpleNamespace(models=models))

        response = generate_assistant_json_reply(
            client,
            "test-model",
            [{"role": "user", "content": "Return JSON."}],
            "Use JSON.",
        )

        self.assertEqual(response, '{"ok": true}')
        self.assertEqual(
            requests["config"].response_mime_type,
            "application/json",
        )


if __name__ == "__main__":
    unittest.main()
