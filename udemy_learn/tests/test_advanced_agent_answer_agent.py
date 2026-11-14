import unittest
from unittest.mock import patch

from udemy_learn.advanced_agent.answer_agent import (
    advanced_agent_answer_prompt,
    answer_agent,
)
from udemy_learn.utils.state import AgentState


class AdvancedAgentAnswerAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.state = AgentState(
            user_input="Find the weather for Paris",
            tasks=[
                {"task": "Get the weather", "state": "completed"},
                {"task": "Check tomorrow's forecast", "state": "failed"},
            ],
            actions=["get_weather(city='Paris')"],
            observations=[
                {
                    "step": 1,
                    "action": "get_weather(city='Paris')",
                    "observation": "Sunny, 20 C",
                }
            ],
            results=[
                "Sunny, 20 C",
                "failed: Check tomorrow's forecast - provider unavailable",
            ],
        )

    def test_prompt_includes_request_and_state_evidence(self) -> None:
        prompt = advanced_agent_answer_prompt(self.state)

        self.assertIn("Find the weather for Paris", prompt)
        self.assertIn('"state": "completed"', prompt)
        self.assertIn('"state": "failed"', prompt)
        self.assertIn("get_weather(city='Paris')", prompt)
        self.assertIn("Sunny, 20 C", prompt)
        self.assertIn("provider unavailable", prompt)
        self.assertIn("clearly disclose", prompt)

    async def test_returns_natural_language_reply_from_assistant(self) -> None:
        client = object()
        with patch(
            "udemy_learn.advanced_agent.answer_agent.generate_assistant_reply",
            return_value=(
                "Paris is sunny at 20 C. Tomorrow's forecast could not be "
                "retrieved because the provider was unavailable."
            ),
        ) as generate_reply:
            answer = await answer_agent(client, "test-model", self.state)

        self.assertIn("Paris is sunny", answer)
        self.assertIs(generate_reply.call_args.args[0], client)
        self.assertEqual(
            generate_reply.call_args.kwargs["system_instruction"],
            "You are a concise, evidence-grounded assistant. Answer the user's "
            "request naturally and disclose any incomplete tasks.",
        )


if __name__ == "__main__":
    unittest.main()
