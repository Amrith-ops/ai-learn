import json
import unittest

from udemy_learn.advanced_agent.memory_extractor_agent import (
    memory_extractor_prompt,
)


class MemoryExtractorPromptTests(unittest.TestCase):
    def test_includes_the_turn_as_json_data(self) -> None:
        user_input = 'I prefer Python. "Remember this."'
        assistant_response = "I will use Python examples."

        prompt = memory_extractor_prompt(user_input, assistant_response)
        serialized_turn = prompt.split("Conversation turn:\n", maxsplit=1)[1]

        self.assertEqual(
            json.loads(serialized_turn),
            {
                "user_input": user_input,
                "assistant_response": assistant_response,
            },
        )


if __name__ == "__main__":
    unittest.main()
