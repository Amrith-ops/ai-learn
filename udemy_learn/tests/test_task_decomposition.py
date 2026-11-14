import unittest
from unittest.mock import patch

from udemy_learn.advanced_agent.task_decomposition import (
    fetch_task_list_from_td_agent,
    parse_fetched_task_list,
)


class TaskDecompositionTests(unittest.TestCase):
    def test_parses_valid_task_list(self) -> None:
        result = parse_fetched_task_list(
            '[{"task": "Get the weather for Paris", "state": "pending"}]'
        )

        self.assertEqual(
            result,
            [{"task": "Get the weather for Paris", "state": "pending"}],
        )

    def test_rejects_task_with_unexpected_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly the keys"):
            parse_fetched_task_list(
                '[{"task": "Get the weather", "state": "pending", "extra": true}]'
            )

    def test_rejects_non_pending_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "state 'pending'"):
            parse_fetched_task_list(
                '[{"task": "Get the weather", "state": "complete"}]'
            )


class FetchTaskListTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetches_reply_in_a_thread_and_parses_it(self) -> None:
        response = '[{"task": "Get the weather for Paris", "state": "pending"}]'
        with patch(
            "udemy_learn.advanced_agent.task_decomposition.generate_assistant_reply",
            return_value=response,
        ) as generate_reply:
            result = await fetch_task_list_from_td_agent(
                object(), "test-model", "Get the weather for Paris"
            )

        self.assertEqual(
            result,
            [{"task": "Get the weather for Paris", "state": "pending"}],
        )
        self.assertEqual(
            generate_reply.call_args.kwargs["system_instruction"],
            "You are task decomposition agent",
        )


if __name__ == "__main__":
    unittest.main()
