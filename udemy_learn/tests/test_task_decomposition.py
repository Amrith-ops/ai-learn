import unittest
from unittest.mock import patch

from udemy_learn.advanced_agent.task_decomposition_agent import (
    fetch_task_list_from_td_agent,
    parse_task_decomposition_response,
    task_decomposition_prompt,
)
from udemy_learn.utils.state import AgentState


class TaskDecompositionTests(unittest.TestCase):
    def test_prompt_includes_relevant_memory_and_finish_rules(self) -> None:
        state = AgentState(
            user_input="What did I ask about SQLite?",
            memory=[
                {
                    "user_request": "How does SQLite connect to a file?",
                    "assistant_response": "sqlite3.connect opens or creates the file.",
                    "score": 0.9,
                    "created_at": "2026-11-14 12:00:00",
                }
            ],
        )

        prompt = task_decomposition_prompt(state.user_input, state)

        self.assertIn("How does SQLite connect to a file?", prompt)
        self.assertIn("sqlite3.connect opens or creates the file.", prompt)
        self.assertIn('{"decision": "Finish"', prompt)
        self.assertIn("Create at least two tasks", prompt)

    def test_parses_finish_response_without_tasks(self) -> None:
        result = parse_task_decomposition_response(
            '{"decision":"Finish","answer":"SQLite opens the database file.",'
            '"tasks":[]}'
        )

        self.assertEqual(
            result,
            {
                "decision": "Finish",
                "answer": "SQLite opens the database file.",
                "tasks": [],
            },
        )

    def test_parses_json_object_inside_fence_and_surrounding_text(self) -> None:
        result = parse_task_decomposition_response(
            'Here is the result:\n```json\n'
            '{"decision":"Finish","answer":"Hello.","tasks":[]}\n```'
        )

        self.assertEqual(result["decision"], "Finish")
        self.assertEqual(result["answer"], "Hello.")

    def test_invalid_json_error_includes_response_preview(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Response preview: 'not valid json'",
        ):
            parse_task_decomposition_response("not valid json")

    def test_parses_multi_task_response(self) -> None:
        result = parse_task_decomposition_response(
            '{"decision":"CreateTasks","answer":"","tasks":['
            '{"task":"Get the weather for Paris","state":"pending"},'
            '{"task":"Get the weather for Rome","state":"pending"}]}'
        )

        self.assertEqual(result["decision"], "CreateTasks")
        self.assertEqual(len(result["tasks"]), 2)

    def test_rejects_create_tasks_response_with_only_one_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two tasks"):
            parse_task_decomposition_response(
                '{"decision":"CreateTasks","answer":"","tasks":['
                '{"task":"Get the weather","state":"pending"}]}'
            )

    def test_rejects_finish_response_with_tasks(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not include tasks"):
            parse_task_decomposition_response(
                '{"decision":"Finish","answer":"Hello","tasks":['
                '{"task":"Greet the user","state":"pending"}]}'
            )


class FetchTaskListTests(unittest.IsolatedAsyncioTestCase):
    async def test_passes_memory_in_prompt_and_parses_reply(self) -> None:
        response = (
            '{"decision":"Finish","answer":"SQLite opens or creates the file.",'
            '"tasks":[]}'
        )
        state = AgentState(
            user_input="What did I ask about SQLite?",
            memory=[
                {
                    "user_request": "How does SQLite connect to a file?",
                    "assistant_response": "sqlite3.connect opens or creates the file.",
                    "score": 0.9,
                    "created_at": "2026-11-14 12:00:00",
                }
            ],
        )
        with patch(
            "udemy_learn.advanced_agent.task_decomposition.generate_assistant_json_reply",
            return_value=response,
        ) as generate_reply:
            result = await fetch_task_list_from_td_agent(
                object(),
                "test-model",
                state.user_input,
                state,
            )

        self.assertEqual(result["decision"], "Finish")
        sent_prompt = generate_reply.call_args.kwargs["messages"][0]["content"]
        self.assertIn("sqlite3.connect opens or creates the file.", sent_prompt)
        self.assertEqual(
            generate_reply.call_args.kwargs["system_instruction"],
            "You route direct-answer requests or decompose actionable requests.",
        )


if __name__ == "__main__":
    unittest.main()
