import unittest
from unittest.mock import patch

from udemy_learn.planner.llm_tool_workflow import run_llm_tool_workflow


class LLMToolWorkflowTests(unittest.TestCase):
    def test_selects_and_executes_tool(self) -> None:
        client = object()
        with (
            patch(
                "udemy_learn.planner.llm_tool_workflow.fetch_llm_tool_decision",
                return_value="get_current_time",
            ) as fetch_decision,
            patch(
                "udemy_learn.planner.llm_tool_workflow.execute_llm_tool",
                return_value="12:34",
            ) as execute_tool,
        ):
            result = run_llm_tool_workflow("What time is it?", client, "test-model")

        self.assertEqual(result, "12:34")
        fetch_decision.assert_called_once_with(
            user_input="What time is it?",
            client=client,
            model="test-model",
        )
        execute_tool.assert_called_once_with("get_current_time")

    def test_returns_none_when_planner_selects_no_tool(self) -> None:
        with (
            patch(
                "udemy_learn.planner.llm_tool_workflow.fetch_llm_tool_decision",
                return_value="None",
            ),
            patch(
                "udemy_learn.planner.llm_tool_workflow.execute_llm_tool",
                return_value=None,
            ) as execute_tool,
        ):
            result = run_llm_tool_workflow("Explain recursion", object(), "test-model")

        self.assertIsNone(result)
        execute_tool.assert_called_once_with("None")


if __name__ == "__main__":
    unittest.main()
