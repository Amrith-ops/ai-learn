import unittest
from unittest.mock import patch

from udemy_learn.planner.tool_executor import LLM_TOOL_HANDLERS, execute_llm_tool


class ToolExecutorTests(unittest.TestCase):
    def test_executes_tool_by_exact_name(self) -> None:
        with patch.dict(LLM_TOOL_HANDLERS, {"get_current_time": lambda: "12:34"}):
            self.assertEqual(execute_llm_tool("get_current_time"), "12:34")

    def test_converts_non_string_tool_result_to_text(self) -> None:
        with patch.dict(LLM_TOOL_HANDLERS, {"roll_dice": lambda: "4"}):
            self.assertEqual(execute_llm_tool("roll_dice"), "4")

    def test_none_selection_does_not_run_a_tool(self) -> None:
        self.assertIsNone(execute_llm_tool(" None "))

    def test_unknown_tool_name_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown tool selected by LLM"):
            execute_llm_tool("delete_everything")


if __name__ == "__main__":
    unittest.main()
