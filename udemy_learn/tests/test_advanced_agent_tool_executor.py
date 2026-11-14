from contextlib import asynccontextmanager
import unittest
from unittest.mock import AsyncMock, patch

from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ErrorData, TextContent

from udemy_learn.advanced_agent.tool_executor import (
    ToolExecutionError,
    execute_and_parse_tool_response,
    execute_tool_with_retry,
    parse_tool_response,
)


class AdvancedAgentToolExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_transient_mcp_timeouts_then_parses_result(self) -> None:
        result = CallToolResult(
            content=[TextContent(type="text", text="Sunny, 20 C")],
            isError=False,
        )
        timeout = McpError(ErrorData(code=408, message="Request timed out"))
        with (
            patch(
                "udemy_learn.advanced_agent.tool_executor.execute_tool",
                new_callable=AsyncMock,
                side_effect=[timeout, timeout, result],
            ) as execute,
            patch(
                "udemy_learn.advanced_agent.tool_executor.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep,
        ):
            response = await execute_and_parse_tool_response(
                object(),
                "get_weather",
                {"city": "Paris"},
            )

        self.assertEqual(response, "Sunny, 20 C")
        self.assertEqual(execute.await_count, 3)
        self.assertEqual(sleep.await_count, 2)

    async def test_does_not_retry_non_transient_mcp_errors(self) -> None:
        error = McpError(ErrorData(code=-32602, message="Invalid arguments"))
        with patch(
            "udemy_learn.advanced_agent.tool_executor.execute_tool",
            new_callable=AsyncMock,
            side_effect=error,
        ) as execute:
            with self.assertRaises(ToolExecutionError) as raised:
                await execute_tool_with_retry(object(), "get_weather", {})

        self.assertIs(raised.exception.__cause__, error)
        self.assertEqual(raised.exception.attempts, 1)
        self.assertFalse(raised.exception.retries_exhausted)
        self.assertEqual(execute.await_count, 1)

    async def test_reports_attempt_count_after_retry_limit_is_exhausted(self) -> None:
        timeout = McpError(ErrorData(code=408, message="Request timed out"))
        with (
            patch(
                "udemy_learn.advanced_agent.tool_executor.execute_tool",
                new_callable=AsyncMock,
                side_effect=timeout,
            ) as execute,
            patch(
                "udemy_learn.advanced_agent.tool_executor.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep,
        ):
            with self.assertRaises(ToolExecutionError) as raised:
                await execute_tool_with_retry(object(), "get_weather", {})

        self.assertEqual(raised.exception.attempts, 3)
        self.assertTrue(raised.exception.retries_exhausted)
        self.assertEqual(execute.await_count, 3)
        self.assertEqual(sleep.await_count, 2)

    async def test_does_not_retry_tool_reported_errors(self) -> None:
        result = CallToolResult(
            content=[TextContent(type="text", text="Service unavailable")],
            isError=True,
        )
        with patch(
            "udemy_learn.advanced_agent.tool_executor.execute_tool",
            new_callable=AsyncMock,
            return_value=result,
        ) as execute:
            response = await execute_tool_with_retry(
                object(),
                "get_weather",
                {},
            )
            with self.assertRaisesRegex(RuntimeError, "get_weather.*failed"):
                parse_tool_response(response, "get_weather")

        self.assertEqual(execute.await_count, 1)

    def test_parses_structured_content_when_no_text_is_returned(self) -> None:
        result = CallToolResult(
            content=[],
            structuredContent={"temperature": 20},
            isError=False,
        )

        self.assertEqual(
            parse_tool_response(result, "get_weather"),
            '{"temperature": 20}',
        )

    async def test_loop_records_attempt_count_and_marks_task_failed(self) -> None:
        from udemy_learn.advanced_agent.loop import run_advanced_agent_loop

        @asynccontextmanager
        async def fake_connect():
            yield object()

        planner_decisions = [
            {
                "task": "Get the weather",
                "tool_name": "get_weather",
                "arguments": {"city": "Paris"},
                "task_state": "pending",
                "reason": "",
            },
            {
                "task": "Get the weather",
                "tool_name": None,
                "arguments": {},
                "task_state": "failed",
                "reason": "Tool failed after three attempts.",
            },
        ]
        with (
            patch(
                "udemy_learn.advanced_agent.loop.connect",
                fake_connect,
            ),
            patch(
                "udemy_learn.advanced_agent.loop.fetch_task_list_from_td_agent",
                new_callable=AsyncMock,
                return_value=[
                    {"task": "Get the weather", "state": "pending"}
                ],
            ),
            patch(
                "udemy_learn.advanced_agent.loop.tool_discovery",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "udemy_learn.advanced_agent.loop.planner_agent",
                new_callable=AsyncMock,
                side_effect=planner_decisions,
            ) as planner,
            patch(
                "udemy_learn.advanced_agent.loop.execute_and_parse_tool_response",
                new_callable=AsyncMock,
                side_effect=ToolExecutionError(
                    "get_weather",
                    3,
                    "Automatic retry limit exhausted: temporary timeout",
                    retries_exhausted=True,
                ),
            ) as execute,
        ):
            state = await run_advanced_agent_loop(
                "Get the weather for Paris",
                object(),
                "test-model",
            )

        self.assertEqual(planner.await_count, 2)
        self.assertEqual(execute.await_count, 1)
        self.assertEqual(state.tasks[0]["state"], "failed")
        self.assertIn("3 attempts", state.observations[0]["observation"])
        self.assertIn(
            "do not retry this call again",
            state.observations[0]["observation"],
        )


if __name__ == "__main__":
    unittest.main()
