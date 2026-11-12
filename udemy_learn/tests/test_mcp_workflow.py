import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult, TextContent, Tool

from udemy_learn.mcp_client.mcp_workflow import (
    ToolExecution,
    parse_tool_selection,
    run_mcp_workflow,
)


class MCPWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = Tool(
            name="generate_password",
            description="Generate a secure password.",
            inputSchema={
                "type": "object",
                "properties": {"length": {"type": "integer"}},
            },
        )

    async def test_passes_model_arguments_to_selected_server_tool(self) -> None:
        session = SimpleNamespace(
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[self.tool])),
            call_tool=AsyncMock(
                return_value=CallToolResult(
                    content=[TextContent(type="text", text="secure-password")],
                    isError=False,
                )
            ),
        )

        @asynccontextmanager
        async def fake_connect():
            yield session

        with (
            patch(
                "udemy_learn.mcp_client.mcp_workflow.connect",
                fake_connect,
            ),
            patch(
                "udemy_learn.mcp_client.mcp_workflow.generate_assistant_reply",
                return_value='{"tool_name":"generate_password","arguments":{"length":16}}',
            ),
        ):
            result = await run_mcp_workflow(
                client=SimpleNamespace(),
                user_input="Generate a 16-character password",
                model="test-model",
            )

        self.assertEqual(
            result,
            ToolExecution(
                tool_name="generate_password",
                content="secure-password",
            ),
        )
        session.call_tool.assert_awaited_once_with(
            "generate_password",
            {"length": 16},
        )

    async def test_no_tool_selection_returns_none_without_calling_server_tool(self) -> None:
        session = SimpleNamespace(
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[self.tool])),
            call_tool=AsyncMock(),
        )

        @asynccontextmanager
        async def fake_connect():
            yield session

        with (
            patch(
                "udemy_learn.mcp_client.mcp_workflow.connect",
                fake_connect,
            ),
            patch(
                "udemy_learn.mcp_client.mcp_workflow.generate_assistant_reply",
                return_value='{"tool_name":null,"arguments":{}}',
            ),
        ):
            result = await run_mcp_workflow(
                client=SimpleNamespace(),
                user_input="Explain recursion",
                model="test-model",
            )

        self.assertIsNone(result)
        session.call_tool.assert_not_awaited()

    def test_rejects_unknown_tool_selected_by_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown MCP tool"):
            parse_tool_selection(
                '{"tool_name":"delete_data","arguments":{}}',
                [self.tool],
            )


if __name__ == "__main__":
    unittest.main()
