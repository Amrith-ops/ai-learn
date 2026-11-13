import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult, TextContent, Tool

from udemy_learn.agent_loop.agent_loop_workflow import agent_loop_run_workflow


class AgentLoopWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_multiple_tools_and_returns_planner_final_answer(self) -> None:
        tools = [
            Tool(name="get_current_time", description="Get current time.", inputSchema={"type": "object"}),
            Tool(name="roll_dice", description="Roll a die.", inputSchema={"type": "object"}),
        ]
        session = SimpleNamespace(
            call_tool=AsyncMock(
                side_effect=[
                    CallToolResult(
                        content=[TextContent(type="text", text="10:00")],
                        isError=False,
                    ),
                    CallToolResult(
                        content=[TextContent(type="text", text="4")],
                        isError=False,
                    ),
                ]
            )
        )

        @asynccontextmanager
        async def fake_connect():
            yield session

        planner_decisions = [
            '{"tool_name":"get_current_time","arguments":{}}',
            '{"tool_name":"roll_dice","arguments":{}}',
            (
                '{"tool_name":null,"arguments":{},'
                '"final_answer":"It is 10:00 and the die roll is 4."}'
            ),
        ]

        with (
            patch(
                "udemy_learn.agent_loop.agent_loop_workflow.connect",
                fake_connect,
            ),
            patch(
                "udemy_learn.agent_loop.agent_loop_workflow.tool_discovery",
                new_callable=AsyncMock,
                return_value=tools,
            ),
            patch(
                "udemy_learn.agent_loop.agent_loop_workflow.fetch_next_action_recommendation_planner",
                new_callable=AsyncMock,
                side_effect=planner_decisions,
            ) as planner,
        ):
            result = await agent_loop_run_workflow(
                "Get the time, then roll a die",
                SimpleNamespace(),
                "test-model",
            )

        self.assertTrue(result.finished)
        self.assertEqual(
            result.final_answer,
            "It is 10:00 and the die roll is 4.",
        )
        self.assertEqual(result.current_step, 2)
        self.assertEqual(
            result.observations,
            [
                {
                    "step": 1,
                    "action": "get_current_time({})",
                    "observation": "10:00",
                },
                {
                    "step": 2,
                    "action": "roll_dice({})",
                    "observation": "4",
                },
            ],
        )
        self.assertEqual(planner.await_count, 3)
        session.call_tool.assert_has_awaits(
            [
                unittest.mock.call("get_current_time", {}),
                unittest.mock.call("roll_dice", {}),
            ]
        )


if __name__ == "__main__":
    unittest.main()
