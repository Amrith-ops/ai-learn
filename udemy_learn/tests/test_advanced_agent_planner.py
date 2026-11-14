import json
import unittest

from mcp.types import Tool

from udemy_learn.advanced_agent.planner import (
    advanced_agent_planner_prompt,
    parse_planner_agent_response,
)
from udemy_learn.utils.state import AgentState


class AdvancedAgentPlannerPromptTests(unittest.TestCase):
    def test_includes_task_state_history_results_and_tool_schemas(self) -> None:
        state = AgentState(
            user_input="Find the weather and summarize it",
            tasks=[
                {"task": "Get the weather", "state": "pending"},
                {"task": "Summarize the result", "state": "pending"},
            ],
            current_task="Get the weather",
            actions=["get_weather({})"],
            observations=[
                {
                    "step": 1,
                    "action": "get_weather({})",
                    "observation": "Sunny, 20 C",
                }
            ],
            results=["Sunny, 20 C"],
        )
        tools = [
            Tool(
                name="get_weather",
                description="Get weather for a city.",
                inputSchema={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ]

        prompt = advanced_agent_planner_prompt(tools, state)

        self.assertIn("Find the weather and summarize it", prompt)
        self.assertIn('"task": "Get the weather"', prompt)
        self.assertIn("get_weather({})", prompt)
        self.assertIn("Sunny, 20 C", prompt)
        self.assertIn('"name": "get_weather"', prompt)
        self.assertIn('"city"', prompt)
        self.assertIn('"tool_name": "exact_tool_name"', prompt)
        self.assertIn('"tool_name": null', prompt)

    def test_defines_recovery_and_failed_task_behavior(self) -> None:
        prompt = advanced_agent_planner_prompt(
            [],
            AgentState(user_input="Check the weather"),
        )
        normalized_prompt = " ".join(prompt.split()).lower()

        self.assertIn("retry limit was exhausted", normalized_prompt)
        self.assertIn("do not request another trial", normalized_prompt)
        self.assertIn('"task_state": "failed"', normalized_prompt)
        self.assertIn("continue with other independent pending tasks", normalized_prompt)
        self.assertIn(
            "final answer discloses failed tasks and their reasons",
            normalized_prompt,
        )

    def test_agent_state_initializes_without_a_selected_task(self) -> None:
        state = AgentState(user_input="Check the weather")

        self.assertEqual(state.current_task, "")
        self.assertEqual(state.results, [])


class PlannerAgentResponseParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = AgentState(
            user_input="Check the weather",
            tasks=[{"task": "Get the weather", "state": "pending"}],
        )
        self.tools = [
            Tool(
                name="get_weather",
                description="Get weather for a city.",
                inputSchema={"type": "object"},
            )
        ]

    def parse(self, **overrides: object) -> dict[str, object]:
        decision: dict[str, object] = {
            "task": "Get the weather",
            "tool_name": "get_weather",
            "arguments": {"city": "Paris"},
            "task_state": "pending",
            "reason": "",
        }
        decision.update(overrides)
        return parse_planner_agent_response(
            json.dumps(decision),
            self.tools,
            self.state,
        )

    def test_parses_a_valid_tool_action(self) -> None:
        decision = self.parse()

        self.assertEqual(decision["task"], "Get the weather")
        self.assertEqual(decision["tool_name"], "get_weather")
        self.assertEqual(decision["arguments"], {"city": "Paris"})
        self.assertEqual(decision["task_state"], "pending")

    def test_accepts_a_rationale_for_a_pending_tool_action(self) -> None:
        decision = self.parse(reason="Use the weather tool to retrieve conditions.")

        self.assertEqual(
            decision["reason"],
            "Use the weather tool to retrieve conditions.",
        )

    def test_ignores_unrecognized_extra_fields(self) -> None:
        decision = self.parse(confidence=0.9)

        self.assertEqual(
            set(decision),
            {"task", "tool_name", "arguments", "task_state", "reason"},
        )

    def test_parses_completed_and_failed_task_decisions(self) -> None:
        for task_state, reason in (
            ("completed", "Weather result is available."),
            ("failed", "The weather service is unavailable."),
        ):
            with self.subTest(task_state=task_state):
                decision = self.parse(
                    tool_name=None,
                    arguments={},
                    task_state=task_state,
                    reason=reason,
                )

                self.assertEqual(decision["task_state"], task_state)
                self.assertEqual(decision["reason"], reason)

    def test_rejects_malformed_or_inconsistent_responses(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            parse_planner_agent_response("not json", self.tools, self.state)

        with self.assertRaisesRegex(ValueError, "missing required keys"):
            parse_planner_agent_response(
                '{"task": "Get the weather"}',
                self.tools,
                self.state,
            )

        with self.assertRaisesRegex(ValueError, "pending task must select a tool"):
            self.parse(tool_name=None)

        with self.assertRaisesRegex(ValueError, "must include a reason"):
            self.parse(
                tool_name=None,
                arguments={},
                task_state="failed",
                reason="",
            )

        with self.assertRaisesRegex(ValueError, "unknown tool"):
            self.parse(tool_name="not_available")

    def test_rejects_a_task_that_is_not_pending(self) -> None:
        self.state.tasks[0]["state"] = "completed"

        with self.assertRaisesRegex(ValueError, "not pending"):
            self.parse()


if __name__ == "__main__":
    unittest.main()
