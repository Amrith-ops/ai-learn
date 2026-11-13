import unittest

from udemy_learn.agent_state.state import (
    create_agent_state,
    finish_agent_state,
    update_action_taken,
    update_current_step,
    update_observations,
)


class AgentStateTests(unittest.TestCase):
    def test_tracks_steps_actions_and_observations(self) -> None:
        state = create_agent_state("Find the current weather", max_steps=2)

        self.assertEqual(update_current_step(state), 1)
        update_action_taken(state, "get_weather")
        update_observations(state, "Sunny, 21 C")

        self.assertEqual(state.current_step, 1)
        self.assertEqual(state.actions, ["get_weather"])
        self.assertEqual(
            state.observations,
            [
                {
                    "step": 1,
                    "action": "get_weather",
                    "observation": "Sunny, 21 C",
                }
            ],
        )

    def test_state_collections_are_not_shared_between_runs(self) -> None:
        first = create_agent_state("First request")
        second = create_agent_state("Second request")

        first.actions.append("roll_dice")

        self.assertEqual(second.actions, [])
        self.assertEqual(second.observations, [])

    def test_step_limit_does_not_advance_state(self) -> None:
        state = create_agent_state("Do the task", max_steps=1)
        update_current_step(state)

        with self.assertRaisesRegex(RuntimeError, "exceeded the limit"):
            update_current_step(state)

        self.assertEqual(state.current_step, 1)

    def test_requires_observation_before_advancing_to_next_step(self) -> None:
        state = create_agent_state("Do the task", max_steps=2)
        update_current_step(state)
        update_action_taken(state, "first action")

        with self.assertRaisesRegex(RuntimeError, "observation before advancing"):
            update_current_step(state)
        with self.assertRaisesRegex(RuntimeError, "observation before finishing"):
            finish_agent_state(state, "Done.")

        update_observations(state, "first result")
        self.assertEqual(update_current_step(state), 2)

    def test_action_and_observation_must_follow_current_step(self) -> None:
        state = create_agent_state("Do the task")

        with self.assertRaisesRegex(RuntimeError, "Advance the agent step"):
            update_action_taken(state, "first action")

        update_current_step(state)
        with self.assertRaisesRegex(RuntimeError, "Record an action"):
            update_observations(state, "result")

    def test_finishing_stores_answer_and_prevents_further_updates(self) -> None:
        state = create_agent_state("Do the task")
        finish_agent_state(state, "Completed.")

        self.assertTrue(state.finished)
        self.assertEqual(state.final_answer, "Completed.")
        with self.assertRaisesRegex(RuntimeError, "finished"):
            update_current_step(state)

    def test_rejects_empty_requests_and_answers(self) -> None:
        with self.assertRaisesRegex(ValueError, "user_input"):
            create_agent_state("  ")

        state = create_agent_state("Do the task")
        with self.assertRaisesRegex(ValueError, "final_answer"):
            finish_agent_state(state, " ")


if __name__ == "__main__":
    unittest.main()
