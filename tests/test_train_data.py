from collections import Counter
import unittest

from src.train_data import build_order, checkpoint_schedule, step_tokens, take_fraction, val_examples
from src.tool_calls import format_tool_calls


class OrderTests(unittest.TestCase):
    def test_mix_shows_every_example_once_per_epoch_in_whole_batches(self):
        order = build_order(5, 3, order="mix", epochs=2, batch=4, seed=0)
        self.assertEqual(len(order), 16)
        self.assertEqual(Counter(order), Counter({key: 2 for key in [("chat", i) for i in range(5)]
                                                  + [("tool", i) for i in range(3)]}))
        self.assertEqual(Counter(order[:8]), Counter(order[8:]))   # each epoch is one full pass
        self.assertNotEqual(order[:8], order[8:])                  # reshuffled between epochs

    def test_seq_learns_chat_then_tool_and_no_batch_crosses_the_boundary(self):
        order = build_order(5, 3, order="seq", epochs=2, batch=4, seed=0)
        sources = [source for source, _ in order]
        # chat: 10 positions -> 8 kept; tool: 6 -> 4 kept (drop_last per stage).
        self.assertEqual(sources, ["chat"] * 8 + ["tool"] * 4)

    def test_order_is_fixed_by_the_seed(self):
        self.assertEqual(build_order(6, 6, order="mix", epochs=2, batch=3, seed=1),
                         build_order(6, 6, order="mix", epochs=2, batch=3, seed=1))
        self.assertNotEqual(build_order(6, 6, order="mix", epochs=2, batch=3, seed=1),
                            build_order(6, 6, order="mix", epochs=2, batch=3, seed=2))

    def test_unknown_order_is_rejected(self):
        with self.assertRaises(ValueError):
            build_order(1, 1, order="tool_first", epochs=1, batch=1, seed=0)

    def test_fraction_takes_a_prefix(self):
        self.assertEqual(take_fraction(list(range(10)), 0.3), [0, 1, 2])


class ScheduleTests(unittest.TestCase):
    def test_step_tokens_sum_each_optimizer_batch(self):
        sequence = [("chat", 0), ("tool", 0), ("chat", 1), ("tool", 1)]
        tokens = {("chat", 0): 3, ("chat", 1): 5, ("tool", 0): 7, ("tool", 1): 11}
        self.assertEqual(step_tokens(sequence, tokens, batch=2), [10, 16])

    def test_checkpoints_land_where_cumulative_tokens_first_reach_each_target(self):
        schedule = checkpoint_schedule([10] * 20, start=25, factor=2)   # cumulative 10, 20, …, 200
        self.assertEqual([(entry["step"], entry["target_tokens"]) for entry in schedule],
                         [(3, 25), (5, 50), (10, 100), (20, 200)])

    def test_one_step_passing_several_targets_is_saved_once_and_the_last_step_is_always_saved(self):
        schedule = checkpoint_schedule([100, 1, 1], start=10, factor=2)
        self.assertEqual([entry["step"] for entry in schedule], [1, 3])
        self.assertIsNone(schedule[-1]["target_tokens"])


class ValTests(unittest.TestCase):
    def test_reference_answers_become_the_final_assistant_turn(self):
        call = {"name": "get_weather", "arguments": {"city": "Hue"}}
        items = [{"id": "a", "set": "c", "group": "chat", "tools": [], "turns": [{"role": "user", "content": "hi"}],
                  "reference": "Hello!"},
                 {"id": "b", "set": "t_seen", "group": "single", "tools": [], "turns": [{"role": "user", "content": "w"}],
                  "expected_calls": [call]},
                 {"id": "c", "set": "t_unseen", "group": "missing_info", "tools": [],
                  "turns": [{"role": "user", "content": "w"}], "expected_calls": []},
                 {"id": "d", "set": "t_rename_nonsense", "group": "single", "tools": [],
                  "turns": [{"role": "user", "content": "w"}], "expected_calls": [call]}]
        val = val_examples(items)
        self.assertEqual(val["chat"][0]["turns"][-1], {"role": "assistant", "content": "Hello!"})
        self.assertEqual(val["tool_seen"][0]["turns"][-1]["content"], format_tool_calls([call]))
        self.assertEqual(val["tool_unseen"], [])   # Missing-info has no reference answer


if __name__ == "__main__":
    unittest.main()
