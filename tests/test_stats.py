import unittest

import numpy as np

from src.eval.run_eval import aggregate, score
from src.eval.stats import (BEHAVIOUR, TOOL, clusters, evaluate, first_held, go_no_go, metric_values,
                            rename_differences, resample_weights, run_differences, term_arrays)


TOOLS = [{"name": "get_weather", "description": "Weather now.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
         {"name": "get_time", "description": "Time now.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Hanoi"}}\n</tool_call><|im_end|>'
WEATHER = [{"name": "get_weather", "arguments": {"city": "Hanoi"}}]
TIME = [{"name": "get_time", "arguments": {"city": "Hanoi"}}]


def item(n, group, question, calls=(), set_name="t_seen", **fields):
    return {"id": f"{set_name}:{n}", "set": set_name, "group": group, "tools": TOOLS, "meta": {},
            "turns": [{"role": "user", "content": question}], "expected_calls": list(calls), **fields}


def record(entry, output, stop="im_end", tokens=10):
    return {"id": entry["id"], "set": entry["set"], "group": entry["group"], "completion": output,
            "generated_tokens": tokens, "stop_reason": stop, "scores": score(entry, output, stop)}


ITEMS = [item(0, "single", "Weather in Hanoi?", WEATHER), item(1, "single", "Time in Hanoi?", TIME),
         item(2, "select", "Hanoi weather please", WEATHER), item(3, "no_tool", "Capital of France?"),
         item(4, "no_tool", "Who wrote Hamlet?")]
OUTPUTS = [CALL, "It is noon.", CALL, CALL, "Shakespeare."]


class TermTests(unittest.TestCase):
    def test_point_estimates_match_run_eval_aggregate(self):
        records = [record(entry, output) for entry, output in zip(ITEMS, OUTPUTS)]
        expected = aggregate(records)["t_seen"]
        num, den = term_arrays(records)
        tool = metric_values(num.sum(0), den.sum(0), TOOL)
        for key in ("format_valid", "call_exact", "tool_name_acc", "false_call_rate", "decision_bacc",
                    "hallucinated_tool", "args_f1"):
            self.assertAlmostEqual(float(tool[key]), expected["tool"][key], msg=key)
        behaviour = metric_values(num.sum(0), den.sum(0), BEHAVIOUR)
        self.assertAlmostEqual(float(behaviour["stop_rate"]), expected["behaviour"]["stop_rate"])
        self.assertAlmostEqual(float(behaviour["avg_len"]), expected["behaviour"]["avg_len"])

    def test_metric_without_samples_of_its_kind_is_nan(self):
        records = [record(ITEMS[3], "Paris.")]
        num, den = term_arrays(records)
        self.assertTrue(np.isnan(metric_values(num.sum(0), den.sum(0), ("call_exact",))["call_exact"]))


class ClusterTests(unittest.TestCase):
    def test_shared_called_tool_or_prompt_links_items(self):
        items = {entry["id"]: entry for entry in ITEMS}
        labels = clusters(ITEMS, items)
        self.assertEqual(labels[0], labels[2])        # both call get_weather
        self.assertNotEqual(labels[0], labels[1])
        self.assertNotEqual(labels[3], labels[4])     # No-tool's offered tools do not link
        repeat = item(5, "no_tool", "  capital of france? ")
        self.assertEqual(len(set(clusters([ITEMS[3], repeat], items))), 1)

    def test_rename_takes_the_links_of_its_original(self):
        renamed = [item(n, "single", entry["turns"][0]["content"] + " (renamed)",
                        [{"name": f"fn_{n}", "arguments": {}}], set_name="t_rename_nonsense",
                        meta={"renamed_from": entry["id"]}) for n, entry in enumerate(ITEMS[:3])]
        items = {entry["id"]: entry for entry in ITEMS + renamed}
        labels = clusters(renamed, items)
        self.assertEqual(labels[0], labels[2])
        self.assertNotEqual(labels[0], labels[1])

    def test_weights_draw_whole_clusters_and_repeat_per_population(self):
        labels = np.array([0, 0, 1, 2])
        weights = resample_weights(labels, "t_seen/single", reps=50)
        self.assertTrue(np.array_equal(weights[:, 0], weights[:, 1]))
        self.assertTrue(np.allclose(weights[:, [0, 2, 3]].sum(1), 3))   # 3 clusters drawn per replicate
        self.assertTrue(np.array_equal(weights, resample_weights(labels, "t_seen/single", reps=50)))


class EmergenceTests(unittest.TestCase):
    def test_first_held_needs_two_following_checkpoints(self):
        self.assertEqual(first_held([False, True, True, True, False]), (1, "reached"))
        self.assertEqual(first_held([False, True, False, True, True]), (3, "unconfirmed"))
        self.assertEqual(first_held([False, False, False, False, True]), (4, "unconfirmed"))
        self.assertEqual(first_held([True, False, True, False]), (None, "not_reached"))


class EvaluateTests(unittest.TestCase):
    def checkpoint(self, name, step, outputs):
        return {"run": "e0" if step == 0 else "run", "checkpoint": name, "seed": 0, "step": step,
                "tokens_seen": step, "assistant_tokens_seen": step,
                "records": {entry["id"]: record(entry, output) for entry, output in zip(ITEMS, outputs)}}

    def test_identical_checkpoints_have_zero_difference(self):
        silent = ["Sure."] * len(ITEMS)
        checkpoints = [self.checkpoint("base", 0, silent), self.checkpoint("step-1", 1, silent)]
        rows, diffs = evaluate(checkpoints, ITEMS, "e0/base", reps=100)
        self.assertEqual(len(rows), len(diffs))
        self.assertTrue(all(row["diff"] == row["diff_low"] == row["diff_high"] == 0 for row in diffs))

    def test_go_no_go_reads_the_last_checkpoint(self):
        good = [CALL, CALL, CALL, "Paris.", "Shakespeare."]
        checkpoints = [self.checkpoint("base", 0, ["Sure."] * len(ITEMS)), self.checkpoint("step-1", 1, good)]
        _, diffs = evaluate(checkpoints, ITEMS, "e0/base", reps=100)
        # go_no_go wants the hard groups too; add them as rows that never move.
        for eval_set, group, metric in (("t_unseen", "single", "call_exact"),
                                        ("t_seen", "missing_info", "clarify_correct"),
                                        ("t_unseen", "missing_info", "clarify_correct"),
                                        ("t_seen", "multi_turn", "grounded_correct"),
                                        ("t_unseen", "multi_turn", "grounded_correct")):
            diffs.append({"run": "run", "step": 1, "eval_set": eval_set, "group": group, "metric": metric,
                          "value": 0, "base_value": 0, "diff": 0, "diff_low": 0, "diff_high": 0})
        result = go_no_go(diffs, "run")
        self.assertEqual(result["core"][0]["diff"], 1.0)    # format_valid of Single: 0 -> 1
        self.assertFalse(result["b"])


class RenameTests(unittest.TestCase):
    RENAMED = [item(n, entry["group"], entry["turns"][0]["content"], entry["expected_calls"],
                    set_name="t_rename_nonsense", meta={"renamed_from": entry["id"]})
               for n, entry in enumerate(ITEMS[:3])]

    def checkpoint(self, seen_outputs, renamed_outputs):
        records = {entry["id"]: record(entry, output) for entry, output in zip(ITEMS, seen_outputs)}
        records.update({entry["id"]: record(entry, output) for entry, output in zip(self.RENAMED, renamed_outputs)})
        return {"run": "run", "checkpoint": "step-1", "seed": 0, "step": 1, "tokens_seen": 1,
                "assistant_tokens_seen": 1, "records": records}

    def rows(self, renamed_outputs):
        found = rename_differences([self.checkpoint(OUTPUTS, renamed_outputs)], ITEMS + self.RENAMED, reps=100)
        return {(row["group"], row["metric"]): row for row in found}

    def test_difference_is_paired_with_the_originals_only(self):
        row = self.rows(["Sure."] * 3)[("tool", "call_exact")]
        self.assertAlmostEqual(row["seen_value"], 2 / 3)    # the originals' own T-seen outputs
        self.assertEqual(row["value"], 0)
        self.assertAlmostEqual(row["diff"], -2 / 3)
        self.assertEqual(self.rows(["Sure."] * 3)[("select", "call_exact")]["diff"], -1.0)

    def test_same_outputs_give_a_zero_width_interval(self):
        row = self.rows(OUTPUTS[:3])[("tool", "call_exact")]
        self.assertEqual((row["diff"], row["diff_low"], row["diff_high"]), (0, 0, 0))


class CompareTests(unittest.TestCase):
    def checkpoint(self, run, step, target, outputs):
        return {"run": run, "checkpoint": f"step-{step}", "seed": 0, "step": step, "tokens_seen": step,
                "assistant_tokens_seen": step, "target_tokens": target,
                "records": {entry["id"]: record(entry, output) for entry, output in zip(ITEMS, outputs)}}

    def rows(self, last_a):
        silent = ["Sure."] * len(ITEMS)
        checkpoints = [self.checkpoint("a", 4, 25000, silent), self.checkpoint("a", 9, None, silent),   # boundary
                       self.checkpoint("a", 12, None, last_a),
                       self.checkpoint("b", 5, 25000, silent), self.checkpoint("b", 12, None, OUTPUTS)]
        return run_differences(checkpoints, ITEMS, "a", "b", reps=100)

    def test_pairs_shared_rungs_and_the_last_step_only(self):
        found = self.rows(OUTPUTS)
        self.assertEqual({(row["snapshot"], row["step_a"], row["step_b"]) for row in found},
                         {("25000", 4, 5), ("last", 12, 12)})

    def test_difference_is_paired_prompt_by_prompt(self):
        same = [row for row in self.rows(OUTPUTS) if row["snapshot"] == "last"]
        self.assertTrue(all(row["diff"] == row["diff_low"] == row["diff_high"] == 0 for row in same))
        at = {(row["snapshot"], row["eval_set"], row["group"], row["metric"]): row
              for row in self.rows(["Sure."] * len(ITEMS))}
        row = at[("last", "t_seen", "tool", "call_exact")]
        self.assertAlmostEqual(row["diff"], -2 / 3)          # a says "Sure." where b calls right on 2 of 3
        self.assertTrue(row["diff_low"] < row["diff"] < row["diff_high"] <= 0)


if __name__ == "__main__":
    unittest.main()
