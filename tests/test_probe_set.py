import json
from collections import Counter
from pathlib import Path
import unittest

from src.eval.probe_set import build, check
from src.eval.scorers import score_clarify, score_grounded, score_no_fitting, score_tool_sample
from src.tool_calls import format_tool_calls


class ProbeSetTests(unittest.TestCase):
    def test_sizes_follow_the_plan(self):
        groups = Counter(item["group"] for item in build())
        self.assertEqual(groups["chat"], 10)
        self.assertEqual(groups["knowledge"], 10)
        self.assertEqual(sum(groups[g] for g in ("single", "select", "parallel", "no_tool", "missing_info", "multi_turn")), 40)

    def test_items_are_consistent_and_reference_answers_score_right(self):
        items = build()
        check(items)
        for item in items:
            with self.subTest(item=item["id"]):
                if item["expected_calls"]:
                    self.assertTrue(score_tool_sample(format_tool_calls(item["expected_calls"]),
                                                      expected_calls=item["expected_calls"], tools=item["tools"])["call_exact"])
                if item["group"] == "multi_turn":
                    self.assertTrue(score_grounded(item["reference"], required=item["required"], forbidden=[])["grounded_rule"])
                if item["group"] == "missing_info":
                    question = f"Sure, what is the {item['synonyms'][0]}?"
                    self.assertTrue(score_clarify(question, missing_param=item["missing_param"],
                                                  synonyms=item["synonyms"])["clarify_correct"])
                if item.get("kind") == "no_fitting_tool":
                    self.assertTrue(score_no_fitting(item["reference"])["no_fitting_said"])

    def test_committed_file_matches_the_definitions(self):
        lines = Path("eval_sets/p.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line) for line in lines], build())


if __name__ == "__main__":
    unittest.main()
