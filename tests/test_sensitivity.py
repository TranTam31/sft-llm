import unittest
from unittest import mock

from src.eval import scorers
from src.eval.sensitivity import check


TOOLS = [{"name": "get_weather", "description": "Weather now.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                         "required": ["city"]}},
         {"name": "get_time", "description": "Time now.",
          "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}, "required": ["zone"]}}]
WEATHER = {"name": "get_weather", "arguments": {"city": "Hanoi", "unit": "C"}}
TIME = {"name": "get_time", "arguments": {"zone": "UTC+7"}}


def item(n, group, **fields):
    return {"id": f"t_seen:{n}", "set": "t_seen", "group": group, "tools": TOOLS, "meta": {},
            "turns": [{"role": "user", "content": "Hi"}], "expected_calls": [], **fields}


ITEMS = [item(0, "single", expected_calls=[WEATHER]), item(1, "parallel", expected_calls=[WEATHER, TIME]),
         item(2, "no_tool", kind="plain", reference="Paris is the capital of France."),
         item(3, "missing_info", missing_param="city", synonyms=["which city", "location"]),
         item(4, "multi_turn", required=[31, "sunny"], forbidden=["rain"], meta={"derivable": False},
              reference="It is sunny and 31 C in Hanoi.")]


class SensitivityTests(unittest.TestCase):
    def test_every_variant_scores_as_expected_on_the_current_scorer(self):
        report = check(ITEMS)
        self.assertEqual({key: entry["failures"] for key, entry in report.items() if entry["failures"]}, {})
        for variant in ("wrong_name", "missing_arg", "wrong_type", "extra_arg", "dropped_call", "broken_json"):
            self.assertIn(("parallel", variant), report)
        self.assertIn(("multi_turn", "forbidden_value"), report)

    def test_a_scorer_blind_to_negation_is_caught(self):
        with mock.patch.object(scorers, "NEGATION", set()):
            report = check(ITEMS)
        self.assertTrue(report[("multi_turn", "negated_value")]["failures"])


if __name__ == "__main__":
    unittest.main()
