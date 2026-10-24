"""Scorer sensitivity on handcrafted outputs: every known error scores wrong, correct output scores right."""

from itertools import permutations
import random
import unittest

from src.eval.scorers import (aggregate_grounded, aggregate_tool_group, extract_tool_calls, max_weight_assignment,
                              score_behavior, score_clarify, score_grounded, score_no_fitting, score_tool_sample)
from src.tool_calls import format_tool_calls


WEATHER = {"name": "get_weather", "arguments": {"city": "Hanoi", "unit": "celsius"}}
TIME = {"name": "get_time", "arguments": {"city": "Hanoi"}}
TOOLS = [{"name": "get_weather"}, {"name": "get_time"}]


def score(output, expected=(WEATHER,)):
    return score_tool_sample(output, expected_calls=list(expected), tools=TOOLS)


class ExtractionTests(unittest.TestCase):
    def test_prose_around_calls_is_still_a_call(self):
        extraction = extract_tool_calls("Sure, let me check.\n" + format_tool_calls([WEATHER]) + "\nOne moment.")
        self.assertEqual(extraction.calls, [WEATHER])
        self.assertEqual(extraction.invalid_blocks, 0)
        self.assertTrue(extraction.prose_outside)

    def test_broken_blocks_are_counted_and_valid_ones_kept(self):
        text = format_tool_calls([WEATHER]) + "\n<tool_call>{bad}</tool_call>\n<tool_call>" + '{"name": "x", "arguments": {}}'
        extraction = extract_tool_calls(text)
        self.assertEqual(extraction.calls, [WEATHER])
        self.assertEqual(extraction.invalid_blocks, 2)

    def test_stray_tags_and_plain_answers(self):
        self.assertEqual(extract_tool_calls("</tool_call>").invalid_blocks, 1)
        self.assertEqual(extract_tool_calls("<tool_call").invalid_blocks, 1)
        plain = extract_tool_calls("It is sunny.<|im_end|>")
        self.assertFalse(plain.has_marker)
        self.assertEqual((plain.calls, plain.invalid_blocks), ([], 0))


class ToolSampleTests(unittest.TestCase):
    def test_correct_outputs_score_right(self):
        for output in (format_tool_calls([WEATHER]), format_tool_calls([WEATHER]) + "<|im_end|>",
                       "Let me look that up. " + format_tool_calls([WEATHER])):
            with self.subTest(output=output):
                result = score(output)
                self.assertTrue(result["format_valid"] and result["call_exact"] and result["tool_name_acc"])
                self.assertEqual(result["args_f1"], 1.0)
        self.assertTrue(score(format_tool_calls([WEATHER])) ["strict_format"])
        self.assertFalse(score("Let me look that up. " + format_tool_calls([WEATHER]))["strict_format"])

    def test_parallel_calls_ignore_order(self):
        self.assertTrue(score(format_tool_calls([TIME, WEATHER]), [WEATHER, TIME])["call_exact"])

    def test_each_known_error_scores_wrong(self):
        errors = {
            "wrong tool name": [{"name": "get_time", "arguments": WEATHER["arguments"]}],
            "missing argument": [{"name": "get_weather", "arguments": {"city": "Hanoi"}}],
            "extra argument": [{"name": "get_weather", "arguments": {**WEATHER["arguments"], "days": 3}}],
            "wrong value": [{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}],
            "wrong type": [{"name": "get_weather", "arguments": {"city": "Hanoi", "unit": 1}}],
            "missing parallel call": [WEATHER],
            "duplicated call": [WEATHER, WEATHER],
        }
        for label, calls in errors.items():
            with self.subTest(error=label):
                expected = [WEATHER, TIME] if label == "missing parallel call" else [WEATHER]
                self.assertFalse(score(format_tool_calls(calls), expected)["call_exact"])

    def test_partial_arguments_get_partial_f1(self):
        result = score(format_tool_calls([{"name": "get_weather", "arguments": {"city": "Hanoi"}}]))
        self.assertTrue(result["tool_name_acc"])
        self.assertAlmostEqual(result["args_f1"], 2 / 3)

    def test_values_swapped_between_parallel_calls_lose_f1(self):
        expected = [{"name": "f", "arguments": {"a": 1, "b": 2}}, {"name": "f", "arguments": {"a": 3, "b": 4}}]
        swapped = [{"name": "f", "arguments": {"a": 1, "b": 4}}, {"name": "f", "arguments": {"a": 3, "b": 2}}]
        tools = [{"name": "f", "parameters": {"properties": {"a": {}, "b": {}}}}]
        self.assertAlmostEqual(score_tool_sample(format_tool_calls(swapped), expected_calls=expected,
                                                 tools=tools)["args_f1"], 0.5)
        reordered = score_tool_sample(format_tool_calls(expected[::-1]), expected_calls=expected, tools=tools)
        self.assertEqual(reordered["args_f1"], 1.0)

    def test_matching_is_exact_and_order_free_beyond_seven_calls(self):
        rng = random.Random(0)
        for _ in range(200):
            rows, columns = rng.randint(1, 5), rng.randint(5, 6)
            matrix = [[rng.randint(0, 3) for _ in range(columns)] for _ in range(rows)]
            brute = max(sum(matrix[r][c] for r, c in enumerate(order)) for order in permutations(range(columns), rows))
            self.assertEqual(max_weight_assignment(matrix), brute)
        expected = [{"name": "f", "arguments": {"a": i, "b": i}} for i in range(9)]
        predicted = [{"name": "f", "arguments": {"a": i, "b": (i + 1) % 9}} for i in range(9)]
        scores = {score_tool_sample(format_tool_calls(order), expected_calls=expected, tools=[])["args_f1"]
                  for order in (predicted, predicted[::-1], predicted[1::-1] + predicted[2:])}
        self.assertEqual(scores, {0.5})

    def test_broken_json_fails_format_and_everything_after(self):
        result = score('<tool_call>{"name": "get_weather", "arguments": {"city": "Hanoi",}}</tool_call>')
        self.assertTrue(result["called"])
        self.assertFalse(result["format_valid"] or result["call_exact"] or result["tool_name_acc"])
        self.assertEqual(result["args_f1"], 0.0)

    def test_one_broken_block_among_valid_ones_fails_format(self):
        result = score(format_tool_calls([WEATHER]) + "<tool_call>{bad}</tool_call>")
        self.assertFalse(result["format_valid"])
        self.assertFalse(result["call_exact"])

    def test_answering_without_calling_is_a_missed_call(self):
        result = score("The weather in Hanoi is 25 C.")
        self.assertTrue(result["missed_call"])
        self.assertFalse(result["called"] or result["format_valid"] or result["call_exact"])

    def test_calling_when_not_needed_is_a_false_call(self):
        result = score(format_tool_calls([WEATHER]), expected=())
        self.assertTrue(result["false_call"])
        self.assertFalse(result["call_exact"])
        self.assertFalse(score("2 + 2 = 4.", expected=())["false_call"])

    def test_hallucinated_tool_is_flagged(self):
        result = score(format_tool_calls([{"name": "search_web", "arguments": {"q": "Hanoi"}}]))
        self.assertTrue(result["hallucinated_tool"])
        self.assertFalse(score(format_tool_calls([WEATHER]))["hallucinated_tool"])


class GroupTests(unittest.TestCase):
    def test_denominators_include_every_sample_and_bacc_is_chance_for_constant_policies(self):
        never = [score("No idea."), score("No idea."), score("4.", expected=()), score("4.", expected=())]
        always = [score(format_tool_calls([WEATHER])), score(format_tool_calls([WEATHER])),
                  score(format_tool_calls([WEATHER]), expected=()), score(format_tool_calls([WEATHER]), expected=())]
        self.assertEqual(aggregate_tool_group(never)["decision_bacc"], 0.5)
        self.assertEqual(aggregate_tool_group(always)["decision_bacc"], 0.5)
        self.assertEqual(aggregate_tool_group(never)["call_exact"], 0.0)
        mixed = aggregate_tool_group([score(format_tool_calls([WEATHER])), score("<tool_call>{bad}</tool_call>"),
                                      score("4.", expected=())])
        self.assertEqual(mixed["call_exact"], 0.5)  # The broken output stays in the denominator.
        self.assertEqual(mixed["decision_bacc"], 1.0)
        self.assertEqual((mixed["n_need_tool"], mixed["n_no_tool"]), (2, 1))


class ClarifyTests(unittest.TestCase):
    def test_asking_for_the_missing_argument_is_correct(self):
        for output in ("Which city would you like the weather for?", "What city do you mean?<|im_end|>"):
            with self.subTest(output=output):
                self.assertTrue(score_clarify(output, missing_param="city")["clarify_correct"])
        self.assertTrue(score_clarify("Which location?", missing_param="city", synonyms=["location"])["clarify_correct"])
        self.assertTrue(score_clarify("What is the start date?", missing_param="start_date")["clarify_correct"])

    def test_each_clarify_error_scores_wrong(self):
        cases = {
            "asks for the wrong argument": "Which unit would you like, celsius or fahrenheit?",
            "guesses and calls": format_tool_calls([WEATHER]) + " Which city?",
            "no question": "Please tell me the city.",
            "answers instead": "It is sunny today.",
        }
        for label, output in cases.items():
            with self.subTest(error=label):
                self.assertFalse(score_clarify(output, missing_param="city")["clarify_correct"])


class GroundedTests(unittest.TestCase):
    def test_required_values_present_and_forbidden_absent(self):
        ok = score_grounded("It is sunny and 31 C in Hanoi.", required=["31", "sunny"], forbidden=["rain"])
        self.assertTrue(ok["grounded_rule"])

    def test_each_grounding_error_scores_wrong(self):
        cases = {
            "missing value": "It is sunny in Hanoi.",
            "forbidden value": "It is sunny and 31 C, with rain later.",
            "calls again": format_tool_calls([WEATHER]),
        }
        for label, output in cases.items():
            with self.subTest(error=label):
                self.assertFalse(score_grounded(output, required=["31", "sunny"], forbidden=["rain"])["grounded_rule"])

    def test_values_match_whole_tokens_only(self):
        self.assertFalse(score_grounded("It is 131 degrees.", required=["31"], forbidden=[])["grounded_rule"])

    def test_numbers_keep_their_sign_and_decimals(self):
        self.assertFalse(score_grounded("It is -31 C today.", required=["31"], forbidden=[])["required_all"])
        self.assertTrue(score_grounded("It is -31 C today.", required=["-31"], forbidden=[])["required_all"])
        self.assertFalse(score_grounded("It is 31.5 C.", required=["31"], forbidden=[])["required_all"])
        self.assertTrue(score_grounded("Born 1990-05-15, total $1,000.", required=["05", "1000"],
                                       forbidden=[])["required_all"])

    def test_denied_values_are_not_stated(self):
        cases = {
            "not": ("It is not 31 C in Hanoi.", ["31"]),
            "contraction": ("The tip amount isn't $10.", [10]),
            "curly apostrophe": ("The tip amount isn’t $10.", [10]),
            "couldn't find": ("I couldn't find Hotel A for those dates.", ["Hotel A"]),
            "status": ("Hotel A is not available tonight.", ["Hotel A", "Available"]),
            "no": ("There are no flights to Chicago, IL today.", ["Chicago, IL"]),
            "never": ("Your package was never in transit.", ["In transit"]),
        }
        for label, (output, required) in cases.items():
            with self.subTest(case=label):
                result = score_grounded(output, required=required, forbidden=[])
                self.assertFalse(result["grounded_rule"])
                self.assertTrue(result["negated_required"])

    def test_negation_elsewhere_does_not_deny_the_value(self):
        cases = {
            "correction": ("The tip is $10, not $15.", [10]),
            "other clause": ("There is no mode, and the mean is 17.5.", [17.5]),
            "other sentence": ("No problem! The tip is $10.", [10]),
            "far before": ("I did not round anything: the exact tip amount is $10.", [10]),
            "not only": ("It is not only 31 C but also humid.", ["31"]),
            "no more than": ("It costs no more than $200 per night.", ["$200 per night"]),
            "t test": ("A paired t test gives 2.5 here.", [2.5]),
            "value holds a negation word": ('Try "And Then There Were None" next.', ["And Then There Were None"]),
            "spans a comma": ("It is now in Chicago, IL.", ["Chicago, IL"]),
            "stated once, denied once": ("It is not 30 C. It is 31 C.", ["31"]),
        }
        for label, (output, required) in cases.items():
            with self.subTest(case=label):
                self.assertTrue(score_grounded(output, required=required, forbidden=[])["grounded_rule"])
        both = score_grounded("It is not 31 C. Actually it is 31 C.", required=["31"], forbidden=[])
        self.assertTrue(both["required_all"])
        self.assertEqual(both["negated_required"], [])

    def test_a_denied_forbidden_value_is_allowed(self):
        self.assertTrue(score_grounded("It is 31 C and it is not going to rain.", required=["31"],
                                       forbidden=["rain"])["grounded_rule"])

    def test_side_score_keeps_only_samples_that_need_the_tool_result(self):
        samples = [{**score_grounded("The tip is $10.", required=[10], forbidden=[]), "derivable": True},
                   {**score_grounded("The tip is $12.", required=[10], forbidden=[]), "derivable": True},
                   {**score_grounded("It is in transit.", required=["In transit"], forbidden=[]), "derivable": False},
                   {**score_grounded("It was delivered.", required=["In transit"], forbidden=[]), "derivable": False},
                   {**score_grounded("Not in transit.", required=["In transit"], forbidden=[]), "derivable": False}]
        metrics = aggregate_grounded(samples)
        self.assertEqual(metrics["grounded_correct"], 0.4)
        self.assertAlmostEqual(metrics["grounded_correct_not_derivable"], 1 / 3)
        self.assertEqual((metrics["n"], metrics["n_not_derivable"]), (5, 3))


class NoFittingToolTests(unittest.TestCase):
    def test_saying_no_tool_fits_is_recognised(self):
        outputs = (
            "None of the tools I have can help with this request, so I can't look it up or do it for you.",
            "I'm sorry, but none of the provided functions can fetch stock data.",
            "There is no suitable tool for looking up player rankings.",
            "The available tools don't cover this, so I can't help with that request.",
            "I don't have access to real-time market data.",
            "I'm unable to access the TrackMania API with the tools provided.",
            "Unfortunately I cannot retrieve the latest data on Panasonic.<|im_end|>",
            "I don’t have a tool for that.",
        )
        for output in outputs:
            with self.subTest(output=output):
                self.assertTrue(score_no_fitting(output)["no_fitting_said"])

    def test_answering_guessing_or_calling_is_not_saying_it(self):
        cases = {
            "makes up data": "Panasonic (6752) is trading at 1,520 yen today.",
            "plain refusal": "Sorry, I won't do that.",
            "general answer": "Matchmaking rank is a number that reflects a player's skill.",
            "calls a tool anyway": format_tool_calls([WEATHER]) + " None of the tools fit.",
        }
        for label, output in cases.items():
            with self.subTest(case=label):
                self.assertFalse(score_no_fitting(output)["no_fitting_said"])

    def test_side_metric_counts_only_3b_samples(self):
        plain = score("Blue light scatters more.", expected=())
        said = {**score("No tool fits this.", expected=()), **score_no_fitting("No tool fits this.")}
        guessed = {**score("It is 1,520 yen.", expected=()), **score_no_fitting("It is 1,520 yen.")}
        metrics = aggregate_tool_group([score(format_tool_calls([WEATHER])), plain, said, guessed])
        self.assertEqual(metrics["no_fitting_said"], 0.5)
        self.assertEqual((metrics["n_no_tool"], metrics["n_no_fitting"]), (3, 2))


class BehaviorTests(unittest.TestCase):
    def test_role_break_and_repetition(self):
        self.assertTrue(score_behavior("Hi<|im_end|>\n<|im_start|>user\nmore", stop_reason="max_new_tokens")["role_break"])
        self.assertFalse(score_behavior("assistant: hi", stop_reason="im_end")["role_break"])
        looped = score_behavior("Paris is the capital. " * 8, stop_reason="max_new_tokens")
        self.assertGreater(looped["repetition_rate"], 0.5)
        self.assertTrue(looped["truncated"])
        self.assertEqual(score_behavior("Paris is the capital of France.", stop_reason="im_end")["repetition_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
