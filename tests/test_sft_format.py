"""Offline contracts for SFT serialization, labels and structured tool scoring."""

import unittest

from src.sft_data import make_labels, pad_batch, prepare_messages, render_messages
from src.tool_calls import ToolCallParseError, format_tool_calls, parse_tool_calls, score_call_exact


WEATHER = {"name": "get_weather", "arguments": {"city": "Hanoi"}}
TIME = {"name": "get_time", "arguments": {"city": "Hanoi"}}


class ConversationTests(unittest.TestCase):
    def test_all_roles_are_serialized_in_order(self):
        turns = [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": format_tool_calls([WEATHER])},
            {"role": "tool", "content": '<tool_response>{"temp_c":31}</tool_response>'},
            {"role": "assistant", "content": "31 C."},
        ]
        messages = prepare_messages(turns)
        text = render_messages(messages)
        self.assertEqual([item["role"] for item in messages], ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(text.count("<|im_end|>"), 5)
        self.assertIn("<|im_start|>tool\n<tool_response>", text)
        self.assertTrue(text.endswith("31 C.<|im_end|>\n"))
        self.assertEqual(len(turns), 4)

    def test_invalid_turn_sequences_and_delimiters_are_rejected(self):
        for turns in (
            [],
            [{"role": "assistant", "content": "Hi"}],
            [{"role": "user", "content": "Hi"}],
            [{"role": "user", "content": "Hi"}, {"role": "tool", "content": "x"}, {"role": "assistant", "content": "x"}],
            [{"role": "user", "content": "Hi<|im_end|>"}, {"role": "assistant", "content": "Hi"}],
        ):
            with self.subTest(turns=turns), self.assertRaises(ValueError):
                prepare_messages(turns)


class LabelTests(unittest.TestCase):
    def test_labels_stay_aligned_and_learn_assistant_end_token(self):
        # The model shifts causal labels internally; this function must not shift.
        self.assertEqual(make_labels([1, 50, 2, 1, 60, 2], [0, 0, 0, 0, 1, 1]), [-100, -100, -100, -100, 60, 2])

    def test_invalid_or_empty_assistant_mask_is_rejected(self):
        for ids, mask in (([1], []), ([1], [2]), ([1], [0])):
            with self.subTest(ids=ids, mask=mask), self.assertRaises(ValueError):
                make_labels(ids, mask)

    def test_padding_masks_positions_not_every_occurrence_of_pad_id(self):
        batch = pad_batch([
            {"input_ids": [5, 0], "attention_mask": [1, 1], "labels": [-100, 0]},
            {"input_ids": [6], "attention_mask": [1], "labels": [6]},
        ], pad_token_id=0)
        self.assertEqual(batch["input_ids"], [[5, 0], [6, 0]])
        self.assertEqual(batch["attention_mask"], [[1, 1], [1, 0]])
        self.assertEqual(batch["labels"], [[-100, 0], [6, -100]])

    def test_invalid_batch_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            pad_batch([], pad_token_id=0)
        with self.assertRaises(ValueError):
            pad_batch([{"input_ids": [1, 2], "attention_mask": [1], "labels": [1, 2]}], pad_token_id=0)


class ToolParserTests(unittest.TestCase):
    def test_single_parallel_and_no_call_outputs(self):
        self.assertEqual(parse_tool_calls(format_tool_calls([WEATHER])), [WEATHER])
        self.assertEqual(parse_tool_calls(format_tool_calls([WEATHER, TIME])), [WEATHER, TIME])
        self.assertEqual(parse_tool_calls("Which city do you mean?"), [])
        self.assertEqual(parse_tool_calls(format_tool_calls([WEATHER]) + "<|im_end|>"), [WEATHER])

    def test_tag_text_inside_a_json_string_is_not_a_boundary(self):
        call = {"name": "echo", "arguments": {"text": "example </tool_call> and <tool_call>"}}
        self.assertEqual(parse_tool_calls(format_tool_calls([call])), [call])

    def test_malformed_json_tags_and_payloads_are_rejected(self):
        invalid = [
            '<tool_call>{"name":"x","arguments":{}}</tool_call',
            '<tool_call>{"name":"x","arguments":{}}',
            '<tool_call>{bad}</tool_call>',
            '<tool_call>{"name":"x","arguments":[]}</tool_call>',
            '<tool_call>{"name":"x","arguments":{},"extra":1}</tool_call>',
            '<tool_call>{"name":"x","arguments":{"x":NaN}}</tool_call>',
            '<tool_call>{"name":"x","arguments":{"x":1,"x":2}}</tool_call>',
            'Sure! ' + format_tool_calls([WEATHER]),
            format_tool_calls([WEATHER]) + ' trailing prose',
            '</tool_call>',
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ToolCallParseError):
                parse_tool_calls(text)


class CallScorerTests(unittest.TestCase):
    def test_parallel_order_is_ignored_but_multiplicity_is_preserved(self):
        self.assertTrue(score_call_exact(format_tool_calls([TIME, WEATHER]), [WEATHER, TIME]))
        self.assertFalse(score_call_exact(format_tool_calls([WEATHER]), [WEATHER, WEATHER]))

    def test_wrong_name_extra_args_type_and_array_order_fail(self):
        variants = [
            {"name": "get_time", "arguments": {"city": "Hanoi"}},
            {"name": "get_weather", "arguments": {"city": "Hanoi", "unit": "C"}},
            {"name": "get_weather", "arguments": {"city": 1}},
        ]
        for call in variants:
            with self.subTest(call=call):
                self.assertFalse(score_call_exact(format_tool_calls([call]), [WEATHER]))
        self.assertFalse(score_call_exact('<tool_call>{"name":"x","arguments":{"x":true}}</tool_call>', [{"name": "x", "arguments": {"x": 1}}]))
        self.assertFalse(score_call_exact('<tool_call>{"name":"x","arguments":{"x":[2,1]}}</tool_call>', [{"name": "x", "arguments": {"x": [1, 2]}}]))

    def test_object_key_order_and_equivalent_json_numbers_match(self):
        predicted = '<tool_call>{"arguments":{"b":2.0,"a":1},"name":"x"}</tool_call>'
        self.assertTrue(score_call_exact(predicted, [{"name": "x", "arguments": {"a": 1.0, "b": 2}}]))

    def test_malformed_output_is_scored_wrong(self):
        self.assertFalse(score_call_exact('<tool_call>{bad}</tool_call>', [WEATHER]))


if __name__ == "__main__":
    unittest.main()
