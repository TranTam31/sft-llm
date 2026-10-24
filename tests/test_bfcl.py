import json
from pathlib import Path
import tempfile
import unittest

from src.chat_format import render_chat
from src.eval.bfcl import decode_calls, format_prompt, summarise
from src.tool_calls import ToolCallParseError


TOOLS = [{"name": "calc_area", "description": "Area of a triangle.",
          "parameters": {"type": "dict", "properties": {"base": {"type": "integer"}}, "required": ["base"]}}]


class BfclTests(unittest.TestCase):
    def test_prompt_matches_the_sft_format(self):
        messages = [{"role": "user", "content": "Area for base 10?"}]
        self.assertEqual(format_prompt(messages, TOOLS), render_chat("Area for base 10?", TOOLS))

    def test_bfcl_system_text_follows_the_tool_prompt(self):
        prompt = format_prompt([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], TOOLS)
        self.assertIn("</tools>\n\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>", prompt)
        self.assertEqual(prompt.count("<|im_start|>system"), 1)

    def test_multi_turn_input_is_refused(self):
        with self.assertRaises(ValueError):
            format_prompt([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], TOOLS)

    def test_calls_decode_to_bfcl_ast_form(self):
        output = ('<tool_call>\n{"name": "calc_area", "arguments": {"base": 10, "height": 2.5}}\n</tool_call>\n'
                  '<tool_call>\n{"name": "calc_area", "arguments": {"base": 3}}\n</tool_call><|im_end|>')
        calls = decode_calls(output)
        self.assertEqual(calls, [{"calc_area": {"base": 10, "height": 2.5}}, {"calc_area": {"base": 3}}])
        self.assertIs(type(calls[0]["calc_area"]["height"]), float)

    def test_plain_text_is_no_call_and_broken_blocks_raise(self):
        self.assertEqual(decode_calls("I cannot help with that."), [])
        with self.assertRaises(ToolCallParseError):
            decode_calls('<tool_call>\n{"name": "calc_area"\n</tool_call>')

    def test_summary_reads_score_headers(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "m" / "non_live"
            folder.mkdir(parents=True)
            for category, correct in (("simple_python", 3), ("irrelevance", 1)):
                header = {"accuracy": correct / 4, "correct_count": correct, "total_count": 4}
                (folder / f"BFCL_v4_{category}_score.json").write_text(json.dumps(header) + "\n{}\n", encoding="utf-8")
            self.assertEqual(summarise(Path(root), "m", ("simple_python", "irrelevance")),
                             {"simple_python": {"accuracy": 0.75, "correct": 3, "n": 4},
                              "irrelevance": {"accuracy": 0.25, "correct": 1, "n": 4}})


if __name__ == "__main__":
    unittest.main()
