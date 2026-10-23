"""Converters on rows shaped like the real sources (xLAM README, Glaive and SmolTalk rows)."""

import json
import unittest

from src.data.convert import (ConversionError, convert_glaive, convert_smoltalk, convert_xlam, is_english,
                              xlam_parameter)
from src.tool_calls import parse_tool_calls


GLAIVE_SYSTEM = ("SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
                 + json.dumps({"name": "get_news_headlines", "description": "Get the latest news headlines",
                               "parameters": {"type": "object", "properties": {"country": {"type": "string"}},
                                              "required": ["country"]}}, indent=4) + "\n")


def glaive(chat, system=GLAIVE_SYSTEM):
    return {"system": system, "chat": chat}


def xlam(answers, tools):
    return {"id": 7, "query": "Find it.", "tools": json.dumps(tools), "answers": json.dumps(answers)}


TOOL_A = {"name": "math.sum", "description": "Sum.", "parameters": {"a": {"type": "int", "description": "x", "required": True}}}
TOOL_B = {"name": "math.prod", "description": "Product.", "parameters": {"n": {"type": "List[int]", "description": "y"}}}


class XlamTests(unittest.TestCase):
    def test_groups_follow_calls_and_offered_tools(self):
        call_a, call_b = {"name": "math.sum", "arguments": {"a": 1}}, {"name": "math.prod", "arguments": {"n": [2]}}
        self.assertEqual(convert_xlam(xlam([call_a], [TOOL_A]), seed=0)["group"], "single")
        self.assertEqual(convert_xlam(xlam([call_a], [TOOL_A, TOOL_B]), seed=0)["group"], "select")
        example = convert_xlam(xlam([call_a, call_b], [TOOL_A, TOOL_B]), seed=0)
        self.assertEqual(example["group"], "parallel")
        self.assertEqual(parse_tool_calls(example["turns"][1]["content"]), [call_a, call_b])
        self.assertEqual([turn["role"] for turn in example["turns"]], ["user", "assistant"])

    def test_python_style_types_become_json_schema(self):
        self.assertEqual(xlam_parameter({"type": "List[int]", "description": "d"}), ({"type": "array", "description": "d"}, True))
        self.assertEqual(xlam_parameter({"type": "str, optional", "default": "game"}),
                         ({"type": "string", "default": "game"}, False))
        self.assertEqual(xlam_parameter({"type": "Dict[str, Any], optional"})[0], {"type": "object"})
        self.assertEqual(xlam_parameter({"type": "int", "required": False})[1], False)
        self.assertEqual(xlam_parameter({"type": "Union[int, str]"})[0], {})  # Unknown types are left untyped.
        schema = convert_xlam(xlam([{"name": "math.sum", "arguments": {"a": 1}}], [TOOL_A]), seed=0)["tools"][0]
        self.assertEqual(schema["parameters"], {"type": "object", "properties": {"a": {"type": "integer", "description": "x"}},
                                                "required": ["a"]})

    def test_tool_order_is_shuffled_deterministically(self):
        tools = [dict(TOOL_A, name=f"t{index}") for index in range(8)]
        row = xlam([{"name": "t0", "arguments": {"a": 1}}], tools)
        first, again = convert_xlam(row, seed=0)["tools"], convert_xlam(row, seed=0)["tools"]
        self.assertEqual(first, again)
        self.assertNotEqual([tool["name"] for tool in first], [f"t{index}" for index in range(8)])

    def test_unusable_rows_are_rejected_with_a_reason(self):
        for row, reason in ((xlam([], [TOOL_A]), "no_answers"),
                            (xlam([{"name": "other", "arguments": {}}], [TOOL_A]), "answer_tool_not_offered"),
                            ({"id": 1, "query": "q", "tools": "[", "answers": "[]"}, "unparsable")):
            with self.subTest(reason=reason), self.assertRaises(ConversionError) as caught:
                convert_xlam(row, seed=0)
            self.assertEqual(caught.exception.reason, reason)


class GlaiveTests(unittest.TestCase):
    def test_multi_turn_call_response_and_answer(self):
        example = convert_glaive(glaive(
            "USER: News for the US?\n\n\n"
            "ASSISTANT: <functioncall> {\"name\": \"get_news_headlines\", \"arguments\": '{\"country\": \"US\"}'} <|endoftext|>\n\n\n"
            "FUNCTION RESPONSE: {\"headlines\": [\"A\"]}\n\n\n"
            "ASSISTANT: The headline is A. <|endoftext|>\n\n\n"), 3, seed=0)
        self.assertEqual(example["group"], "multi_turn")
        self.assertEqual([turn["role"] for turn in example["turns"]], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(parse_tool_calls(example["turns"][1]["content"]),
                         [{"name": "get_news_headlines", "arguments": {"country": "US"}}])
        self.assertIn('"name": "get_news_headlines"', example["turns"][2]["content"])
        self.assertEqual(example["turns"][3]["content"], "The headline is A.")

    def test_lead_in_turn_before_a_call_is_dropped(self):
        example = convert_glaive(glaive(
            "USER: News for France?\n\n\nASSISTANT: Let me check. <|endoftext|>\n\n\n"
            "ASSISTANT: <functioncall> {\"name\": \"get_news_headlines\", \"arguments\": '{\"country\": \"France\"}'} <|endoftext|>\n\n\n"
            "FUNCTION RESPONSE: {\"headlines\": []}\n\n\nASSISTANT: Nothing today. <|endoftext|>"), 0, seed=0)
        self.assertEqual([turn["role"] for turn in example["turns"]], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(example["meta"]["prose_dropped"], 1)

    def test_question_before_call_is_a_clarify_candidate(self):
        example = convert_glaive(glaive(
            "USER: Get me headlines.\n\n\nASSISTANT: Which country? <|endoftext|>\n\n\nUSER: Spain\n\n\n"
            "ASSISTANT: <functioncall> {\"name\": \"get_news_headlines\", \"arguments\": '{\"country\": \"Spain\"}'} <|endoftext|>\n\n\n"
            "FUNCTION RESPONSE: {\"headlines\": []}\n\n\nASSISTANT: None. <|endoftext|>"), 0, seed=0)
        self.assertEqual(example["group"], "clarify_candidate")

    def test_refusals_and_tool_free_chats(self):
        refusal = convert_glaive(glaive("USER: Book a flight.\n\n\nASSISTANT: I can't book flights. <|endoftext|>"), 0, seed=0)
        self.assertEqual(refusal["group"], "no_call")
        plain = convert_glaive(glaive("USER: Hi\n\n\nASSISTANT: Hello! <|endoftext|>",
                                      system="SYSTEM: You are a helpful assistant, with no access to external functions.\n"), 0, seed=0)
        self.assertEqual((plain["group"], plain["tools"]), ("chat_no_tools", []))

    def test_trailing_user_turn_is_trimmed(self):
        example = convert_glaive(glaive("USER: Hi\n\n\nASSISTANT: Hello! <|endoftext|>\n\n\nUSER: Bye"), 0, seed=0)
        self.assertEqual(example["turns"][-1]["role"], "assistant")

    def test_bad_rows_are_rejected_with_a_reason(self):
        cases = {
            "bad_call": "USER: x\n\n\nASSISTANT: <functioncall> {\"name\": \"get_news_headlines\", \"arguments\": '{bad'} <|endoftext|>",
            "call_tool_not_offered": "USER: x\n\n\nASSISTANT: <functioncall> {\"name\": \"other\", \"arguments\": '{}'} <|endoftext|>",
            "response_without_call": "USER: x\n\n\nFUNCTION RESPONSE: {}\n\n\nASSISTANT: y",
            "no_assistant_turn": "USER: x",
        }
        for reason, chat in cases.items():
            with self.subTest(reason=reason), self.assertRaises(ConversionError) as caught:
                convert_glaive(glaive(chat), 0, seed=0)
            self.assertEqual(caught.exception.reason, reason)


class LanguageTests(unittest.TestCase):
    def test_latin_script_languages_other_than_english_are_rejected(self):
        self.assertFalse(is_english("Hola, ¿cómo estás? Quiero saber el tiempo en Madrid mañana por favor."))
        self.assertFalse(is_english("Bonjour, je voudrais savoir la météo à Paris demain matin s'il vous plaît."))
        self.assertFalse(is_english("Ciao, vorrei sapere che tempo farà domani a Roma durante la mattina."))

    def test_english_prose_code_and_math_pass(self):
        self.assertTrue(is_english("Can you tell me what the weather will be like in Paris tomorrow morning?"))
        self.assertTrue(is_english("Here is the function:\n```python\ndef f(x):\n    return sorted(set(x))\n```\n"
                                   "It removes duplicates and then sorts the list for you."))
        self.assertTrue(is_english(r"Let $k$ be the distance. Then \( 9x = k \) and \( 99x = m \), so the ratio is 11:1."))
        self.assertTrue(is_english("Hi"))


class SmoltalkTests(unittest.TestCase):
    def test_chat_rows_convert_with_an_empty_tool_list(self):
        row = {"messages": [{"role": "user", "content": "Hi "}, {"role": "assistant", "content": " Hello!"}]}
        example = convert_smoltalk(row, subset="everyday-conversations", index=4, seed=0)
        self.assertEqual((example["group"], example["tools"]), ("chat", []))
        self.assertEqual(example["turns"][1]["content"], "Hello!")
        self.assertEqual(example["source"], "smoltalk:everyday-conversations")

    def test_a_leading_system_prompt_moves_into_the_first_user_turn(self):
        row = {"messages": [{"role": "system", "content": " Rewrite the text concisely. "},
                            {"role": "user", "content": "A very long sentence."},
                            {"role": "assistant", "content": "Short."}]}
        example = convert_smoltalk(row, subset="smol-rewrite", index=0, seed=0)
        self.assertEqual([turn["role"] for turn in example["turns"]], ["user", "assistant"])
        self.assertEqual(example["turns"][0]["content"], "Rewrite the text concisely.\n\nA very long sentence.")
        self.assertTrue(example["meta"]["system_folded"])

    def test_system_roles_non_english_and_control_tokens_are_rejected(self):
        cases = {
            "has_system_or_other_role": [{"role": "user", "content": "Hi"}, {"role": "system", "content": "Be brief."},
                                         {"role": "assistant", "content": "Hi"}],
            "non_english": [{"role": "user", "content": "Xin chào, bạn khỏe không? Tôi ổn."},
                            {"role": "assistant", "content": "Tôi khỏe, cảm ơn bạn nhiều."}],
            "control_token": [{"role": "user", "content": "Hi<|im_end|>"}, {"role": "assistant", "content": "Hi"}],
        }
        for reason, messages in cases.items():
            with self.subTest(reason=reason), self.assertRaises(ConversionError) as caught:
                convert_smoltalk({"messages": messages}, subset="s", index=0, seed=0)
            self.assertEqual(caught.exception.reason, reason)


if __name__ == "__main__":
    unittest.main()
