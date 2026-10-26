import importlib.util
import json
from pathlib import Path
import unittest

from src.chat_format import render_chat
from src.eval.run_eval import FEW_SHOT, aggregate, few_shot_prefix, limit_per_set, render, score


HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "tokenizers"))
SMOLLM2_INSTRUCT_TEMPLATE = (
    "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}"
    "{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
TOOLS = [{"name": "get_weather", "description": "Weather now.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Hanoi"}}\n</tool_call><|im_end|>'


def item(group, **fields):
    return {"id": f"x:{group}", "set": "t_seen", "group": group, "tools": TOOLS,
            "turns": [{"role": "user", "content": "Weather in Hanoi?"}], "expected_calls": [], **fields}


class RenderTests(unittest.TestCase):
    def test_single_turn_prompt_is_the_sft_prompt(self):
        self.assertEqual(render(item("single")), render_chat("Weather in Hanoi?", TOOLS))

    def test_multi_turn_prompt_ends_after_the_tool_turn(self):
        turns = [{"role": "user", "content": "Weather in Hanoi?"}, {"role": "assistant", "content": CALL[:-10]},
                 {"role": "tool", "content": '<tool_response>\n{"name": "get_weather", "content": 31}\n</tool_response>'}]
        prompt = render(item("multi_turn", turns=turns))
        self.assertTrue(prompt.endswith("</tool_response><|im_end|>\n<|im_start|>assistant\n"))
        with self.assertRaises(ValueError):
            render(item("multi_turn", turns=turns[:2]))

    def test_few_shot_prefix_holds_three_whole_conversations(self):
        prefix = few_shot_prefix()
        self.assertEqual(prefix.count("<|im_start|>system"), 3)
        self.assertTrue(render(item("single"), few_shot=True).startswith(prefix))
        self.assertTrue(prefix.endswith("<|im_end|>\n"))

    def test_few_shot_tools_appear_in_no_eval_set(self):
        files = [Path("eval_sets/p.jsonl"), *Path("eval_sets").glob("dev/*.jsonl"), *Path("eval_sets").glob("test/*.jsonl")]
        names = {tool["name"] for path in files for line in path.read_text(encoding="utf-8").splitlines()
                 for tool in json.loads(line)["tools"]}
        self.assertFalse({tool["name"] for example in FEW_SHOT for tool in example["tools"]} & names)

    @unittest.skipUnless(HAS_RUNTIME, "Install the model runtime or run these checks in Colab.")
    def test_instruct_template_renders_the_same_text_once_a_system_prompt_is_given(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
        tokenizer.chat_template = SMOLLM2_INSTRUCT_TEMPLATE
        self.assertEqual(render(item("single"), template="native", tokenizer=tokenizer), render(item("single")))


class ScoreTests(unittest.TestCase):
    def test_each_group_gets_its_own_scores(self):
        single = score(item("single", expected_calls=[{"name": "get_weather", "arguments": {"city": "Hanoi"}}]),
                       CALL, "im_end")
        self.assertTrue(single["call_exact"] and single["im_end_stop"])
        no_fit = score(item("no_tool", kind="no_fitting_tool"), "None of the tools can do that.<|im_end|>", "im_end")
        self.assertTrue(no_fit["no_fitting_said"])
        self.assertNotIn("no_fitting_said", score(item("no_tool", kind="plain"), "Hi<|im_end|>", "im_end"))
        clarify = score(item("missing_info", missing_param="city", synonyms=[]), "Which city?<|im_end|>", "im_end")
        self.assertTrue(clarify["clarify_correct"])
        cut = score(item("missing_info", missing_param="city", synonyms=[]), "Which city? user Which city?", "max_new_tokens")
        self.assertFalse(cut["clarify_correct"])
        grounded = score(item("multi_turn", required=[31], forbidden=[], meta={"derivable": False}),
                         "It is 31 C.<|im_end|>", "im_end")
        self.assertTrue(grounded["grounded_rule"])
        self.assertFalse(grounded["derivable"])

    def test_probe_items_get_behaviour_only(self):
        probe = {**item("multi_turn", required=[31], forbidden=[], meta={"domain": "seen"}), "set": "p"}
        self.assertEqual(set(score(probe, "31<|im_end|>", "im_end")),
                         {"im_end_stop", "eot_stop", "truncated", "role_break", "repetition_rate"})

    def test_aggregate_pools_tool_groups_for_the_decision_and_skips_p(self):
        expected = [{"name": "get_weather", "arguments": {"city": "Hanoi"}}]
        rows = [(item("single", expected_calls=expected), CALL), (item("no_tool", kind="plain"), CALL),
                (item("missing_info", missing_param="city", synonyms=[]), "Which city?<|im_end|>"),
                (item("multi_turn", required=[31], forbidden=[], meta={"derivable": True}), "31<|im_end|>"),
                ({**item("single", expected_calls=expected), "set": "p"}, CALL)]
        records = [{"id": it["id"], "set": it["set"], "group": it["group"], "completion": out, "generated_tokens": 5,
                    "stop_reason": "im_end", "scores": score(it, out, "im_end")} for it, out in rows]
        metrics = aggregate(records)
        self.assertEqual(list(metrics), ["t_seen"])
        tool = metrics["t_seen"]["tool"]
        self.assertEqual((tool["call_exact"], tool["false_call_rate"], tool["decision_bacc"]), (1.0, 1.0, 0.5))
        groups = metrics["t_seen"]["groups"]
        self.assertEqual(groups["missing_info"]["clarify_correct"], 1.0)
        self.assertEqual(groups["multi_turn"]["grounded_correct"], 1.0)
        self.assertEqual(metrics["t_seen"]["behaviour"]["n"], 4)

    def test_rescore_rewrites_scores_and_keeps_the_run_settings(self):
        import tempfile

        from src.eval.run_eval import rescore, write_results

        it = item("missing_info", missing_param="city", synonyms=[])
        with tempfile.TemporaryDirectory() as tmp:
            root, folder = Path(tmp) / "eval_sets", Path(tmp) / "run"
            (root / "dev").mkdir(parents=True)
            (root / "dev" / "t_seen.jsonl").write_text(json.dumps(it) + "\n", encoding="utf-8")
            (root / "p.jsonl").write_text("", encoding="utf-8")
            record = {"id": it["id"], "set": "t_seen", "group": "missing_info", "completion": "Which city?",
                      "generated_tokens": 3, "stop_reason": "max_new_tokens", "scores": {"clarify_correct": True}}
            write_results(folder, [record], {"split": "dev", "model": "m", "metrics": {}})
            summary = rescore(folder, root)
            self.assertEqual(summary["model"], "m")
            self.assertEqual(summary["metrics"]["t_seen"]["groups"]["missing_info"]["clarify_correct"], 0.0)
            saved = json.loads((folder / "outputs.jsonl").read_text(encoding="utf-8"))
            self.assertFalse(saved["scores"]["clarify_correct"])

    def test_merge_replaces_only_the_regenerated_sets(self):
        import tempfile

        from src.eval.run_eval import merge_results, write_results

        it = item("missing_info", missing_param="city", synonyms=[])
        chat = {"id": "c:1", "set": "c", "group": "chat", "tools": [], "turns": [], "reference": "Hi."}
        with tempfile.TemporaryDirectory() as tmp:
            root, folder = Path(tmp) / "eval_sets", Path(tmp) / "run"
            (root / "dev").mkdir(parents=True)
            (root / "dev" / "t_seen.jsonl").write_text(json.dumps(it) + "\n", encoding="utf-8")
            old = [{"id": it["id"], "set": "t_seen", "group": "missing_info", "completion": "Which city?<|im_end|>",
                    "generated_tokens": 3, "stop_reason": "im_end", "scores": {"clarify_correct": False}},
                   {"id": "c:1", "set": "c", "group": "chat", "completion": "Hello hello", "generated_tokens": 256,
                    "stop_reason": "max_new_tokens", "scores": score(chat, "Hello hello", "max_new_tokens")}]
            write_results(folder, old, {"split": "dev", "model": "m", "max_new_tokens": {"c": 256, "t_seen": 256},
                                        "metrics": {}})
            new = [{**old[1], "completion": "Hello.<|im_end|>", "generated_tokens": 3, "stop_reason": "im_end",
                    "scores": score(chat, "Hello.<|im_end|>", "im_end")}]
            summary = merge_results(folder, new, {"generation_seconds": 1.0, "max_new_tokens": {"c": 512}}, root)
            self.assertEqual(summary["max_new_tokens"], {"c": 512, "t_seen": 256})
            self.assertEqual(summary["reruns"][0]["sets"], ["c"])
            self.assertEqual(summary["metrics"]["c"]["behaviour"]["stop_rate"], 1.0)
            # The kept sample is scored again under the current rules.
            self.assertEqual(summary["metrics"]["t_seen"]["groups"]["missing_info"]["clarify_correct"], 1.0)
            saved = [json.loads(line) for line in (folder / "outputs.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["completion"] for r in saved], ["Which city?<|im_end|>", "Hello.<|im_end|>"])

    def test_limit_keeps_the_first_items_of_each_set(self):
        items = [{"set": name, "n": n} for name in ("a", "b") for n in range(3)]
        self.assertEqual([(i["set"], i["n"]) for i in limit_per_set(items, 2)], [("a", 0), ("a", 1), ("b", 0), ("b", 1)])


class TokenLimitTests(unittest.TestCase):
    def test_chat_set_gets_the_longer_limit_and_order_is_kept(self):
        from unittest import mock

        from src.eval import run_eval

        items = [item("single"), {**item("chat"), "id": "c:1", "set": "c"}, {**item("no_tool"), "id": "x:2"}]
        calls = []

        def fake_generate(model, tokenizer, prompts, *, batch_size, max_new_tokens):
            calls.append((len(prompts), max_new_tokens))
            return [{"completion": f"{max_new_tokens}<|im_end|>", "generated_tokens": 1, "stop_reason": "im_end"}
                    for _ in prompts]

        with mock.patch.object(run_eval, "generate", fake_generate):
            records, summary = run_eval.run(None, None, items, template="project", few_shot=False, batch_size=4)
        self.assertEqual(sorted(calls), [(1, 512), (2, 256)])
        self.assertEqual([r["completion"] for r in records], ["256<|im_end|>", "512<|im_end|>", "256<|im_end|>"])
        self.assertEqual(summary["max_new_tokens"], {"t_seen": 256, "c": 512})


@unittest.skipUnless(HAS_RUNTIME, "Install the model runtime or run these checks in Colab.")
class GenerateTests(unittest.TestCase):
    def test_batched_generation_keeps_order_and_cuts_at_the_stop(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        from src.eval.run_eval import generate

        vocab = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "[UNK]": 3, "Hello": 4, "world": 5}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
        torch.manual_seed(0)
        config = LlamaConfig(vocab_size=6, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, eos_token_id=0, bos_token_id=0)
        model = LlamaForCausalLM(config).eval()
        prompts = ["Hello", "Hello world Hello world world", "world Hello"]
        batched = generate(model, tokenizer, prompts, batch_size=2, max_new_tokens=6)
        alone = [generate(model, tokenizer, [prompt], batch_size=1, max_new_tokens=6)[0] for prompt in prompts]
        self.assertEqual([r["completion"] for r in batched], [r["completion"] for r in alone])
        for result in batched:
            self.assertIn(result["stop_reason"], ("im_end", "endoftext", "max_new_tokens"))
            if result["stop_reason"] != "max_new_tokens":
                self.assertEqual(result["completion"].count("<|im_end|>") + result["completion"].count("<|endoftext|>"), 1)


if __name__ == "__main__":
    unittest.main()
