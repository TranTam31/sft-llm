"""Real tiny random Llama checks. No pretrained weights are downloaded."""

import importlib.util
import unittest


HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "tokenizers"))


@unittest.skipUnless(HAS_RUNTIME, "Install the model runtime or run these checks in Colab.")
class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from src.playground import Playground

        vocab = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2,
                 "[UNK]": 3, "Hello": 4, "world": 5, "a": 6, "b": 7}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
        config = LlamaConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, eos_token_id=0, bos_token_id=0)
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        cls.lab = Playground(model, tokenizer, {"model_id": "tiny-random-test-only"})

    def test_next_token_probabilities_are_not_topk_normalized(self):
        rows = self.lab.next_tokens("Hello", top_k=3)
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(sum(row["probability"] for row in rows), 3 / 8)

    def test_real_generation_detects_eot_and_slices_away_prompt(self):
        record = self.lab.generate("Hello world", max_new_tokens=4)
        self.assertEqual(record["generated_token_ids"], [0])
        self.assertEqual(record["stop_reason"], "endoftext")
        self.assertFalse(record["truncated"])
        self.assertGreater(record["tokens_per_second"], 0)

    def test_prompt_set_records_keep_ids_and_stop_reasons(self):
        from src.playground import run_prompts
        prompts = [{"id": "raw", "mode": "raw", "prompt": "Hello"},
                   {"id": "chat", "mode": "chat", "prompt": "Hello world", "tools": []}]
        records = run_prompts(self.lab, prompts, max_new_tokens=2, top_k=2)
        self.assertEqual([record["prompt_id"] for record in records], ["raw", "chat"])
        self.assertEqual([record["mode"] for record in records], ["raw", "chat"])
        self.assertTrue(all(record["stop_reason"] == "endoftext" for record in records))
        self.assertTrue(all(len(record["next_token_candidates"]) == 2 for record in records))

    def test_long_prompt_is_rejected_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "768"):
            self.lab.next_tokens("Hello " * 769)


if __name__ == "__main__":
    unittest.main()
