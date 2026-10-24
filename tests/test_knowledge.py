import importlib.util
import math
import unittest

from src.eval.knowledge import windows


HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


class WindowTests(unittest.TestCase):
    def test_windows_do_not_overlap_and_drop_an_unscorable_tail(self):
        self.assertEqual(windows(list(range(7)), 3), [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(windows(list(range(8)), 3), [[0, 1, 2], [3, 4, 5], [6, 7]])


@unittest.skipUnless(HAS_RUNTIME, "Install the model runtime or run these checks in Colab.")
class ScoreTests(unittest.TestCase):
    @staticmethod
    def tiny(seed):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(seed)
        config = LlamaConfig(vocab_size=11, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1)
        return LlamaForCausalLM(config).eval()

    def test_perplexity_matches_a_direct_computation_and_kl_is_zero_against_itself(self):
        import torch

        from src.eval.knowledge import score

        model, chunks = self.tiny(0), [[1, 4, 2, 7, 3], [5, 9]]
        result = score(model, chunks, base=model)
        nll = 0.0
        with torch.inference_mode():
            for chunk in chunks:
                log_probs = model(torch.tensor([chunk])).logits[0, :-1].float().log_softmax(-1)
                nll -= sum(log_probs[i, token].item() for i, token in enumerate(chunk[1:]))
        self.assertEqual(result["tokens"], 5)
        self.assertAlmostEqual(result["perplexity"], math.exp(nll / 5), places=4)
        self.assertAlmostEqual(result["kl_to_base"], 0.0, places=6)
        self.assertGreater(score(model, chunks, base=self.tiny(1))["kl_to_base"], 0.0)
        self.assertNotIn("kl_to_base", score(model, chunks))


if __name__ == "__main__":
    unittest.main()
