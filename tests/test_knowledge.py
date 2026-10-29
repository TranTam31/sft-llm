import importlib.util
import math
import unittest

from src.eval.knowledge import paired_differences, window_articles, windows


HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


class WindowTests(unittest.TestCase):
    def test_windows_do_not_overlap_and_drop_an_unscorable_tail(self):
        self.assertEqual(windows(list(range(7)), 3), [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(windows(list(range(8)), 3), [[0, 1, 2], [3, 4, 5], [6, 7]])

    def test_windows_take_the_article_their_first_token_falls_in(self):
        text = " = A = \n x\n = = Sub = = \n y\n = B = \n z\n"
        token_starts = list(range(0, len(text), 4))          # one token every 4 characters
        b = text.index(" = B")
        labels = window_articles(text, token_starts, 2, len(token_starts) // 2)
        self.assertEqual(labels[0], 0)
        self.assertEqual(labels, [0 if token_starts[2 * i] < b else 1 for i in range(len(labels))])


class CompareTests(unittest.TestCase):
    @staticmethod
    def result(nll, kl=None):
        rows = [{"nll": n, "tokens": 10, "article": i // 2} for i, n in enumerate(nll)]
        if kl is not None:
            for row, value in zip(rows, kl):
                row["kl"] = value
        out = {"perplexity": math.exp(sum(nll) / (10 * len(nll))), "windows": rows}
        if kl is not None:
            out["kl_to_base"] = sum(kl) / (10 * len(kl))
        return out

    def test_same_scores_give_zero_and_kl_needs_both_sides(self):
        a = self.result([20.0, 22.0, 18.0, 25.0], kl=[1.0, 1.0, 1.0, 1.0])
        rows = paired_differences(a, a, reps=200)
        self.assertEqual([r["metric"] for r in rows], ["perplexity", "kl_to_base"])
        self.assertTrue(all(r["diff"] == r["diff_low"] == r["diff_high"] == 0 for r in rows))
        self.assertEqual(rows[0]["articles"], 2)
        self.assertEqual([r["metric"] for r in paired_differences(a, self.result([20.0] * 4), reps=200)],
                         ["perplexity"])

    def test_higher_nll_everywhere_gives_a_positive_interval(self):
        rows = paired_differences(self.result([21.0, 23.0, 19.0, 26.0]), self.result([20.0, 22.0, 18.0, 25.0]),
                                  reps=200)
        self.assertGreater(rows[0]["diff"], 0)
        self.assertGreater(rows[0]["diff_low"], 0)


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
        self.assertEqual([w["tokens"] for w in result["windows"]], [4, 1])
        self.assertAlmostEqual(sum(w["nll"] for w in result["windows"]), nll, places=4)


if __name__ == "__main__":
    unittest.main()
