import unittest

from src.eval.benchmarks import resolve, summarise


class BenchmarkTests(unittest.TestCase):
    def test_summary_keeps_accuracies_errors_and_counts(self):
        raw = {"results": {"arc_easy": {"acc,none": 0.6, "acc_stderr,none": 0.02, "acc_norm,none": 0.55,
                                        "acc_norm_stderr,none": 0.022, "alias": "arc_easy"}},
               "n-samples": {"arc_easy": {"original": 2376, "effective": 500}}}
        self.assertEqual(summarise(raw, tasks=("arc_easy",)),
                         {"arc_easy": {"acc": 0.6, "acc_norm": 0.55, "acc_stderr": 0.02, "acc_norm_stderr": 0.022, "n": 500}})

    def test_models_resolve_from_the_pinned_config(self):
        self.assertEqual(resolve("chat_reference", None, "configs/models.yaml"),
                         ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951"))
        self.assertEqual(resolve("org/model", "abc", "configs/models.yaml"), ("org/model", "abc"))


if __name__ == "__main__":
    unittest.main()
