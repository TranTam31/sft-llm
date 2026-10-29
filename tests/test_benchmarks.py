import unittest

from src.eval.benchmarks import item_scores, paired_differences, resolve, summarise


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

    def test_item_scores_keep_both_accuracies_per_document(self):
        samples = {"arc_easy": [{"doc_id": 0, "acc": 1.0, "acc_norm": 0.0}, {"doc_id": 1, "acc": 0.0, "acc_norm": 1.0}]}
        self.assertEqual(item_scores(samples, tasks=("arc_easy",)), {"arc_easy": {"0": [1.0, 0.0], "1": [0.0, 1.0]}})

    def test_paired_difference_uses_shared_items_only(self):
        a = {"items": {"arc_easy": {"0": [1, 1], "1": [1, 0], "2": [0, 0]}}}
        b = {"items": {"arc_easy": {"0": [0, 1], "1": [0, 0]}}}
        rows = {r["metric"]: r for r in paired_differences(a, b, reps=200)}
        self.assertEqual(rows["acc"]["n"], 2)
        self.assertEqual(rows["acc"]["diff"], 1.0)               # A right on both shared items, B on neither
        self.assertEqual((rows["acc"]["diff_low"], rows["acc"]["diff_high"]), (1.0, 1.0))
        self.assertEqual(rows["acc_norm"]["diff"], 0.0)
        self.assertLessEqual(rows["acc_norm"]["diff_low"], 0.0)


if __name__ == "__main__":
    unittest.main()
