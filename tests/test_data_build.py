"""Counting and reporting in the data build, with a fake length measure (no downloads)."""

import unittest

from src.data.build import SourceStats, process, render_report
from src.data.convert import ConversionError
from src.sft_data import SequenceTooLongError


def convert(row, index):
    if row == "bad":
        raise ConversionError("bad_call")
    return {"id": str(index), "group": "single", "turns": [], "tools": [], "meta": {"length": row}}


def measure(example):
    if example["meta"]["length"] > 10:
        raise SequenceTooLongError("too long")
    example["meta"]["n_tokens"] = example["meta"]["length"]


CONFIG = {"seed": 0, "max_seq_len": 10,
          "xlam": {"repo": "x/xlam", "revision": "a"}, "glaive": {"repo": "g/glaive", "revision": "b"},
          "smoltalk": {"repo": "s/smoltalk", "revision": "c", "excluded": {"apigen-80k": "tool data"}}}


class ProcessTests(unittest.TestCase):
    def test_every_row_is_kept_or_counted_with_a_reason(self):
        stats = SourceStats()
        kept = process(enumerate([3, "bad", 30, 5]), convert, measure, stats)
        self.assertEqual([example["id"] for example in kept], ["0", "3"])
        self.assertEqual((stats.rows, stats.kept), (4, 2))
        self.assertEqual(dict(stats.dropped), {"bad_call": 1, "too_long": 1})
        self.assertAlmostEqual(stats.too_long_rate(), 1 / 3)  # Conversion errors are not length losses.

    def test_cap_stops_after_enough_kept_examples(self):
        stats = SourceStats()
        kept = process(enumerate([30, 1, 2, 3, 4]), convert, measure, stats, cap=2)
        self.assertEqual(len(kept), 2)
        self.assertEqual(stats.rows, 3)

    def test_report_flags_sources_losing_too_much_to_length(self):
        long_source, short_source = SourceStats(), SourceStats()
        process(enumerate([30, 30, 1]), convert, measure, long_source)
        process(enumerate([1, 2]), convert, measure, short_source)
        report = render_report({"glaive": ([], long_source), "xlam": ([], short_source)}, {"xlam": "abc"}, CONFIG)
        self.assertIn("66.7% ⚠", report)
        self.assertIn("Hiện có: glaive.", report)
        self.assertIn("`apigen-80k`: tool data", report)
        self.assertIn("| `xlam.jsonl` | `abc` |", report)

    def test_report_lists_config_notes(self):
        report = render_report({}, {}, {**CONFIG, "notes": ["Kept 1024 tokens."]})
        self.assertIn("## Ghi chú\n\n- Kept 1024 tokens.", report)


if __name__ == "__main__":
    unittest.main()
