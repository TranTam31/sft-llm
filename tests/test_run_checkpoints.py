import json
from pathlib import Path
import tempfile
import unittest

from src.eval.run_checkpoints import list_snapshots, pending


class SnapshotListTests(unittest.TestCase):
    def test_saved_snapshots_are_listed_oldest_first_and_scored_ones_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            snapshots, results = Path(root) / "snapshots", Path(root) / "results"
            for name in ("step-00120", "step-00007", "step-00030"):
                (snapshots / name).mkdir(parents=True)
                (snapshots / name / "meta.json").write_text(json.dumps({"step": int(name[5:])}), encoding="utf-8")
            (snapshots / "step-00050").mkdir()          # still being written: no meta.json yet
            (results / "step-00007").mkdir(parents=True)
            (results / "step-00007" / "metrics.json").write_text("{}", encoding="utf-8")
            names = list_snapshots(snapshots)
            self.assertEqual(names, ["step-00007", "step-00030", "step-00120"])
            self.assertEqual(pending(names, results), ["step-00030", "step-00120"])


if __name__ == "__main__":
    unittest.main()
