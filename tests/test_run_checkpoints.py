import json
from pathlib import Path
import tempfile
import unittest

from src.eval.run_checkpoints import list_snapshots, pending, results_store


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


class ResultsStoreTests(unittest.TestCase):
    def test_the_store_is_private_read_first_and_written_per_snapshot(self):
        calls = []

        class Hub:
            def create_repo(self, repo_id, private, exist_ok):
                calls.append(("create", repo_id, private))

            def snapshot_download(self, repo_id, allow_patterns, local_dir):
                calls.append(("download", allow_patterns, Path(local_dir).name))

            def upload_folder(self, *, repo_id, folder_path, path_in_repo, commit_message):
                calls.append(("upload", Path(folder_path).name, path_in_repo))

        save = results_store("user/results", "e2_mix_s0", Path("results") / "e2_mix_s0", api=Hub())
        save("step-00005")
        self.assertEqual(calls, [("create", "user/results", True), ("download", ["e2_mix_s0/*"], "results"),
                                 ("upload", "step-00005", "e2_mix_s0/step-00005")])


if __name__ == "__main__":
    unittest.main()
