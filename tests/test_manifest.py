from pathlib import Path
import tempfile
import unittest

from src.eval.manifest import check, render


class ManifestTests(unittest.TestCase):
    def test_a_changed_missing_or_unlisted_file_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("dev/t_seen.jsonl", "test/t_seen.jsonl", "p.jsonl"):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text('{"id": 1}\n', encoding="utf-8")
            manifest = root / "MANIFEST.md"
            manifest.write_text(render(root, note="Hub commit abc."), encoding="utf-8")
            self.assertEqual(check(root, manifest), [])
            (root / "test/t_seen.jsonl").write_text('{"id": 2}\n', encoding="utf-8")
            (root / "p.jsonl").unlink()
            (root / "dev/c.jsonl").write_text("{}\n", encoding="utf-8")
            self.assertEqual(check(root, manifest), ["missing: p.jsonl", "changed: test/t_seen.jsonl",
                                                     "not in manifest: dev/c.jsonl"])


if __name__ == "__main__":
    unittest.main()
