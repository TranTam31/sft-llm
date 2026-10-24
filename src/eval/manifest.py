"""Freeze the eval sets (PLAN 3.1, 6): record the SHA-256 of every dev/test file and of P in eval_sets/MANIFEST.md,
and check local or downloaded copies against it before any run that reports results.

Run: python -m src.eval.manifest --write   (once, when the sets are final)
     python -m src.eval.manifest           (check; exits non-zero on a missing or changed file)
"""

import argparse
import hashlib
from pathlib import Path
import re
import sys


ROOT = Path("eval_sets")
MANIFEST = ROOT / "MANIFEST.md"
ROW = re.compile(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|")


def frozen_files(root: Path = ROOT) -> list[Path]:
    return sorted([*root.glob("dev/*.jsonl"), *root.glob("test/*.jsonl"), root / "p.jsonl"])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(root: Path = ROOT, note: str = "") -> str:
    rows = [f"| `{path.relative_to(root).as_posix()}` | `{digest(path)}` |" for path in frozen_files(root)]
    return "\n".join([
        "# Manifest bộ eval đã đóng băng", "",
        "Sinh bởi `python -m src.eval.manifest --write`. Đừng sửa tay. Mọi lần chạy eval để báo cáo phải qua "
        "`python -m src.eval.manifest` trước. Dev/test nằm trên dataset Hub private (thư mục `eval_sets/`); "
        "`p.jsonl` nằm trong repo. Đổi một file sau khi đóng băng là đổi giao thức: ghi lý do vào "
        "`EXPERIMENTS.md` (PLAN 3.1).", "",
        *([note, ""] if note else []),
        "| File | SHA-256 |", "|---|---|", *rows, ""])


def recorded(manifest: Path = MANIFEST) -> dict:
    return {match.group(1): match.group(2) for line in manifest.read_text(encoding="utf-8").splitlines()
            for match in [ROW.match(line)] if match}


def check(root: Path = ROOT, manifest: Path = MANIFEST) -> list[str]:
    """Problems: a recorded file that is missing or changed, or a dev/test file the manifest does not list."""
    problems, expected = [], recorded(manifest)
    for name, value in expected.items():
        path = root / name
        if not path.exists():
            problems.append(f"missing: {name}")
        elif digest(path) != value:
            problems.append(f"changed: {name}")
    problems += [f"not in manifest: {path.relative_to(root).as_posix()}" for path in frozen_files(root)
                 if path.exists() and path.relative_to(root).as_posix() not in expected]
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--note", default="", help="One line kept above the table, e.g. the Hub commit.")
    args = parser.parse_args()
    if args.write:
        MANIFEST.write_text(render(note=args.note), encoding="utf-8", newline="\n")
        print(f"[manifest] wrote {MANIFEST} ({len(frozen_files())} files)")
        return
    problems = check()
    for problem in problems:
        print(f"[manifest] {problem}")
    if problems:
        sys.exit(1)
    print(f"[manifest] {len(recorded())} files match {MANIFEST}")


if __name__ == "__main__":
    main()
