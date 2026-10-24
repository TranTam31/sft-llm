"""Score every eval snapshot of a training run on the frozen dev (or test) sets, the same way run_eval scores one model
(PLAN 3.6, 3.7).

Run: python -m src.eval.run_checkpoints --run e2_pilot [--hub-model-id USER/REPO] [--split dev] [--batch-size 32]
     [--fetch DATASET_REPO] [--cleanup]
Snapshots are read from results/runs/<run>/snapshots/step-NNNNN (written by src.train), or downloaded one at a time
from snapshots/ of the run's Hub repo. Writes results/<run>/step-NNNNN/{outputs.jsonl,metrics.json}; metrics.json
carries the snapshot's meta.json (step, tokens_seen, assistant_tokens_seen). Snapshots already scored are skipped, so
a sweep cut off by a disconnect picks up where it stopped.
"""

import argparse
import json
from pathlib import Path
import shutil

from src.eval import manifest
from src.eval.run_eval import aggregate, fetch, load_items, load_model, print_summary, run, write_results


def run_dir(run_name: str) -> Path:
    return Path("results") / "runs" / run_name


def list_snapshots(local: Path, hub_model_id: str | None = None) -> list[str]:
    """Snapshot names, oldest first: those saved locally and, with hub_model_id, those uploaded to the Hub."""
    names = {path.parent.name for path in local.glob("step-*/meta.json")}
    if hub_model_id:
        from huggingface_hub import list_repo_files

        names |= {path.split("/")[1] for path in list_repo_files(hub_model_id)
                  if path.startswith("snapshots/step-") and path.endswith("/meta.json")}
    return sorted(names)


def local_snapshot(name: str, local: Path, hub_model_id: str | None) -> tuple[Path, bool]:
    """The snapshot folder, downloaded first when only the Hub has it; the flag says it was downloaded."""
    folder = local / name
    if (folder / "meta.json").exists():
        return folder, False
    from huggingface_hub import snapshot_download

    snapshot_download(hub_model_id, allow_patterns=[f"snapshots/{name}/*"], local_dir=str(local.parent))
    return folder, True


def pending(names: list[str], results: Path) -> list[str]:
    return [name for name in names if not (results / name / "metrics.json").exists()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--hub-model-id", help="The run's Hub repo, for snapshots not saved locally.")
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fetch", metavar="DATASET_REPO", help="Download dev/test from this dataset Hub repo first.")
    parser.add_argument("--cleanup", action="store_true", help="Delete snapshots downloaded here once scored.")
    args = parser.parse_args()

    if args.fetch:
        fetch(args.fetch)
    problems = manifest.check()
    if problems:
        raise SystemExit("Eval sets differ from eval_sets/MANIFEST.md: " + "; ".join(problems))
    local = run_dir(args.run) / "snapshots"
    results = Path("results") / args.run
    names = pending(list_snapshots(local, args.hub_model_id), results)
    print(f"[run_checkpoints] {len(names)} snapshots to score: {names}", flush=True)
    items = load_items(args.split)
    for name in names:
        import torch

        folder, downloaded = local_snapshot(name, local, args.hub_model_id)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        model, tokenizer = load_model(str(folder), None, args.device)
        records, timing = run(model, tokenizer, items, template="project", few_shot=False, batch_size=args.batch_size)
        summary = {"model": str(folder), "revision": None, "split": args.split, "template": "project",
                   "few_shot": False, "limit": None, "batch_size": args.batch_size, "device": args.device,
                   **timing, "snapshot": meta, "metrics": aggregate(records)}
        write_results(results / name, records, summary)
        print_summary(summary)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if downloaded and args.cleanup:
            shutil.rmtree(folder)


if __name__ == "__main__":
    main()
