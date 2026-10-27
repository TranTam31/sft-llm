"""Score every eval snapshot of a training run on the frozen dev (or test) sets, the same way run_eval scores one model
(PLAN 3.6, 3.7).

Run: python -m src.eval.run_checkpoints --run e2_pilot [--hub-model-id USER/REPO] [--split dev] [--batch-size 32]
     [--fetch DATASET_REPO] [--cleanup] [--sets c] [--results-repo USER/REPO]
Snapshots are read from results/runs/<run>/snapshots/step-NNNNN (written by src.train), or downloaded one at a time
from snapshots/ of the run's Hub repo. Writes results/<run>/step-NNNNN/{outputs.jsonl,metrics.json}; metrics.json
carries the snapshot's meta.json (step, tokens_seen, assistant_tokens_seen). Snapshots already scored are skipped, so
a sweep cut off by a disconnect picks up where it stopped. --sets regenerates only those sets on every scored snapshot
and merges them into its results (run_eval.merge_results), e.g. after a set's token limit changed. --results-repo
(private: outputs quote the gated eval sets) first downloads <run>/ from that repo, so a new runtime also skips what
was scored, and uploads each snapshot's results there as soon as they are written.
"""

import argparse
import json
from pathlib import Path
import shutil

from src.eval import manifest
from src.eval.run_eval import (SETS, aggregate, fetch, load_items, load_model, merge_results, print_summary, run,
                               write_results)


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


def results_store(repo_id: str, run: str, results: Path, *, api=None):
    """Download what <run>/ of the results repo holds into results/, and return save(name) that uploads one
    snapshot's folder back."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.snapshot_download(repo_id, allow_patterns=[f"{run}/*"], local_dir=str(results.parent))

    def save(name: str):
        api.upload_folder(repo_id=repo_id, folder_path=str(results / name), path_in_repo=f"{run}/{name}",
                          commit_message=f"{run}: results of {name}")
    return save


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
    parser.add_argument("--sets", nargs="+", choices=SETS, help="Regenerate only these sets on scored snapshots.")
    parser.add_argument("--results-repo", help="Private Hub repo holding <run>/ results: read first, written per snapshot.")
    args = parser.parse_args()

    if args.fetch:
        fetch(args.fetch)
    problems = manifest.check()
    if problems:
        raise SystemExit("Eval sets differ from eval_sets/MANIFEST.md: " + "; ".join(problems))
    local = run_dir(args.run) / "snapshots"
    results = Path("results") / args.run
    save = results_store(args.results_repo, args.run, results) if args.results_repo else None
    names = list_snapshots(local, args.hub_model_id)
    if args.sets:
        names = [name for name in names if name not in pending(names, results)]
    else:
        names = pending(names, results)
    print(f"[run_checkpoints] {len(names)} snapshots to score: {names}", flush=True)
    items = load_items(args.split, sets=args.sets or SETS, probe=not args.sets)
    for name in names:
        import torch

        folder, downloaded = local_snapshot(name, local, args.hub_model_id)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        model, tokenizer = load_model(str(folder), None, args.device)
        records, timing = run(model, tokenizer, items, template="project", few_shot=False, batch_size=args.batch_size)
        if args.sets:
            summary = merge_results(results / name, records, timing)
        else:
            summary = {"model": str(folder), "revision": None, "split": args.split, "template": "project",
                       "few_shot": False, "limit": None, "batch_size": args.batch_size, "device": args.device,
                       **timing, "snapshot": meta, "metrics": aggregate(records)}
            write_results(results / name, records, summary)
        print_summary(summary)
        if save is not None:
            save(name)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if downloaded and args.cleanup:
            shutil.rmtree(folder)


if __name__ == "__main__":
    main()
