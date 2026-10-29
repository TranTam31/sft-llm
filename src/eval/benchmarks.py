"""General-knowledge benchmarks from PLAN 3.3 through lm-evaluation-harness: ARC-Easy and HellaSwag, a fixed subset of
~500 items each, zero-shot and scored by log-likelihood, so base and Instruct models are judged the same way.

Run: python -m src.eval.benchmarks --model chat_reference [--limit 500] [--output results/benchmarks/<name>.json]
--model is a key of configs/models.yaml, a Hub id (then --revision pins it) or a local snapshot directory. The subset
is the first --limit items of each task's evaluation split, the same at every checkpoint. The output keeps each item's
0/1 scores, so two models can be compared item by item:

python -m src.eval.benchmarks --compare A.json B.json   (A minus B, paired bootstrap CI over the shared items)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


TASKS = ("arc_easy", "hellaswag")
METRICS = ("acc", "acc_norm")


def summarise(raw: dict, tasks=TASKS) -> dict:
    """Accuracy, length-normalised accuracy and their standard errors per task from simple_evaluate's results."""
    summary = {}
    for task in tasks:
        row = raw["results"][task]
        summary[task] = {metric: row[f"{metric},none"] for metric in METRICS}
        summary[task].update({f"{metric}_stderr": row[f"{metric}_stderr,none"] for metric in METRICS})
        summary[task]["n"] = raw["n-samples"][task]["effective"]
    return summary


def item_scores(samples: dict, tasks=TASKS) -> dict:
    """{task: {doc_id: [acc, acc_norm]}} from simple_evaluate's logged samples."""
    return {task: {str(sample["doc_id"]): [float(sample[metric]) for metric in METRICS] for sample in samples[task]}
            for task in tasks}


def paired_differences(a: dict, b: dict, reps: int = 2000, seed: int = 0, level: float = 0.95) -> list[dict]:
    """A minus B per task and metric on the items both scored, with a bootstrap CI that resamples the same items for
    both sides (PLAN 3.5). Items are drawn independently: ARC-Easy questions and HellaSwag contexts share no
    cluster structure the eval sets would have to respect."""
    rows = []
    for task in a["items"]:
        shared = sorted(set(a["items"][task]) & set(b["items"].get(task, {})), key=int)
        if not shared:
            continue
        left = np.array([a["items"][task][i] for i in shared])
        right = np.array([b["items"][task][i] for i in shared])
        rng = np.random.default_rng([seed, len(shared)])
        draws = rng.integers(0, len(shared), size=(reps, len(shared)))
        boot = left[draws].mean(1) - right[draws].mean(1)
        tail = 100 * (1 - level) / 2
        for column, metric in enumerate(METRICS):
            low, high = np.percentile(boot[:, column], [tail, 100 - tail])
            rows.append({"task": task, "metric": metric, "n": len(shared), "a": float(left[:, column].mean()),
                         "b": float(right[:, column].mean()),
                         "diff": float(left[:, column].mean() - right[:, column].mean()),
                         "diff_low": float(low), "diff_high": float(high)})
    return rows


def resolve(model: str, revision: str | None, config_path: str) -> tuple[str, str | None]:
    models = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if model in models:
        return models[model]["id"], models[model]["revision"]
    return model, revision


def run(model_id: str, revision: str | None, *, limit: int, device: str, batch_size: int) -> dict:
    import lm_eval
    import torch

    model_args = {"pretrained": model_id, "revision": revision or "main", "device": device,
                  "dtype": "float32" if device == "cpu" else "float16", "batch_size": batch_size}
    raw = lm_eval.simple_evaluate(model="hf", model_args=model_args, tasks=list(TASKS), num_fewshot=0,
                                  limit=limit, log_samples=True, random_seed=0, numpy_random_seed=0,
                                  torch_random_seed=0, fewshot_random_seed=0)
    from importlib.metadata import version
    return {"model": model_id, "revision": revision, "limit": limit, "num_fewshot": 0, "device": device,
            "torch": torch.__version__, "lm_eval": version("lm_eval"), "tasks": summarise(raw),
            "items": item_scores(raw["samples"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="chat_reference")
    parser.add_argument("--revision")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Two result files: print A minus B.")
    args = parser.parse_args()
    if args.compare:
        a, b = (json.loads(Path(path).read_text(encoding="utf-8")) for path in args.compare)
        print(f"{a['model']} minus {b['model']} (95% paired bootstrap CI):")
        for r in paired_differences(a, b):
            print(f"  {r['task']:10} {r['metric']:8} n={r['n']} {r['a']:.3f} - {r['b']:.3f} = {r['diff']:+.3f} "
                  f"[{r['diff_low']:+.3f}, {r['diff_high']:+.3f}]")
        return
    model_id, revision = resolve(args.model, args.revision, args.config)
    result = run(model_id, revision, limit=args.limit, device=args.device, batch_size=args.batch_size)
    output = Path(args.output or f"results/benchmarks/{model_id.split('/')[-1]}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
