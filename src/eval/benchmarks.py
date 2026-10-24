"""General-knowledge benchmarks from PLAN 3.3 through lm-evaluation-harness: ARC-Easy and HellaSwag, a fixed subset of
~500 items each, zero-shot and scored by log-likelihood, so base and Instruct models are judged the same way.

Run: python -m src.eval.benchmarks --model chat_reference [--limit 500] [--output results/benchmarks/<name>.json]
--model is a key of configs/models.yaml or a Hub id (then --revision pins it). The subset is the first --limit items
of each task's evaluation split, the same at every checkpoint.
"""

import argparse
import json
from pathlib import Path

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
                                  limit=limit, log_samples=False, random_seed=0, numpy_random_seed=0,
                                  torch_random_seed=0, fewshot_random_seed=0)
    from importlib.metadata import version
    return {"model": model_id, "revision": revision, "limit": limit, "num_fewshot": 0, "device": device,
            "torch": torch.__version__, "lm_eval": version("lm_eval"), "tasks": summarise(raw)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="chat_reference")
    parser.add_argument("--revision")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output")
    args = parser.parse_args()
    model_id, revision = resolve(args.model, args.revision, args.config)
    result = run(model_id, revision, limit=args.limit, device=args.device, batch_size=args.batch_size)
    output = Path(args.output or f"results/benchmarks/{model_id.split('/')[-1]}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
