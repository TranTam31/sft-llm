"""Forgetting probes on raw text (PLAN 3.3): perplexity on the wikitext-2 test split, without any chat template, and
the mean per-token KL(model ‖ base) on the same tokens, a measure of drift from base rather than of forgetting.

Run: python -m src.eval.knowledge --model <key | Hub id | checkpoint dir> [--base base] [--output results/...json]
The text is the split's lines joined as-is, cut into non-overlapping windows of --window tokens; every token after
the first of its window is scored, the same way at every checkpoint. The output keeps each window's sums and the
article it starts in, so two models can be compared on the same windows:

python -m src.eval.knowledge --compare A.json B.json   (A minus B, bootstrap CI resampling whole articles)
"""

import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path
import re

import numpy as np


WIKITEXT = {"repo": "Salesforce/wikitext", "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "file": "wikitext-2-raw-v1/test-00000-of-00001.parquet"}


def windows(token_ids: list[int], size: int) -> list[list[int]]:
    """Non-overlapping windows; a last window shorter than 2 tokens has nothing to score and is dropped."""
    return [token_ids[start:start + size] for start in range(0, len(token_ids), size)
            if len(token_ids[start:start + size]) >= 2]


HEADING = re.compile(r"^ = (?!=).*(?<!=) = $", re.MULTILINE)   # " = Title = ", not " = = Section = = "


def window_articles(text: str, token_starts: list[int], size: int, count: int) -> list[int]:
    """Index of the article (top-level heading) each window's first token falls in; text before the first heading is
    article 0, like the heading after it."""
    starts = [match.start() for match in HEADING.finditer(text)]
    return [max(bisect_right(starts, token_starts[index * size]) - 1, 0) for index in range(count)]


def load_text() -> str:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(WIKITEXT["repo"], WIKITEXT["file"], revision=WIKITEXT["revision"], repo_type="dataset")
    return "".join(pd.read_parquet(path)["text"])


def score(model, chunks: list[list[int]], base=None) -> dict:
    """Mean token NLL (and KL to base when given), accumulated in float64 over every scored token, and each window's
    sums under `windows`."""
    import torch

    device = next(model.parameters()).device
    per_window = []
    with torch.inference_mode():
        for chunk in chunks:
            ids = torch.tensor([chunk], device=device)
            log_probs = model(ids).logits[0, :-1].float().log_softmax(-1)
            targets = ids[0, 1:]
            row = {"nll": -log_probs.gather(-1, targets[:, None]).sum().item(), "tokens": len(chunk) - 1}
            if base is not None:
                base_log_probs = base(ids).logits[0, :-1].float().log_softmax(-1)
                row["kl"] = (log_probs.exp() * (log_probs - base_log_probs)).sum().item()
            per_window.append(row)
    nll_sum, count = sum(row["nll"] for row in per_window), sum(row["tokens"] for row in per_window)
    result = {"perplexity": math.exp(nll_sum / count), "nll": nll_sum / count, "tokens": count}
    if base is not None:
        result["kl_to_base"] = sum(row["kl"] for row in per_window) / count
    result["windows"] = per_window
    return result


def paired_differences(a: dict, b: dict, reps: int = 2000) -> list[dict]:
    """A minus B for perplexity (and KL to base when both have it) on the same windows, with a bootstrap CI that
    resamples whole articles for both sides: windows of one article are not independent (PLAN 3.5)."""
    from src.eval.stats import interval, resample_weights

    if [w["tokens"] for w in a["windows"]] != [w["tokens"] for w in b["windows"]]:
        raise SystemExit("The two files were not scored on the same windows.")
    articles = np.array([w["article"] for w in a["windows"]])
    weights = resample_weights(np.unique(articles, return_inverse=True)[1], "wikitext", reps)
    tokens = np.array([w["tokens"] for w in a["windows"]], dtype=float)
    rows = []
    for metric, key, value in (("perplexity", "nll", lambda total: np.exp(total / (weights @ tokens))),
                               ("kl_to_base", "kl", lambda total: total / (weights @ tokens))):
        if not all(key in w for w in a["windows"] + b["windows"]):
            continue
        side = [np.array([w[key] for w in result["windows"]]) for result in (a, b)]
        boot = value(weights @ side[0]) - value(weights @ side[1])
        low, high = interval(boot)
        rows.append({"metric": metric, "a": a[metric], "b": b[metric], "diff": a[metric] - b[metric],
                     "diff_low": low, "diff_high": high, "articles": len(set(articles.tolist()))})
    return rows


def main():
    from src.eval.benchmarks import resolve
    from src.eval.run_eval import load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--base", default="base", help="Reference for the KL; the same model skips it.")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Two result files: print A minus B.")
    args = parser.parse_args()
    if args.compare:
        a, b = (json.loads(Path(path).read_text(encoding="utf-8")) for path in args.compare)
        print(f"{a['model']} minus {b['model']} (95% CI, articles resampled):")
        for r in paired_differences(a, b):
            print(f"  {r['metric']:10} {r['a']:.4f} - {r['b']:.4f} = {r['diff']:+.4f} "
                  f"[{r['diff_low']:+.4f}, {r['diff_high']:+.4f}] ({r['articles']} articles)")
        return
    if not args.model:
        parser.error("--model is required unless --compare is given")
    model_id, revision = resolve(args.model, args.revision, args.config)
    base_id, base_revision = resolve(args.base, None, args.config)
    model, tokenizer = load_model(model_id, revision, args.device)
    base = None if (base_id, base_revision) == (model_id, revision) else load_model(base_id, base_revision, args.device)[0]
    text = load_text()
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    chunks = windows(encoded.input_ids, args.window)
    result = {"model": model_id, "revision": revision, "base": base_id, "base_revision": base_revision,
              "dataset": WIKITEXT, "window": args.window, **score(model, chunks, base)}
    labels = window_articles(text, [start for start, _ in encoded.offset_mapping], args.window, len(chunks))
    for row, article in zip(result["windows"], labels):
        row["article"] = article
    output = Path(args.output or f"results/knowledge/{Path(model_id).name}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
