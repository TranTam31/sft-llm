"""Forgetting probes on raw text (PLAN 3.3): perplexity on the wikitext-2 test split, without any chat template, and
the mean per-token KL(model ‖ base) on the same tokens, a measure of drift from base rather than of forgetting.

Run: python -m src.eval.knowledge --model <key | Hub id | checkpoint dir> [--base base] [--output results/...json]
The text is the split's lines joined as-is, cut into non-overlapping windows of --window tokens; every token after
the first of its window is scored, the same way at every checkpoint.
"""

import argparse
import json
import math
from pathlib import Path


WIKITEXT = {"repo": "Salesforce/wikitext", "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "file": "wikitext-2-raw-v1/test-00000-of-00001.parquet"}


def windows(token_ids: list[int], size: int) -> list[list[int]]:
    """Non-overlapping windows; a last window shorter than 2 tokens has nothing to score and is dropped."""
    return [token_ids[start:start + size] for start in range(0, len(token_ids), size)
            if len(token_ids[start:start + size]) >= 2]


def load_text() -> str:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(WIKITEXT["repo"], WIKITEXT["file"], revision=WIKITEXT["revision"], repo_type="dataset")
    return "".join(pd.read_parquet(path)["text"])


def score(model, chunks: list[list[int]], base=None) -> dict:
    """Mean token NLL (and KL to base when given), accumulated in float64 over every scored token."""
    import torch

    device = next(model.parameters()).device
    nll_sum, kl_sum, count = 0.0, 0.0, 0
    with torch.inference_mode():
        for chunk in chunks:
            ids = torch.tensor([chunk], device=device)
            log_probs = model(ids).logits[0, :-1].float().log_softmax(-1)
            targets = ids[0, 1:]
            nll_sum += -log_probs.gather(-1, targets[:, None]).sum().item()
            if base is not None:
                base_log_probs = base(ids).logits[0, :-1].float().log_softmax(-1)
                kl_sum += (log_probs.exp() * (log_probs - base_log_probs)).sum().item()
            count += len(chunk) - 1
    result = {"perplexity": math.exp(nll_sum / count), "nll": nll_sum / count, "tokens": count}
    if base is not None:
        result["kl_to_base"] = kl_sum / count
    return result


def main():
    from src.eval.benchmarks import resolve
    from src.eval.run_eval import load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--base", default="base", help="Reference for the KL; the same model skips it.")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()
    model_id, revision = resolve(args.model, args.revision, args.config)
    base_id, base_revision = resolve(args.base, None, args.config)
    model, tokenizer = load_model(model_id, revision, args.device)
    base = None if (base_id, base_revision) == (model_id, revision) else load_model(base_id, base_revision, args.device)[0]
    chunks = windows(tokenizer(load_text(), add_special_tokens=False).input_ids, args.window)
    result = {"model": model_id, "revision": revision, "base": base_id, "base_revision": base_revision,
              "dataset": WIKITEXT, "window": args.window, **score(model, chunks, base)}
    output = Path(args.output or f"results/knowledge/{Path(model_id).name}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
