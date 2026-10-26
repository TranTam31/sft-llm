"""Run one model or checkpoint on the frozen eval sets (PLAN 3.6): greedy generation, rule scoring, metrics per set and
group. P is generated alongside for reading by eye and is not scored.

Run: python -m src.eval.run_eval --model base --run e0 --checkpoint base-zero-shot [--split dev] [--few-shot]
     python -m src.eval.run_eval --model chat_reference --template native --run e0 --checkpoint instruct
--model is a key of configs/models.yaml, a Hub id (then --revision pins it) or a local checkpoint directory.
Writes results/<run>/<checkpoint>/outputs.jsonl (raw outputs with per-sample scores) and metrics.json.
--rescore results/<run>/<checkpoint> scores saved outputs again after a rule change, without generating.
Bootstrap CIs and results/all.csv are computed later from outputs.jsonl.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from src.chat_format import system_content
from src.eval import manifest
from src.eval.scorers import (aggregate_grounded, aggregate_tool_group, score_behavior, score_clarify, score_grounded,
                              score_no_fitting, score_tool_sample)
from src.playground import classify_stop
from src.sft_data import prepare_messages, render_messages


ROOT = Path("eval_sets")
SETS = ("t_seen", "t_unseen", "t_rename_nonsense", "t_rename_synonym", "c")
TOOL_GROUPS = ("single", "select", "parallel", "no_tool")
MAX_NEW_TOKENS = 256
# PLAN 3.6 (2026-10-26): about 1 in 5 C references runs past 256 tokens, which capped C's stop_rate near 0.8.
MAX_NEW_TOKENS_BY_SET = {"c": 512}

# PLAN 3.6: three fixed examples for base few-shot, with tools that appear in no eval set (checked in tests).
# One call, one direct answer despite offered tools, one clarifying question.
FEW_SHOT = [
    {"tools": [{"name": "convert_length", "description": "Convert a length between metric and imperial units.",
                "parameters": {"type": "object", "properties": {
                    "value": {"type": "number", "description": "The length to convert."},
                    "from_unit": {"type": "string", "description": "Unit of the value, e.g. 'mile'."},
                    "to_unit": {"type": "string", "description": "Unit to convert to, e.g. 'km'."}},
                    "required": ["value", "from_unit", "to_unit"]}}],
     "turns": [{"role": "user", "content": "How many kilometres is 26.2 miles?"},
               {"role": "assistant", "content": '<tool_call>\n{"name": "convert_length", "arguments": '
                                                '{"value": 26.2, "from_unit": "mile", "to_unit": "km"}}\n</tool_call>'}]},
    {"tools": [{"name": "find_parking_spots", "description": "List free parking spots near an address.",
                "parameters": {"type": "object", "properties": {
                    "address": {"type": "string", "description": "Street address to search around."}},
                    "required": ["address"]}}],
     "turns": [{"role": "user", "content": "What is the capital of Australia?"},
               {"role": "assistant", "content": "The capital of Australia is Canberra."}]},
    {"tools": [{"name": "lookup_book_by_isbn", "description": "Get the title, author and year of a book from its ISBN.",
                "parameters": {"type": "object", "properties": {
                    "isbn": {"type": "string", "description": "The 10- or 13-digit ISBN."}},
                    "required": ["isbn"]}}],
     "turns": [{"role": "user", "content": "Can you tell me who wrote the book I'm holding?"},
               {"role": "assistant", "content": "Sure. What is the book's ISBN? It is usually printed on the back "
                                                "cover above the barcode."}]},
]


def load_items(split: str, root: Path = ROOT, sets=SETS, probe: bool = True) -> list[dict]:
    items = [json.loads(line) for name in sets
             for line in (root / split / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if probe:
        items += [json.loads(line) for line in (root / "p.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return items


def fetch(repo_id: str, revision: str | None = None, root: Path = ROOT):
    """Dev/test live on the private dataset Hub (git-ignored here); MANIFEST.md and p.jsonl come with the repo."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=str(root.parent),
                      allow_patterns=[f"{root.name}/dev/*.jsonl", f"{root.name}/test/*.jsonl"])


def limit_per_set(items: list[dict], limit: int) -> list[dict]:
    counts = {}
    kept = []
    for item in items:
        counts[item["set"]] = counts.get(item["set"], 0) + 1
        if counts[item["set"]] <= limit:
            kept.append(item)
    return kept


def conversation(item: dict) -> list[dict]:
    """System prompt with the item's tools, then its turns; the model answers the last user or tool turn."""
    if item["turns"][-1]["role"] not in ("user", "tool"):
        raise ValueError(f"{item['id']}: the last turn must be user or tool.")
    return prepare_messages([*item["turns"], {"role": "assistant", "content": ""}], item["tools"])[:-1]


def few_shot_prefix() -> str:
    return "".join(render_messages(prepare_messages(example["turns"], example["tools"])) for example in FEW_SHOT)


def render(item: dict, *, template: str = "project", few_shot: bool = False, tokenizer=None) -> str:
    """project: the SFT ChatML prompt. native: the model's own chat template (PLAN 3.6, Instruct models)."""
    messages = conversation(item)
    if template == "native":
        if few_shot:
            raise ValueError("Few-shot is for base models with the project template.")
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if template != "project":
        raise ValueError(f"Unknown template {template!r}.")
    return (few_shot_prefix() if few_shot else "") + render_messages(messages) + "<|im_start|>assistant\n"


def score(item: dict, output: str, stop_reason: str) -> dict:
    scores = score_behavior(output, stop_reason=stop_reason)
    if item["set"] == "p":  # read by eye, not scored
        return scores
    group = item["group"]
    if group in TOOL_GROUPS:
        scores.update(score_tool_sample(output, expected_calls=item["expected_calls"], tools=item["tools"]))
        if item.get("kind") == "no_fitting_tool":
            scores.update(score_no_fitting(output))
    elif group == "missing_info":
        scores.update(score_clarify(output, missing_param=item["missing_param"], synonyms=item["synonyms"],
                                    stop_reason=stop_reason))
    elif group == "multi_turn":
        scores.update(score_grounded(output, required=item["required"], forbidden=item["forbidden"],
                                     stop_reason=stop_reason))
        scores["derivable"] = item["meta"]["derivable"]
    return scores


def _mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else None


def behaviour(records: list[dict]) -> dict:
    scores = [record["scores"] for record in records]
    return {"stop_rate": _mean(s["im_end_stop"] for s in scores), "eot_stop_rate": _mean(s["eot_stop"] for s in scores),
            "truncated_rate": _mean(s["truncated"] for s in scores),
            "role_break_rate": _mean(s["role_break"] for s in scores),
            "repetition_rate": _mean(s["repetition_rate"] for s in scores),
            "avg_len": _mean(record["generated_tokens"] for record in records), "n": len(records)}


def aggregate(records: list[dict]) -> dict:
    """Per eval set: behaviour over every sample; tool metrics over Single/Select/Parallel/No-tool together (so
    decision_bacc sees both kinds) and per group; clarify and grounded metrics for their groups. P is skipped."""
    metrics = {}
    for name in dict.fromkeys(record["set"] for record in records if record["set"] != "p"):
        rows = [record for record in records if record["set"] == name]
        by_group = {}
        for record in rows:
            by_group.setdefault(record["group"], []).append(record)
        result = {"behaviour": behaviour(rows), "groups": {}}
        tool_rows = [record["scores"] for record in rows if record["group"] in TOOL_GROUPS]
        if tool_rows:
            result["tool"] = aggregate_tool_group(tool_rows)
        for group, group_rows in by_group.items():
            scores = [record["scores"] for record in group_rows]
            entry = {"behaviour": behaviour(group_rows)}
            if group in TOOL_GROUPS:
                entry.update(aggregate_tool_group(scores))
            elif group == "missing_info":
                entry["clarify_correct"] = _mean(s["clarify_correct"] for s in scores)
            elif group == "multi_turn":
                entry.update(aggregate_grounded(scores))
            result["groups"][group] = entry
        metrics[name] = result
    return metrics


def generate(model, tokenizer, prompts: list[str], *, batch_size: int, max_new_tokens: int = MAX_NEW_TOKENS):
    """Greedy, batched with left padding, longest prompts first; each row is cut at its first stop token."""
    import torch

    vocab = tokenizer.get_vocab()
    im_end_id, eot_id = vocab["<|im_end|>"], vocab["<|endoftext|>"]
    tokenizer.pad_token, tokenizer.padding_side = "<|endoftext|>", "left"
    device = next(model.parameters()).device
    order = sorted(range(len(prompts)), key=lambda index: -len(prompts[index]))
    results = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        encoded = tokenizer([prompts[index] for index in batch], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
        with torch.inference_mode():
            output = model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                    eos_token_id=[im_end_id, eot_id], pad_token_id=eot_id, use_cache=True)
        for row, index in enumerate(batch):
            new_ids = output[row, encoded.input_ids.shape[1]:].tolist()
            stop_at = next((i for i, token in enumerate(new_ids) if token in (im_end_id, eot_id)), None)
            new_ids = new_ids if stop_at is None else new_ids[:stop_at + 1]
            results[index] = {"completion": tokenizer.decode(new_ids, skip_special_tokens=False),
                              "generated_tokens": len(new_ids),
                              "stop_reason": classify_stop(new_ids, im_end_id=im_end_id, eot_id=eot_id,
                                                           limit=max_new_tokens)}
    return results


def load_model(model_id: str, revision: str | None, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {} if Path(model_id).is_dir() else {"revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16 if device == "cuda" else torch.float32, **kwargs).to(device)
    return model.eval(), tokenizer


def token_limit(item: dict) -> int:
    return MAX_NEW_TOKENS_BY_SET.get(item["set"], MAX_NEW_TOKENS)


def run(model, tokenizer, items: list[dict], *, template: str, few_shot: bool, batch_size: int) -> tuple[list, dict]:
    """Generates each token limit's items in their own batches, so the longer limit of C does not slow the rest."""
    prompts = [render(item, template=template, few_shot=few_shot, tokenizer=tokenizer) for item in items]
    started = perf_counter()
    outputs = [None] * len(items)
    for limit in sorted({token_limit(item) for item in items}):
        indices = [index for index, item in enumerate(items) if token_limit(item) == limit]
        for index, output in zip(indices, generate(model, tokenizer, [prompts[i] for i in indices],
                                                   batch_size=batch_size, max_new_tokens=limit)):
            outputs[index] = output
    seconds = perf_counter() - started
    records = [{"id": item["id"], "set": item["set"], "group": item["group"], **output,
                "scores": score(item, output["completion"], output["stop_reason"])}
               for item, output in zip(items, outputs)]
    return records, {"generation_seconds": seconds, "n_prompts": len(items),
                     "max_new_tokens": {**{name: MAX_NEW_TOKENS for name in dict.fromkeys(i["set"] for i in items)},
                                        **{name: limit for name, limit in MAX_NEW_TOKENS_BY_SET.items()
                                           if any(i["set"] == name for i in items)}}}


def rescore(folder: Path, root: Path = ROOT) -> dict:
    """Score saved outputs again under the current rules (no generation); rewrites outputs.jsonl and metrics.json."""
    summary = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (folder / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    sets = [name for name in dict.fromkeys(record["set"] for record in records) if name != "p"]
    items = {item["id"]: item for item in load_items(summary["split"], root, sets)}
    for record in records:
        record["scores"] = score(items[record["id"]], record["completion"], record["stop_reason"])
    summary = {**summary, "metrics": aggregate(records)}
    write_results(folder, records, summary)
    return summary


def merge_results(folder: Path, records: list[dict], timing: dict, root: Path = ROOT) -> dict:
    """Replace the saved records of the sets just generated (e.g. C after its token limit changed); the other sets'
    outputs stay as they were but are scored again under the current rules. The summary notes which sets were redone."""
    summary = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    sets = set(record["set"] for record in records)
    kept = [json.loads(line) for line in (folder / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    kept = [record for record in kept if record["set"] not in sets]
    items = {item["id"]: item for item in load_items(
        summary["split"], root, [name for name in dict.fromkeys(record["set"] for record in kept) if name != "p"], probe=False)}
    for record in kept:
        if record["set"] != "p":
            record["scores"] = score(items[record["id"]], record["completion"], record["stop_reason"])
    merged = kept + records
    reruns = [*summary.get("reruns", []), {"sets": sorted(sets), **timing}]
    limits = {**summary.get("max_new_tokens", {}), **timing.get("max_new_tokens", {})}
    summary = {**summary, "max_new_tokens": limits, "reruns": reruns, "metrics": aggregate(merged)}
    write_results(folder, merged, summary)
    return summary


def write_results(folder: Path, records: list[dict], summary: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "outputs.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (folder / "metrics.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def print_summary(summary: dict) -> None:
    print(json.dumps({key: value for key, value in summary.items() if key != "metrics"}, indent=1))
    for name, result in summary["metrics"].items():
        tool = result.get("tool", {})
        print(name, {key: round(value, 3) for key, value in {
            "stop_rate": result["behaviour"]["stop_rate"], "call_exact": tool.get("call_exact"),
            "decision_bacc": tool.get("decision_bacc"),
            "clarify_correct": result["groups"].get("missing_info", {}).get("clarify_correct"),
            "grounded_correct": result["groups"].get("multi_turn", {}).get("grounded_correct")}.items()
            if value is not None})


def main():
    from src.eval.benchmarks import resolve

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="base")
    parser.add_argument("--revision")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--template", default="project", choices=("project", "native"))
    parser.add_argument("--few-shot", action="store_true")
    parser.add_argument("--run")
    parser.add_argument("--checkpoint")
    parser.add_argument("--rescore", metavar="FOLDER", help="Re-score results/<run>/<checkpoint> without generating.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="First N items per set, for a quick check; not for reporting.")
    parser.add_argument("--sets", nargs="+", choices=SETS,
                        help="Generate only these sets and merge them into the checkpoint's saved results.")
    parser.add_argument("--fetch", metavar="DATASET_REPO", help="Download dev/test from this dataset Hub repo first.")
    args = parser.parse_args()

    if args.fetch:
        fetch(args.fetch)

    problems = manifest.check()
    if problems:
        raise SystemExit("Eval sets differ from eval_sets/MANIFEST.md: " + "; ".join(problems))
    if args.rescore:
        print_summary(rescore(Path(args.rescore)))
        return
    if not (args.run and args.checkpoint):
        parser.error("--run and --checkpoint are required unless --rescore is given")
    items = load_items(args.split, sets=args.sets or SETS, probe=not args.sets)
    if args.limit:
        items = limit_per_set(items, args.limit)
    model_id, revision = resolve(args.model, args.revision, args.config)
    model, tokenizer = load_model(model_id, revision, args.device)
    records, timing = run(model, tokenizer, items, template=args.template, few_shot=args.few_shot,
                          batch_size=args.batch_size)
    if args.sets:
        print_summary(merge_results(Path("results") / args.run / args.checkpoint, records, timing))
        return
    summary = {"model": model_id, "revision": revision, "split": args.split, "template": args.template,
               "few_shot": args.few_shot, "limit": args.limit, "batch_size": args.batch_size, "device": args.device,
               **timing, "metrics": aggregate(records)}
    write_results(Path("results") / args.run / args.checkpoint, records, summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
