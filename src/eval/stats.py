"""Bootstrap CIs and emergence points (PLAN 3.5) from saved outputs.jsonl; no generation.

Run: python -m src.eval.stats [--split dev] [--baseline e0/base-zero-shot] [--go-no-go e2_pilot]
Reads every results/<run>/<checkpoint>/{outputs.jsonl,metrics.json} of the split and writes, under results/:
- all.csv: every metric per eval set and group with its 95% CI (PLAN 3.6 columns, plus `checkpoint`);
- diff_base.csv: the paired difference from the baseline (base zero-shot) with its CI;
- emergence.csv: t_reach at each threshold and t_improve per run, skill, T-seen/T-unseen;
- rename.csv: per T-rename set, the paired difference from the same prompts' T-seen originals (H3), with its CI.
--go-no-go RUN prints the model go/no-go criteria (PLAN 6, Phase 2) read at the run's last checkpoint.

Resampling is by cluster: two prompts share a cluster when they share their first user turn or a tool that matters
for the answer (tools called in Single/Select/Parallel, tools offered in Missing-info/Multi-turn; No-tool's tools are
distractors and do not link). Clusters are the connected components of that link inside each population, and a
T-rename prompt takes its T-seen original's links. The same resampled clusters are used for every checkpoint, so
differences are paired.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import zlib

import numpy as np

from src.eval.run_eval import ROOT, TOOL_GROUPS, load_items


REPS = 2000
SEED = 0
LEVEL = 0.95
IMPROVE = 0.2              # t_improve: gain over base zero-shot
NEAR_CEILING = 0.8         # base at or above this cannot gain IMPROVE
RESULTS = Path("results")

BEHAVIOUR = ("stop_rate", "eot_stop_rate", "truncated_rate", "role_break_rate", "repetition_rate", "avg_len")
NEED_TOOL = ("format_valid", "strict_format", "call_exact", "tool_name_acc", "args_f1", "missed_call")
TOOL = (*NEED_TOOL, "hallucinated_tool", "false_call_rate", "decision_bacc", "no_fitting_said")
GROUP_METRICS = {"missing_info": ("clarify_correct",),
                 "multi_turn": ("grounded_correct", "grounded_correct_not_derivable")}
# PLAN 3.5: one metric per skill, main threshold first, then the sensitivity set; cell = (eval sets, group).
SKILLS = {
    "format": ("format_valid", "tool", (0.5, 0.3, 0.7)),
    "stop": ("stop_rate", "all", (0.5, 0.3, 0.7)),
    "tool_choice": ("tool_name_acc", "tool", (0.5, 0.3, 0.7)),
    "call_exact": ("call_exact", "tool", (0.5, 0.3, 0.7)),
    "decision": ("decision_bacc", "tool", (0.75, 0.65, 0.85)),
    "clarify": ("clarify_correct", "missing_info", (0.5, 0.3, 0.7)),
    "grounded": ("grounded_correct", "multi_turn", (0.5, 0.3, 0.7)),
}
SKILL_SETS = ("t_seen", "t_unseen")
FIELDS = ("run", "checkpoint", "seed", "step", "tokens_seen", "assistant_tokens_seen", "split", "eval_set", "group",
          "metric", "value", "ci_low", "ci_high")


# ---- Per-sample terms ----------------------------------------------------------------------

def terms(record: dict) -> dict:
    """Numerator and denominator each sample adds to every metric; a metric is sum(num) / sum(den) over its
    population, except decision_bacc (mean of two such ratios). Same definitions as run_eval.aggregate."""
    s = record["scores"]
    out = {"stop_rate": (s["im_end_stop"], 1), "eot_stop_rate": (s["eot_stop"], 1),
           "truncated_rate": (s["truncated"], 1), "role_break_rate": (s["role_break"], 1),
           "repetition_rate": (s["repetition_rate"], 1), "avg_len": (record["generated_tokens"], 1)}
    if "needs_tool" in s:
        need = s["needs_tool"]
        out.update({key: (s[key] if need else 0, need) for key in NEED_TOOL})
        out["hallucinated_tool"] = (s["hallucinated_tool"], 1)
        out["false_call_rate"] = (s["false_call"] if not need else 0, not need)
        out["called_need"] = (s["called"] if need else 0, need)
        out["silent_no_tool"] = (not s["called"] if not need else 0, not need)
        if "no_fitting_said" in s:
            out["no_fitting_said"] = (s["no_fitting_said"], 1)
    if "clarify_correct" in s:
        out["clarify_correct"] = (s["clarify_correct"], 1)
    if "grounded_rule" in s:
        out["grounded_correct"] = (s["grounded_rule"], 1)
        out["grounded_correct_not_derivable"] = (s["grounded_rule"] if not s["derivable"] else 0, not s["derivable"])
    return out


TERMS = (*BEHAVIOUR, *NEED_TOOL, "hallucinated_tool", "false_call_rate", "called_need", "silent_no_tool",
         "no_fitting_said", "clarify_correct", "grounded_correct", "grounded_correct_not_derivable")


def term_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(n x terms) numerators and denominators."""
    num = np.zeros((len(records), len(TERMS)))
    den = np.zeros((len(records), len(TERMS)))
    for row, record in enumerate(records):
        for name, (n, d) in terms(record).items():
            column = TERMS.index(name)
            num[row, column], den[row, column] = float(n), float(d)
    return num, den


def metric_values(num_sums: np.ndarray, den_sums: np.ndarray, metrics: tuple) -> dict:
    """Metric values from summed terms; the last axis is the term, any leading axes (replicates) are kept."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = num_sums / den_sums
    out = {}
    for name in metrics:
        if name == "decision_bacc":
            out[name] = 0.5 * (ratio[..., TERMS.index("called_need")] + ratio[..., TERMS.index("silent_no_tool")])
        else:
            out[name] = ratio[..., TERMS.index(name)]
    return out


# ---- Clusters and resampling ---------------------------------------------------------------

def first_user(item: dict) -> str:
    return next(turn["content"] for turn in item["turns"] if turn["role"] == "user").strip().lower()


def links(item: dict, items: dict) -> set:
    source = items.get(item["meta"].get("renamed_from"), item)
    keys = {("prompt", first_user(source))}
    if source["group"] in ("single", "select", "parallel"):
        keys |= {("tool", call["name"]) for call in source["expected_calls"]}
    elif source["group"] in ("missing_info", "multi_turn"):
        keys |= {("tool", tool["name"]) for tool in source["tools"]}
    return keys


def clusters(population: list[dict], items: dict) -> np.ndarray:
    """Cluster label per item: connected components of shared links (union-find)."""
    parent = list(range(len(population)))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    owner = {}
    for node, item in enumerate(population):
        for key in links(item, items):
            if key in owner:
                parent[find(node)] = find(owner[key])
            else:
                owner[key] = node
    roots = [find(node) for node in range(len(population))]
    labels = {root: label for label, root in enumerate(dict.fromkeys(roots))}
    return np.array([labels[root] for root in roots])


def resample_weights(labels: np.ndarray, name: str, reps: int = REPS, seed: int = SEED) -> np.ndarray:
    """(reps x n) weight of each item: how many times its cluster was drawn. Seeded per population name, so every
    checkpoint scored on that population gets the same draws."""
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    k = labels.max() + 1
    draws = rng.integers(0, k, size=(reps, k))
    counts = np.zeros((reps, k))
    np.add.at(counts, (np.arange(reps)[:, None], draws), 1)
    return counts[:, labels]


def interval(samples: np.ndarray, level: float = LEVEL) -> tuple[float, float]:
    tail = 100 * (1 - level) / 2
    low, high = np.nanpercentile(samples, [tail, 100 - tail])
    return float(low), float(high)


# ---- Populations ---------------------------------------------------------------------------

def cells(items: list[dict]) -> list[tuple[str, str, list[dict], tuple]]:
    """(eval set, group, items, metrics): every sample with behaviour metrics, the four tool groups together with
    tool metrics (so decision_bacc sees both kinds), and each group with its own metrics."""
    out = []
    for name in dict.fromkeys(item["set"] for item in items):
        rows = [item for item in items if item["set"] == name]
        out.append((name, "all", rows, BEHAVIOUR))
        tool_rows = [item for item in rows if item["group"] in TOOL_GROUPS]
        if tool_rows:
            out.append((name, "tool", tool_rows, TOOL))
        for group in dict.fromkeys(item["group"] for item in rows):
            group_rows = [item for item in rows if item["group"] == group]
            extra = TOOL if group in TOOL_GROUPS else GROUP_METRICS.get(group, ())
            out.append((name, group, group_rows, BEHAVIOUR + extra))
    return out


# ---- Checkpoints ---------------------------------------------------------------------------

def find_checkpoints(split: str, root: Path = RESULTS) -> list[dict]:
    """Every scored folder of the split: run, checkpoint name, snapshot meta (step 0 for base models), records."""
    found = []
    for path in sorted(root.glob("*/*/metrics.json")):
        folder = path.parent
        if not (folder / "outputs.jsonl").exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("split") != split:
            continue
        meta = summary.get("snapshot") or {}
        if not meta and summary.get("template") == "project" and not summary.get("few_shot"):   # base, zero-shot
            meta = {"step": 0, "tokens_seen": 0, "assistant_tokens_seen": 0}
        records = {record["id"]: record for line in (folder / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
                   if line for record in [json.loads(line)] if record["set"] != "p"}
        found.append({"run": folder.parent.name, "checkpoint": folder.name, "seed": meta.get("seed"),
                      "step": meta.get("step"), "tokens_seen": meta.get("tokens_seen"),
                      "assistant_tokens_seen": meta.get("assistant_tokens_seen"), "records": records})
    return found


def evaluate(checkpoints: list[dict], items: list[dict], baseline: str | None, reps: int = REPS):
    """Rows of all.csv and of diff_base.csv (paired difference from the baseline checkpoint, 'run/checkpoint')."""
    by_id = {item["id"]: item for item in items}
    base = next((c for c in checkpoints if f"{c['run']}/{c['checkpoint']}" == baseline), None)
    if baseline and base is None:
        raise SystemExit(f"Baseline {baseline} not found among the scored checkpoints.")
    rows, diffs = [], []
    for name, group, population, metrics in cells(items):
        weights = resample_weights(clusters(population, by_id), f"{name}/{group}", reps)
        ids = [item["id"] for item in population]
        prepared = []
        for checkpoint in checkpoints:
            missing = [i for i in ids if i not in checkpoint["records"]]
            if missing:
                raise SystemExit(f"{checkpoint['run']}/{checkpoint['checkpoint']} lacks {len(missing)} samples of {name}.")
            num, den = term_arrays([checkpoint["records"][i] for i in ids])
            point = metric_values(num.sum(0), den.sum(0), metrics)
            boot = metric_values(weights @ num, weights @ den, metrics)
            prepared.append((checkpoint, point, boot))
        base_entry = next((entry for entry in prepared if entry[0] is base), None)
        for checkpoint, point, boot in prepared:
            head = {key: checkpoint[key] for key in FIELDS[:6]}
            for metric in metrics:
                value = float(point[metric])
                if math.isnan(value):    # no sample of that kind in this population
                    continue
                low, high = interval(boot[metric])
                rows.append({**head, "split": None, "eval_set": name, "group": group, "metric": metric,
                             "value": value, "ci_low": low, "ci_high": high})
                if base_entry is not None:
                    base_value = float(base_entry[1][metric])
                    low, high = interval(boot[metric] - base_entry[2][metric])
                    diffs.append({**rows[-1], "base_value": base_value, "diff": value - base_value,
                                  "diff_low": low, "diff_high": high})
    return rows, diffs


RENAME_METRICS = ("format_valid", "call_exact", "tool_name_acc", "args_f1")


def rename_differences(checkpoints: list[dict], items: list[dict], reps: int = REPS) -> list[dict]:
    """Rows of rename.csv (H3): per T-rename set, the metric on the renamed prompts minus the metric on their own
    T-seen originals, with the CI of that difference. Both sides take the same resampled clusters (the rename cell's
    draws, so the renamed value's CI matches all.csv), so the difference is paired prompt by prompt."""
    by_id = {item["id"]: item for item in items}
    out = []
    for name in dict.fromkeys(item["set"] for item in items if item["meta"].get("renamed_from")):
        renamed = [item for item in items if item["set"] == name and item["meta"]["renamed_from"] in by_id]
        groups = [("tool", renamed)] + [(group, [item for item in renamed if item["group"] == group])
                                        for group in dict.fromkeys(item["group"] for item in renamed)]
        for group, population in groups:
            weights = resample_weights(clusters(population, by_id), f"{name}/{group}", reps)
            originals = [item["meta"]["renamed_from"] for item in population]
            for checkpoint in checkpoints:
                sides = []
                for ids in (originals, [item["id"] for item in population]):
                    missing = [i for i in ids if i not in checkpoint["records"]]
                    if missing:
                        raise SystemExit(f"{checkpoint['run']}/{checkpoint['checkpoint']} lacks {len(missing)} "
                                         f"samples paired in {name}.")
                    num, den = term_arrays([checkpoint["records"][i] for i in ids])
                    sides.append((metric_values(num.sum(0), den.sum(0), RENAME_METRICS),
                                  metric_values(weights @ num, weights @ den, RENAME_METRICS)))
                (seen, seen_boot), (value, boot) = sides
                head = {key: checkpoint[key] for key in FIELDS[:6]}
                for metric in RENAME_METRICS:
                    low, high = interval(boot[metric] - seen_boot[metric])
                    out.append({**head, "split": None, "eval_set": name, "group": group, "metric": metric,
                                "seen_value": float(seen[metric]), "value": float(value[metric]),
                                "diff": float(value[metric] - seen[metric]), "diff_low": low, "diff_high": high})
    return out


# ---- Emergence and go/no-go ----------------------------------------------------------------

def first_held(flags: list[bool]) -> tuple[int | None, str]:
    """PLAN 3.5: first index whose condition holds there and at the 2 following checkpoints. When it holds only from
    one of the last two checkpoints to the end, there are too few checkpoints left to confirm."""
    for index in range(len(flags)):
        if all(flags[index:index + 3]):
            return index, "reached" if index + 3 <= len(flags) else "unconfirmed"
    return None, "not_reached"


def emergence(rows: list[dict], diffs: list[dict], baseline: str) -> list[dict]:
    """t_reach per threshold and t_improve for every trained run; the curve starts at the baseline (step 0)."""
    base_run, base_checkpoint = baseline.split("/")
    value = {(r["run"], r["checkpoint"], r["eval_set"], r["group"], r["metric"]): r for r in rows}
    diff = {(r["run"], r["checkpoint"], r["eval_set"], r["group"], r["metric"]): r for r in diffs}
    runs = {}
    for row in rows:
        if row["run"] != base_run and row["step"] is not None:
            runs.setdefault(row["run"], {})[row["checkpoint"]] = (row["step"], row["assistant_tokens_seen"])
    out = []
    for run, found in runs.items():
        order = [(base_run, base_checkpoint, 0, 0)] + [(run, name, *found[name])
                                                       for name in sorted(found, key=lambda n: found[n][0])]
        for skill, (metric, group, thresholds) in SKILLS.items():
            for eval_set in SKILL_SETS:
                keys = [(r, c, eval_set, group, metric) for r, c, _, _ in order]
                if not all(key in value for key in keys):
                    continue
                values = [value[key]["value"] for key in keys]
                head = {"run": run, "skill": skill, "metric": metric, "eval_set": eval_set, "group": group}
                for theta in thresholds:
                    index, status = first_held([v >= theta for v in values])
                    out.append({**head, "kind": "t_reach", "threshold": theta, "status": status,
                                **_at(order, index)})
                if values[0] >= NEAR_CEILING:
                    out.append({**head, "kind": "t_improve", "threshold": IMPROVE, "status": "near_ceiling_at_base",
                                **_at(order, None)})
                    continue
                flags = [False] + [diff[key]["diff"] >= IMPROVE and diff[key]["diff_low"] > 0 for key in keys[1:]]
                index, status = first_held(flags)
                out.append({**head, "kind": "t_improve", "threshold": IMPROVE, "status": status, **_at(order, index)})
    return out


def _at(order: list, index: int | None) -> dict:
    if index is None:
        return {"step": None, "assistant_tokens_seen": None}
    return {"step": order[index][2], "assistant_tokens_seen": order[index][3]}


def go_no_go(diffs: list[dict], run: str) -> dict:
    """PLAN 6, Phase 2, read at the run's last checkpoint against the baseline. (a) Single T-seen format_valid and
    call_exact both gain >= 0.2 with the CI's lower bound > 0. (b) at least one of call_exact Single T-unseen,
    clarify_correct, grounded_correct (T-seen or T-unseen) has the CI's lower bound > 0."""
    rows = [row for row in diffs if row["run"] == run]
    last = max(row["step"] for row in rows)
    at = {(r["eval_set"], r["group"], r["metric"]): r for r in rows if r["step"] == last}
    core = [at[("t_seen", "single", metric)] for metric in ("format_valid", "call_exact")]
    hard = [at[key] for key in (("t_unseen", "single", "call_exact"),
                                ("t_seen", "missing_info", "clarify_correct"),
                                ("t_unseen", "missing_info", "clarify_correct"),
                                ("t_seen", "multi_turn", "grounded_correct"),
                                ("t_unseen", "multi_turn", "grounded_correct"))]
    return {"step": last, "core": core, "hard": hard,
            "a": all(r["diff"] >= IMPROVE and r["diff_low"] > 0 for r in core),
            "b": any(r["diff_low"] > 0 for r in hard)}


# ---- Output --------------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_go_no_go(result: dict) -> None:
    print(f"Go/no-go at step {result['step']} (difference from base, {LEVEL:.0%} CI):")
    for label, rows in (("(a) core", result["core"]), ("(b) hard", result["hard"])):
        for r in rows:
            print(f"  {label} {r['eval_set']:9} {r['group']:13} {r['metric']:17} {r['value']:.3f} - "
                  f"{r['base_value']:.3f} = {r['diff']:+.3f} [{r['diff_low']:+.3f}, {r['diff_high']:+.3f}]")
    print(f"  (a) {'met' if result['a'] else 'not met'}; (b) {'met' if result['b'] else 'not met'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--baseline", default="e0/base-zero-shot", help="run/checkpoint the differences are from.")
    parser.add_argument("--go-no-go", metavar="RUN", help="Print the model go/no-go criteria for this run.")
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    items = load_items(args.split, ROOT, probe=False)
    checkpoints = find_checkpoints(args.split, args.out)
    print(f"[stats] {len(checkpoints)} checkpoints on {args.split}: "
          + ", ".join(f"{c['run']}/{c['checkpoint']}" for c in checkpoints), flush=True)
    rows, diffs = evaluate(checkpoints, items, args.baseline, args.reps)
    for row in rows + diffs:
        row["split"] = args.split
    suffix = "" if args.split == "dev" else f"_{args.split}"
    write_csv(args.out / f"all{suffix}.csv", rows)
    write_csv(args.out / f"diff_base{suffix}.csv", diffs)
    found = emergence(rows, diffs, args.baseline)
    if found:
        write_csv(args.out / f"emergence{suffix}.csv", found)
    renames = rename_differences(checkpoints, items, args.reps)
    for row in renames:
        row["split"] = args.split
    if renames:
        write_csv(args.out / f"rename{suffix}.csv", renames)
    print(f"[stats] wrote {len(rows)} rows to all{suffix}.csv, {len(diffs)} to diff_base{suffix}.csv, "
          f"{len(found)} to emergence{suffix}.csv, {len(renames)} to rename{suffix}.csv")
    if args.go_no_go:
        print_go_no_go(go_no_go(diffs, args.go_no_go))


if __name__ == "__main__":
    main()
