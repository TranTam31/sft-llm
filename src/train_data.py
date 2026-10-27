"""Training data for one run: the D_chat/D_tool subset, the example order of E1/E2, the checkpoint schedule and the
dev examples for split validation loss (PLAN 3.7, 5.1).

The whole order is laid out before training, so the Trainer reads it sequentially: every seed's order is saved, and
the assistant tokens learned by each optimizer step are known in advance, which places checkpoints on a log scale of
assistant tokens rather than steps.
"""

from itertools import accumulate
import json
from pathlib import Path
import random

from src.tool_calls import format_tool_calls


ORDERS = ("mix", "seq")  # E2 shuffles D_chat and D_tool together; E1 learns all of D_chat, then all of D_tool.


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def take_fraction(examples: list[dict], fraction: float) -> list[dict]:
    """The first share of a file; D_chat and D_tool are written shuffled, so a prefix keeps the group mix."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    return examples[:round(len(examples) * fraction)]


def _epochs(keys: list, epochs: int, rng) -> list:
    order = []
    for _ in range(epochs):
        shuffled = list(keys)
        rng.shuffle(shuffled)
        order += shuffled
    return order


def _whole_batches(keys: list, batch: int) -> list:
    return keys[:len(keys) - len(keys) % batch]


def build_order(n_chat: int, n_tool: int, *, order: str, epochs: int, batch: int, seed: int) -> list[tuple]:
    """(source, index) per position. Every example appears `epochs` times; an incomplete last batch of a stage is
    dropped (drop_last per stage, PLAN 5.1), so no optimizer step mixes the chat and tool stages of E1."""
    rng = random.Random(f"{seed}:order:{order}")
    chat = [("chat", index) for index in range(n_chat)]
    tool = [("tool", index) for index in range(n_tool)]
    if order == "mix":
        return _whole_batches(_epochs(chat + tool, epochs, rng), batch)
    if order == "seq":
        return _whole_batches(_epochs(chat, epochs, rng), batch) + _whole_batches(_epochs(tool, epochs, rng), batch)
    raise ValueError(f"order must be one of {ORDERS}")


def step_tokens(sequence: list, assistant_tokens: dict, *, batch: int) -> list[int]:
    """Assistant tokens learned by each optimizer step: the sum over the step's `batch` examples."""
    return [sum(assistant_tokens[key] for key in sequence[start:start + batch])
            for start in range(0, len(sequence), batch)]


def checkpoint_schedule(tokens_per_step: list[int], *, start: int, factor: float) -> list[dict]:
    """Steps (1-based) where cumulative assistant tokens first reach start, start x factor, …, plus the last step.
    Targets are absolute token counts, so runs with other orders or batch sizes are compared at the same points."""
    if start <= 0 or factor <= 1:
        raise ValueError("start must be positive and factor above 1.")
    cumulative = list(accumulate(tokens_per_step))
    schedule, target = [], start
    for step, seen in enumerate(cumulative, start=1):
        while seen >= target:
            if not schedule or schedule[-1]["step"] != step:
                schedule.append({"step": step, "target_tokens": target, "assistant_tokens": seen})
            target *= factor
    if cumulative and (not schedule or schedule[-1]["step"] != len(cumulative)):
        schedule.append({"step": len(cumulative), "target_tokens": None, "assistant_tokens": cumulative[-1]})
    return schedule


def add_stage_snapshots(schedule: list[dict], tokens_per_step: list[int], sequence: list, *, batch: int, start: int,
                        factor: float) -> list[dict]:
    """E1 (seq) also snapshots its last chat step, the "chat only" point of PLAN 5.1 (`boundary`), and a second grid
    counting assistant tokens from that boundary (`stage_target_tokens`): on the cumulative grid alone, D_chat's ~70%
    of the tokens leaves the tool stage with two snapshots. A step on both grids is saved once, carrying both targets;
    a boundary past the run's last step adds nothing."""
    boundary = sum(source == "chat" for source, _ in sequence) // batch
    if not 0 < boundary < len(tokens_per_step):
        return schedule
    cumulative = list(accumulate(tokens_per_step))
    entries = {entry["step"]: dict(entry) for entry in schedule}
    added = [(boundary, {"boundary": True})]
    added += [(boundary + entry["step"], {"stage_target_tokens": entry["target_tokens"]})
              for entry in checkpoint_schedule(tokens_per_step[boundary:], start=start, factor=factor)
              if entry["target_tokens"] is not None]
    for step, keys in added:
        entry = entries.setdefault(step, {"step": step, "target_tokens": None, "assistant_tokens": cumulative[step - 1]})
        entry.update(keys)
    return [entries[step] for step in sorted(entries)]


def val_examples(items: list[dict]) -> dict[str, list[dict]]:
    """Dev items that carry a reference answer, as conversations for validation loss: chat (C) and tool, split by
    seen and unseen tools. Calls are serialized as the train data serializes them; Missing-info has no reference."""
    sets = {"c": "chat", "t_seen": "tool_seen", "t_unseen": "tool_unseen"}
    result = {name: [] for name in sets.values()}
    for item in items:
        name = sets.get(item.get("set"))
        if name is None:
            continue
        if item.get("expected_calls"):
            answer = format_tool_calls(item["expected_calls"])
        elif item.get("reference") is not None:
            answer = item["reference"]
        else:
            continue
        result[name].append({"id": item["id"], "tools": item["tools"],
                             "turns": item["turns"] + [{"role": "assistant", "content": answer}]})
    return result
