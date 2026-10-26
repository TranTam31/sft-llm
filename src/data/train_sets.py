"""Build the fixed train sets D_chat and D_tool from the split train pool (PLAN 4.3, 4.5, 5.1).

Run: python -m src.data.train_sets --config configs/data.yaml [--from-hub USER/REPO] [--push-to USER/REPO]
Reads data/split/{train,tools}.jsonl, the frozen eval sets (eval_sets/{dev,test}/*.jsonl) and P (eval_sets/p.jsonl);
writes data/train/{d_chat,d_tool}.jsonl, data/train/drafts/no_tool.jsonl (Git-ignored) and data/TRAIN.md (tracked).

D_tool mixes, by share of n_tool: xLAM calls (single / select / parallel), Glaive Multi-turn, Missing-info and
No-tool. Glaive Missing-info keeps only conversations whose later call needs a required value the first user turn
does not state; the Glaive pool runs short, and the rest is filled with xLAM requests rewritten to lack a required
value (drafted like the eval's rewrites, src.eval.build_sets.rewrite_draft; labels in
data/train/labels/missing_info_rewrite.jsonl hold the rewritten query and the question back). Without the rewrite
settings the shortfall goes to the call groups instead. No-tool is drafted the way the eval
drafts it (src.eval.build_sets.no_tool_drafts), from held-in chat prompts D_chat does not use and from xLAM
requests D_tool does not use, offered seen tools only; a labelling pass (data/train/labels/no_tool.jsonl, on the
Hub) keeps each draft that is a self-contained request no offered tool helps with; a label names the tools it saw
and counts only for a draft offering exactly those. D_tool is written only once enough drafts are labelled.

No prompt repeats another train prompt or any dev, test or P prompt: exact or near duplicates (MinHash, the split's
settings) of the first user turn are skipped.
"""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys

from src.data.split import embed_texts, tool_text, write_jsonl
from src.eval.build_sets import (
    CALL_GROUPS, NO_TOOL_KINDS, GrowingIndex, apply_rewrite, call_item, clarify_draft, no_tool_drafts, rewrite_draft,
    side_tools, user_text, with_id,
)
from src.eval.scorers import score_clarify


TOOL_GROUPS = (*CALL_GROUPS, "multi_turn", "missing_info", "no_tool")


def first_user(item: dict) -> str:
    return next(turn["content"] for turn in item["turns"] if turn["role"] == "user")


def asks_for_missing(example: dict) -> bool:
    """A Glaive Missing-info conversation worth learning: the later call needs a required value that the first user
    turn does not state, so asking back was right (eval labelling dropped about a sixth for 'no argument missing')."""
    draft = clarify_draft(example)
    if draft is None:
        return False
    call = draft["draft"]["later_call"]
    schema = next((tool.get("parameters") or {} for tool in example["tools"] if tool["name"] == call["name"]), {})
    required = schema.get("required")  # A few Glaive schemas carry a bare boolean here: no usable list.
    return isinstance(required, list) and bool(set(draft["draft"]["missing_param_candidates"]) & set(required))


def take(candidates: list, n: int, index: GrowingIndex, *, rng) -> tuple[list, Counter]:
    """Like build_sets.pick, keyed on the first user turn: whole conversations stay distinct by their opening."""
    order = list(candidates)
    rng.shuffle(order)
    picked, skipped = [], Counter()
    for item in order:
        if len(picked) == n:
            break
        verdict = index.match(first_user(item))
        if verdict:
            skipped[verdict] += 1
            continue
        index.add(first_user(item))
        picked.append(item)
    return picked, skipped


def targets(settings: dict) -> dict:
    """Items per D_tool group from n_tool and the shares (PLAN 5.1: about 20% No-tool)."""
    n_tool, shares = settings["n_tool"], settings["shares"]
    counts = {group: round(n_tool * shares[group]) for group in ("multi_turn", "missing_info", "no_tool")}
    n_call = n_tool - sum(counts.values())
    split = settings["call_split"]
    for group in CALL_GROUPS[:-1]:
        counts[group] = round(n_call * split[group])
    counts[CALL_GROUPS[-1]] = n_call - sum(counts[group] for group in CALL_GROUPS[:-1])
    return counts


def spread(shortfall: int, counts: dict) -> dict:
    """Missing-info the pool cannot supply goes to the call groups in proportion to their targets."""
    total = sum(counts[group] for group in CALL_GROUPS)
    extra = {group: shortfall * counts[group] // total for group in CALL_GROUPS}
    extra[CALL_GROUPS[0]] += shortfall - sum(extra.values())
    return {group: counts[group] + extra.get(group, 0) for group in counts}


def train_item(example: dict, group: str) -> dict:
    return {"id": example["id"], "source": example["source"], "group": group, "tools": example["tools"],
            "turns": example["turns"], "meta": example["meta"]}


def draft_key(draft: dict) -> tuple:
    """A No-tool label holds for one request and the tools it was offered (names, sorted)."""
    return draft["id"], tuple(sorted(tool["name"] for tool in draft["tools"]))


def label_key(label: dict) -> tuple:
    return label["id"], tuple(label["tools"])


def no_tool_example(draft: dict) -> dict:
    """A labelled draft as a train conversation: the request, then the reference reply."""
    return {"id": draft["id"], "source": draft["meta"]["source"], "group": "no_tool", "kind": draft["kind"],
            "tools": draft["tools"], "turns": draft["turns"] + [{"role": "assistant", "content": draft["reference"]}],
            "meta": {**draft["meta"], "n_tools": len(draft["tools"])}}


def rewrite_drafts(by_group: dict, taken: set, n: int, index: GrowingIndex, *, rng) -> tuple[list, Counter]:
    """Missing-info drafts from xLAM single/select requests no other part of the train sets uses: the request states
    a required value, which the labelling pass removes (src.eval.build_sets.rewrite_draft)."""
    candidates = [{**with_id(example, item, "train"), "id": f"train:{example['id']}:rewrite"}
                  for group in ("single", "select") for example in by_group[group] if example["id"] not in taken
                  for item in [rewrite_draft(example)] if item is not None]
    return take(candidates, n, index, rng=rng)


def rewrite_example(draft: dict, label: dict) -> dict:
    """A labelled rewrite as a train conversation: the rewritten request, then the question back. The label is
    checked the way the eval's rewrites are, and the reply must pass score_clarify."""
    item = apply_rewrite(draft, label)  # raises when the param is not removable or the query still states it
    reply = label["reply"]
    if not score_clarify(reply, missing_param=item["missing_param"], synonyms=item["synonyms"])["clarify_correct"]:
        raise ValueError(f"{draft['id']}: the reply does not ask for {item['missing_param']}")
    return {"id": draft["id"], "source": draft["meta"]["source"], "group": "missing_info", "tools": draft["tools"],
            "turns": [*item["turns"], {"role": "assistant", "content": reply}],
            "meta": {**item["meta"], "missing_param": item["missing_param"], "synonyms": item["synonyms"]}}


def run(pool: list, tools: list, eval_items: list, config: dict, *, labels=None, rewrite_labels=None,
        embed=embed_texts, measure=None) -> dict:
    """measure(example) -> token count or None when over max_seq_len; only new No-tool conversations need it."""
    settings, seed = config["train_sets"], config["seed"]
    rng = random.Random(seed)
    index = GrowingIndex(seed=seed, **config["split"]["minhash"])
    for item in eval_items:
        # Train is keyed on first turns, so a multi-turn eval item must block its opening too, not only its
        # joined user turns.
        for text in {user_text(item), first_user(item)}:
            index.add(text)
    by_group = defaultdict(list)
    for example in pool:
        by_group[example["group"]].append(example)
    stats, counts = {}, targets(settings)

    d_chat, skipped = take(by_group["chat"], settings["n_chat"], index, rng=rng)
    stats["chat"] = {"available": len(by_group["chat"]), "wanted": settings["n_chat"], "picked": len(d_chat),
                     "skipped": dict(skipped)}

    picked = {}
    clarify = [example for example in by_group["clarify_candidate"] if asks_for_missing(example)]
    picked["missing_info"], skipped = take(clarify, counts["missing_info"], index, rng=rng)
    stats["missing_info"] = {"available": len(by_group["clarify_candidate"]), "filtered": len(clarify),
                             "wanted": counts["missing_info"], "picked": len(picked["missing_info"]),
                             "skipped": dict(skipped)}
    shortfall = counts["missing_info"] - len(picked["missing_info"])
    target_counts = counts
    # The call groups are drawn at their size before any rewrite fills the shortfall, so No-tool (drafted from what
    # they leave) stays as labelled; filled rewrites then take the place of the last call items of each group.
    counts = spread(shortfall, counts)
    picked["multi_turn"], skipped = take(by_group["multi_turn"], counts["multi_turn"], index, rng=rng)
    stats["multi_turn"] = {"available": len(by_group["multi_turn"]), "wanted": counts["multi_turn"],
                           "picked": len(picked["multi_turn"]), "skipped": dict(skipped)}
    for group in CALL_GROUPS:
        candidates = [example for example in by_group[group] if call_item(example) is not None]
        picked[group], skipped = take(candidates, counts[group], index, rng=rng)
        stats[group] = {"available": len(by_group[group]), "wanted": counts[group], "picked": len(picked[group]),
                        "skipped": dict(skipped)}

    # No-tool drafts last, from what the other groups left: chat prompts outside D_chat, single requests outside
    # D_tool. Offered tools are seen tools that the train pool offers.
    used = {example["id"] for items in (d_chat, *picked.values()) for example in items}
    prompts = [example for example in by_group["chat"] if example["id"] not in used and len(example["turns"]) > 1
               and example["turns"][0]["role"] == "user" and example["turns"][1]["role"] == "assistant"]
    singles = [with_id(example, item, "train") for example in by_group["single"] if example["id"] not in used
               for item in [call_item(example)] if item is not None and len(item["expected_calls"]) == 1]
    print(f"[train_sets] embedding {len(tools)} tools and {len(prompts)} prompts…", flush=True)
    vectors = embed([tool["text"] for tool in tools], config["split"]["embedder"])
    prompt_vectors = embed([example["turns"][0]["content"] for example in prompts], config["split"]["embedder"])
    catalog = {tool["text"]: (tool["cluster"], vectors[row]) for row, tool in enumerate(tools)}
    offer = side_tools(pool, tools, vectors, {}, unseen=False)
    # Own random streams: a change in an earlier group does not redraw every No-tool offer, which would void
    # labels (a label holds for the tools it saw).
    drafts, stats["no_tool_drafts"] = no_tool_drafts(
        "train", offer, prompts, prompt_vectors, singles, catalog, settings, index, random.Random(f"{seed}:no_tool"),
        want=counts["no_tool"], offer_rng=lambda draft_id: random.Random(f"{seed}:offer:{draft_id}"))

    rewrite_settings = settings.get("missing_info_rewrite")
    rewrites, kept_rewrites = [], []
    if rewrite_settings and shortfall:
        taken = used | {draft["meta"]["source_id"] for draft in drafts}
        rewrites, skipped = rewrite_drafts(by_group, taken, math.ceil(shortfall * rewrite_settings["draft_margin"]),
                                           index, rng=random.Random(f"{seed}:rewrite"))
        stats["missing_info_rewrite"] = {"wanted": shortfall, "drafts": len(rewrites), "skipped": dict(skipped),
                                         "labelled": 0, "ok": 0, "picked": 0, "repeated": 0, "too_long": 0}
        if rewrite_labels is not None:
            by_id = {label["id"]: label for label in rewrite_labels}
            stat = stats["missing_info_rewrite"]
            stat["labelled"] = sum(draft["id"] in by_id for draft in rewrites)
            stat["ok"] = sum(by_id.get(draft["id"], {}).get("ok", False) for draft in rewrites)
            for draft in rewrites:
                if len(kept_rewrites) == shortfall:
                    break
                label = by_id.get(draft["id"])
                if not (label and label.get("ok")):
                    continue
                example = rewrite_example(draft, label)
                if index.match(label["query"]):
                    stat["repeated"] += 1
                    continue
                n_tokens = measure(example) if measure else 0
                if n_tokens is None:
                    stat["too_long"] += 1
                    continue
                index.add(label["query"])
                example["meta"]["n_tokens"] = n_tokens
                kept_rewrites.append(example)
            stat["picked"] = len(kept_rewrites)
    if kept_rewrites:
        picked["missing_info"] += kept_rewrites
        final = spread(shortfall - len(kept_rewrites), target_counts)
        for group in CALL_GROUPS:
            picked[group] = picked[group][:final[group]]
            stats[group]["wanted"] = final[group]
            stats[group]["picked"] = len(picked[group])
        counts = final
    rewrites_done = not (rewrite_settings and shortfall) or len(kept_rewrites) == shortfall

    stats["no_tool"] = {"wanted": counts["no_tool"], "drafts": len(drafts), "labelled": 0, "picked": 0}
    d_tool = None
    if labels is not None:
        labelled = {label_key(label) for label in labels}
        ok = {label_key(label) for label in labels if label.get("ok")}
        keys = {draft_key(draft) for draft in drafts}
        stats["no_tool"]["labelled"] = len(keys & labelled)
        stats["no_tool"]["ok"] = len(keys & ok)
        stats["no_tool"]["unmatched_labels"] = len(labelled - keys)
        n_hard = round(counts["no_tool"] * settings["no_tool"]["hard_fraction"] / 2)
        want = {"related": n_hard, "no_fitting_tool": n_hard, "plain": counts["no_tool"] - 2 * n_hard}
        too_long = Counter()
        picked["no_tool"] = []
        for kind in NO_TOOL_KINDS:
            kept = []
            for draft in drafts:
                if len(kept) == want[kind]:
                    break
                if draft["kind"] != kind:
                    continue
                # Length first: drafts over max_seq_len are counted even when the labelling pass skipped them.
                example = no_tool_example(draft)
                n_tokens = measure(example) if measure else 0
                if n_tokens is None:
                    too_long[kind] += 1
                    continue
                if draft_key(draft) not in ok:
                    continue
                example["meta"]["n_tokens"] = n_tokens
                kept.append(example)
            stats["no_tool"][kind] = {"wanted": want[kind], "picked": len(kept), "too_long": too_long[kind]}
            picked["no_tool"] += kept
        stats["no_tool"]["picked"] = len(picked["no_tool"])
        if rewrites_done and all(stats["no_tool"][kind]["picked"] == want[kind] for kind in NO_TOOL_KINDS):
            d_tool = [train_item(example, group) for group in TOOL_GROUPS if group != "no_tool"
                      for example in picked[group]] + picked["no_tool"]
            random.Random(f"{seed}:d_tool").shuffle(d_tool)  # A prefix (the pilot's 10%) keeps the mix.
    d_chat = [train_item(example, "chat") for example in d_chat]
    random.Random(f"{seed}:d_chat").shuffle(d_chat)
    return {"d_chat": d_chat, "d_tool": d_tool, "drafts": drafts, "rewrite_drafts": rewrites, "stats": stats,
            "targets": counts, "n_eval_prompts": len(eval_items)}


def summarize(items: list) -> dict:
    tokens = [item["meta"].get("n_tokens", 0) for item in items]
    assistant = [item["meta"].get("n_assistant_tokens") for item in items]
    return {"n": len(items), "median_tokens": statistics.median(tokens) if tokens else 0,
            "assistant_tokens": sum(value or 0 for value in assistant),
            "assistant_missing": sum(value is None for value in assistant)}


def render_report(result: dict, config: dict, hashes: dict) -> str:
    settings, stats = config["train_sets"], result["stats"]
    lines = ["# Báo cáo tập train", "",
             "Sinh tự động bởi `python -m src.data.train_sets`. Đừng sửa tay; chạy lại để cập nhật.", "",
             f"Seed {config['seed']}. D_chat {settings['n_chat']} mẫu, D_tool {settings['n_tool']} mẫu (PLAN 5.1; số "
             "chính thức chốt sau pilot, pilot lấy một phần đầu của mỗi file vì thứ tự đã xáo). Không prompt nào lặp lại "
             f"prompt train khác hay {result['n_eval_prompts']} prompt của dev, test và P: lượt user đầu trùng exact "
             "hoặc gần trùng (MinHash, cùng tham số với bước tách) thì bị bỏ qua.", "",
             "## Tỉ lệ D_tool", "",
             "Đã chốt (2026-10-24): 50% gọi tool xLAM (single / select / parallel chia gần đều), 20% Multi-turn, "
             "10% Missing-info, 20% No-tool. Missing-info lấy trước từ Glaive (giữ hội thoại có lời gọi sau cần một "
             "tham số bắt buộc mà lượt user đầu chưa nêu). Từ 2026-10-26, phần Glaive thiếu được lấp bằng câu hỏi xLAM "
             "viết lại cho thiếu một tham số bắt buộc, giống cách dựng bộ eval, vì pilot cho thấy model chỉ học hỏi lại "
             "trên kiểu câu Glaive (mẫu xLAM viết lại đúng 0/36). Nhóm gọi tool vẫn được rút với cỡ như trước khi lấp, "
             "để nháp No-tool không đổi; mẫu viết lại thay chỗ các mẫu cuối của mỗi nhóm gọi tool.", "",
             "| Nhóm | Pool | Sau lọc | Cần | Lấy | Bỏ vì trùng |", "|---|---|---|---|---|---|"]
    for group in ("chat", *CALL_GROUPS, "multi_turn", "missing_info"):
        stat = stats[group]
        skipped = ", ".join(f"{kind}: {count}" for kind, count in stat["skipped"].items()) or "–"
        lines.append(f"| {group} | {stat['available']} | {stat.get('filtered', '–')} | {stat['wanted']} | "
                     f"{stat['picked']} | {skipped} |")
    if "missing_info_rewrite" in stats:
        stat = stats["missing_info_rewrite"]
        lines += ["", "## Missing-info viết lại từ xLAM", "",
                  "Nháp: câu hỏi xLAM single/select chưa dùng ở đâu trong tập train, có nêu giá trị của một tham số bắt "
                  "buộc (`src.eval.build_sets.rewrite_draft`). Nhãn (`labels/missing_info_rewrite.jsonl`): câu hỏi viết "
                  "lại không còn giá trị đó, `missing_param`, từ đồng nghĩa và câu hỏi lại của assistant. Kiểm như bộ "
                  "eval (`apply_rewrite`), câu hỏi lại phải đạt `score_clarify`, câu viết lại không được trùng prompt "
                  "train/eval.", "",
                  "| Cần | Nháp | Có nhãn | Đạt | Lấy | Bỏ vì trùng | Bỏ vì quá dài |", "|---|---|---|---|---|---|---|",
                  f"| {stat['wanted']} | {stat['drafts']} | {stat['labelled']} | {stat['ok']} | {stat['picked']} | "
                  f"{stat['repeated']} | {stat['too_long']} |"]
    no_tool = stats["no_tool"]
    lines += ["", "## No-tool (PLAN 4.3)", "",
              "Nháp dựng như bộ eval: `plain` là prompt chat (ngoài D_chat) kèm tool ở các cụm xa chủ đề nhất; `related` "
              "kèm tool cùng cụm gần prompt nhất; `no_fitting_tool` là câu hỏi xLAM single (ngoài D_tool) chỉ kèm tool "
              "ở cụm xa, trả lời bằng câu cố định. Chỉ dùng tool seen có trong pool train. Câu trả lời của `plain` và "
              "`related` là lượt assistant đầu của SmolTalk. Lượt gán nhãn giữ nháp là yêu cầu tự đủ ngữ cảnh, trả lời "
              "được bằng kiến thức chung (hoặc với 3b: thật sự cần dữ liệu hay hành động bên ngoài) và không tool nào "
              "giúp được.", "",
              f"Nháp: {no_tool['drafts']}; có nhãn: {no_tool['labelled']}; đạt: {no_tool.get('ok', 0)}; nhãn không khớp "
              f"nháp nào: {no_tool.get('unmatched_labels', 0)}.", "",
              "| Loại | Cần | Lấy | Bỏ vì quá dài |", "|---|---|---|---|"]
    for kind in NO_TOOL_KINDS:
        stat = no_tool.get(kind, {"wanted": "–", "picked": 0, "too_long": 0})
        lines.append(f"| {kind} | {stat['wanted']} | {stat['picked']} | {stat['too_long']} |")
    lines += ["", "| Nháp | Lấy | Cần | Bỏ vì trùng |", "|---|---|---|---|"]
    for kind, stat in stats["no_tool_drafts"].items():
        skipped = ", ".join(f"{name}: {count}" for name, count in stat["skipped"].items()) or "–"
        lines.append(f"| {kind} | {stat['picked']} | {stat['wanted']} | {skipped} |")
    lines += ["", "## Kích thước", "", "| File | Nhóm | Mẫu | Độ dài trung vị | Token assistant |", "|---|---|---|---|---|"]
    for name in ("d_chat", "d_tool"):
        items = result[name] or []
        for group in sorted({item["group"] for item in items}):
            summary = summarize([item for item in items if item["group"] == group])
            lines.append(f"| {name} | {group} | {summary['n']} | {summary['median_tokens']:.0f} | "
                         f"{summary['assistant_tokens']} |")
    if result["d_tool"] is None:
        lines += ["", "**D_tool chưa ghi:** chưa đủ nháp No-tool có nhãn."]
    if settings.get("notes"):
        lines += ["", "## Ghi chú", "", *[f"- {note}" for note in settings["notes"]]]
    lines += ["", "## Hash", "", "| File | SHA-256 |", "|---|---|",
              *[f"| `{name}` | `{digest}` |" for name, digest in sorted(hashes.items())], ""]
    return "\n".join(lines)


def load_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_eval_items(directory: Path) -> list:
    return [item for path in sorted(directory.glob("*/*.jsonl")) if path.parent.name in ("dev", "test")
            for item in load_jsonl(path)] + load_jsonl(directory / "p.jsonl")


def length_measurer(config: dict):
    from transformers import AutoTokenizer

    from src.playground import resolve_model
    from src.sft_data import SequenceTooLongError, configure_tokenizer, encode_sft

    model_id, revision = resolve_model(config["tokenizer_model"])
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(model_id, revision=revision))

    def measure(example):
        try:
            feature = encode_sft(tokenizer, example["turns"], tools=example["tools"], max_seq_len=config["max_seq_len"])
        except SequenceTooLongError:
            return None
        example["meta"]["n_assistant_tokens"] = sum(label != -100 for label in feature["labels"])
        return len(feature["input_ids"])
    return measure


def cached_embed(path: Path):
    """embed_texts with an on-disk cache keyed by text; reruns embed only texts not seen before."""
    import hashlib

    import numpy as np

    def embed(texts, embedder):
        def key(text):
            return hashlib.sha256(f"{embedder['id']}@{embedder['revision']}\n{text}".encode()).hexdigest()

        cache = dict(np.load(path)) if path.exists() else {}
        missing = sorted({text for text in texts if key(text) not in cache})
        if missing:
            cache.update(zip(map(key, missing), embed_texts(missing, embedder)))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **cache)
        return np.stack([cache[key(text)] for text in texts])
    return embed


def fetch(repo_id: str, split_dir: Path, labels_dir: Path):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, repo_type="dataset", local_dir=str(split_dir.parent),
                      allow_patterns=[f"{split_dir.name}/train.jsonl", f"{split_dir.name}/tools.jsonl"])
    snapshot_download(repo_id, repo_type="dataset", local_dir=str(labels_dir.parent.parent),
                      allow_patterns=[f"{labels_dir.parent.name}/{labels_dir.name}/*.jsonl"])


def push(directory: Path, repo_id: str):
    from huggingface_hub import upload_folder

    upload_folder(folder_path=str(directory), repo_id=repo_id, repo_type="dataset", path_in_repo="train",
                  allow_patterns=["d_chat.jsonl", "d_tool.jsonl", "drafts/*.jsonl", "labels/*.jsonl"],
                  commit_message="Train sets D_chat and D_tool (src.data.train_sets)")


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--from-hub", help="Download data/split/{train,tools} and the train labels when missing.")
    parser.add_argument("--push-to", help="Private HF dataset repo; files go under train/.")
    parser.add_argument("--report", default="data/TRAIN.md")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    split_dir = Path(config["split"]["output_dir"])
    output_dir = Path(config["train_sets"]["output_dir"])
    labels_path = output_dir / "labels" / "no_tool.jsonl"
    if args.from_hub and not (split_dir / "train.jsonl").exists():
        fetch(args.from_hub, split_dir, labels_path.parent)
    pool, tools = load_jsonl(split_dir / "train.jsonl"), load_jsonl(split_dir / "tools.jsonl")
    labels = load_jsonl(labels_path) if labels_path.exists() else None
    rewrite_path = output_dir / "labels" / "missing_info_rewrite.jsonl"
    rewrite_labels = load_jsonl(rewrite_path) if rewrite_path.exists() else None
    result = run(pool, tools, load_eval_items(Path(config["eval_sets"]["output_dir"])), config, labels=labels,
                 rewrite_labels=rewrite_labels, embed=cached_embed(output_dir / "cache" / "embeddings.npz"),
                 measure=length_measurer(config))
    (output_dir / "drafts").mkdir(parents=True, exist_ok=True)
    hashes = {"d_chat.jsonl": write_jsonl(output_dir / "d_chat.jsonl", result["d_chat"]),
              "drafts/no_tool.jsonl": write_jsonl(output_dir / "drafts" / "no_tool.jsonl", result["drafts"])}
    if result["rewrite_drafts"]:
        hashes["drafts/missing_info_rewrite.jsonl"] = write_jsonl(output_dir / "drafts" / "missing_info_rewrite.jsonl",
                                                                 result["rewrite_drafts"])
    if result["d_tool"] is not None:
        hashes["d_tool.jsonl"] = write_jsonl(output_dir / "d_tool.jsonl", result["d_tool"])
    Path(args.report).write_text(render_report(result, config, hashes), encoding="utf-8")
    for name, digest in hashes.items():
        print(f"[train_sets] {name}: {digest[:12]}", flush=True)
    if result["d_tool"] is None:
        sys.exit(f"[train_sets] D_tool not written: label the No-tool and Missing-info rewrite drafts in "
                 f"{output_dir}/drafts first")
    if args.push_to:
        push(output_dir, args.push_to)


if __name__ == "__main__":
    main()
