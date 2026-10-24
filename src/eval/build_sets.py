"""Build the rule-labelled eval sets from the split pools and split each into dev and test (PLAN 3.1, 3.2, 4.2, 4.4).

Run: python -m src.eval.build_sets --config configs/data.yaml [--from-hub USER/REPO] [--push-to USER/REPO]
Reads data/split/*.jsonl, writes eval_sets/{dev,test}/*.jsonl, eval_sets/drafts/*.jsonl (Git-ignored)
and eval_sets/REPORT.md (tracked).

Labelled here, from the data itself:
- t_seen / t_unseen, groups single / select / parallel: the prompt is the user turn, the answer is the calls.
  t_unseen drops every example offering a tool that sits close to a seen tool (PLAN 4.2 item 6), and caps
  the share of each called cluster per split (unseen_cluster_cap), topping up from the pool last.
- t_rename_nonsense: t_seen with tool and parameter names replaced consistently (PLAN 4.4, second level).
- c: held-out SmolTalk chat; the prompt is every turn before the last assistant reply.
Drafted for a labelling pass, never scored as they are: Missing-info (missing_param candidates) and
Multi-turn (values from the tool result that the reference answer repeats), plus xLAM requests to rewrite
without one required value, since Glaive's Missing-info prompts repeat too much. t_rename_synonym (PLAN 4.4, first
level) renames t_seen with the labelled rename_synonyms table; P is written by hand. No-tool (PLAN 4.3) is drafted from held-out
chat the labelling pass kept and from xLAM requests offered only far tools, then checked draft by draft.
"""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path
import random
import re
import string
import sys

import numpy as np

from src.data.split import NearDuplicateIndex, embed_texts, normalize_text, shingles, tool_text, write_jsonl
from src.eval.scorers import _normalize
from src.tool_calls import ToolCallParseError, parse_tool_calls


CALL_GROUPS = ("single", "select", "parallel")
SPLITS = ("test", "dev")


# ---- Items -------------------------------------------------------------------------------

def plain(value):
    """parse_tool_calls keeps decimals exact as Decimal; items are written as plain JSON numbers."""
    if isinstance(value, Decimal):
        return float(value)  # Only JSON floats are parsed as Decimal; integers stay int.
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


def call_item(example: dict) -> dict | None:
    """First user turn and the calls that answer it; None when the first reply is not a call."""
    turns = example["turns"]
    if len(turns) < 2 or turns[0]["role"] != "user" or turns[1]["role"] != "assistant":
        return None
    try:
        calls = parse_tool_calls(turns[1]["content"])
    except ToolCallParseError:
        return None
    if not calls:
        return None
    return {"group": example["group"], "tools": example["tools"], "turns": turns[:1], "expected_calls": plain(calls)}


def chat_item(example: dict) -> dict:
    return {"group": "chat", "tools": [], "turns": example["turns"][:-1], "reference": example["turns"][-1]["content"]}


def _leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item)
    elif value is not None and not isinstance(value, bool):
        yield value


def _mentions(text: str, value) -> bool:
    words = _normalize(str(value))
    return bool(words) and f" {words} " in f" {_normalize(text)} "


def clarify_draft(example: dict) -> dict | None:
    """Missing-info: the first user turn, then a question back. Candidates for missing_param are the
    arguments of the later call whose value the first user turn does not contain."""
    turns = example["turns"]
    if len(turns) < 2 or turns[0]["role"] != "user" or turns[1]["role"] != "assistant":
        return None
    later = [turn for turn in turns[2:] if turn["role"] == "assistant" and turn["content"].startswith("<tool_call>")]
    try:
        call = parse_tool_calls(later[0]["content"])[0] if later else None
    except ToolCallParseError:
        call = None
    if call is None:
        return None
    candidates = [key for key, value in call["arguments"].items() if not _mentions(turns[0]["content"], value)]
    return {"group": "missing_info", "tools": example["tools"], "turns": turns[:1],
            "draft": {"missing_param_candidates": candidates, "reference_question": turns[1]["content"],
                      "later_call": plain(call)}}


def multi_turn_draft(example: dict) -> dict | None:
    """Multi-turn: everything up to the first tool result; the reply after it is the reference.
    Required-value candidates are scalar values of the result that the reference repeats."""
    turns = example["turns"]
    position = next((index for index, turn in enumerate(turns) if turn["role"] == "tool"), None)
    if position is None or position + 1 >= len(turns) or turns[position + 1]["role"] != "assistant":
        return None
    reference = turns[position + 1]["content"]
    if reference.startswith("<tool_call>"):
        return None
    results = []
    for block in re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", turns[position]["content"], re.DOTALL):
        try:
            results.append(json.loads(block)["content"])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    required = []
    for value in _leaves(results):
        if _mentions(reference, value) and value not in required:
            required.append(value)
    return {"group": "multi_turn", "tools": example["tools"], "turns": turns[:position + 1], "reference": reference,
            "draft": {"required_candidates": required, "forbidden": []}}


def rewrite_draft(example: dict) -> dict | None:
    """Missing-info from xLAM: a one-call request whose query states a required argument. The labelling
    pass rewrites the query without that value; removable lists the required arguments the query states."""
    item = call_item(example) if example["group"] in ("single", "select") else None
    if item is None or len(item["expected_calls"]) != 1:
        return None
    call = item["expected_calls"][0]
    schema = next((tool.get("parameters") or {} for tool in item["tools"] if tool["name"] == call["name"]), {})
    removable = [{"param": param, "value": call["arguments"][param]} for param in schema.get("required", [])
                 if param in call["arguments"] and not isinstance(call["arguments"][param], (dict, list, bool))
                 and _mentions(item["turns"][0]["content"], call["arguments"][param])]
    if not removable:
        return None
    return {"group": "missing_info", "tools": item["tools"], "turns": item["turns"],
            "draft": {"removable": removable, "original_call": call}}


def apply_rewrite(draft: dict, label: dict) -> dict:
    """A rewrite_draft item with its labelled query: the request now lacks missing_param, so the right
    answer is no call and a question naming it (scored by score_clarify with the synonyms)."""
    removable = {entry["param"]: entry["value"] for entry in draft["draft"]["removable"]}
    param, query = label["missing_param"], label["query"]
    if param not in removable:
        raise ValueError(f"{draft['id']}: {param} is not a removable argument")
    if _mentions(query, removable[param]):
        raise ValueError(f"{draft['id']}: the query still states {param}={removable[param]!r}")
    return {"id": draft["id"], "set": draft["set"], "group": "missing_info", "tools": draft["tools"],
            "turns": [{"role": "user", "content": query}], "expected_calls": [], "missing_param": param,
            "synonyms": list(label.get("synonyms", [])),
            "meta": {**draft["meta"], "rewritten_from": draft["turns"][0]["content"],
                     "original_call": draft["draft"]["original_call"], "label_source": "xlam_rewrite"}}


# ---- T-unseen filter (PLAN 4.2 item 6) ---------------------------------------------------

def near_seen_tools(tools: list[dict], vectors: np.ndarray, *, threshold: float, block=2048) -> dict:
    """Unseen tool text -> (cosine, nearest seen tool name) for every unseen tool at or above threshold."""
    unseen = np.array([tool["unseen"] for tool in tools])
    seen_rows, near = np.flatnonzero(~unseen), {}
    if not len(seen_rows):
        return near
    unseen_rows = np.flatnonzero(unseen)
    for start in range(0, len(unseen_rows), block):
        rows = unseen_rows[start:start + block]
        similarity = vectors[rows] @ vectors[seen_rows].T
        best = similarity.argmax(axis=1)
        for offset, (row, column) in enumerate(zip(rows, best)):
            score = float(similarity[offset, column])
            if score >= threshold:
                near[tools[row]["text"]] = (score, tools[seen_rows[column]]["name"])
    return near


# ---- T-rename, nonsense level (PLAN 4.4) -------------------------------------------------

IDENTIFIER = re.compile(r"[_\d]|[a-z][A-Z]")  # Only names like get_weather or getWeather are replaced in prose.


def _fresh_name(rng, used: set) -> str:
    while True:
        name = "fn_" + "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(3))
        if name not in used:
            used.add(name)
            return name


def synonym_of(synonyms: dict, tool: str, param: str) -> str | None:
    """A parameter's synonym: the per-tool entry when the shared one does not fit that tool."""
    return synonyms["params"].get(f"{tool}.{param}", synonyms["params"].get(param))


def rename_item(item: dict, *, seed: int, synonyms: dict | None = None) -> tuple[dict, list]:
    """Rename every tool and its top-level parameters consistently in schema, prompt and calls; description
    text is kept. Without synonyms: fn_xxx and p1, p2… (nonsense level). With synonyms ({"tools": {name: new},
    "params": {param or tool.param: new}}, see synonym_table): the labelled names; a name without one stays.
    Returns the item and the argument keys the schema did not declare."""
    rng = random.Random(f"{seed}:rename:{item['id']}")
    used, tool_map, param_maps, tools = set(), {}, {}, []
    for tool in item["tools"]:
        new_name = _fresh_name(rng, used) if synonyms is None else synonyms["tools"].get(tool["name"], tool["name"])
        tool_map[tool["name"]] = new_name
        parameters = dict(tool.get("parameters") or {})
        properties = parameters.get("properties") or {}
        if synonyms is None:
            param_map = {param: f"p{index}" for index, param in enumerate(properties, start=1)}
        else:
            param_map = {param: synonym_of(synonyms, tool["name"], param) or param for param in properties}
        param_maps[tool["name"]] = param_map
        if properties:
            parameters["properties"] = {param_map[param]: spec for param, spec in properties.items()}
        if "required" in parameters:
            parameters["required"] = [param_map.get(param, param) for param in parameters["required"]]
        tools.append({**tool, "name": new_name, "parameters": parameters})
    undeclared, calls = [], []
    for call in item["expected_calls"]:
        param_map = param_maps.get(call["name"], {})
        undeclared += [f"{call['name']}.{key}" for key in call["arguments"] if key not in param_map]
        calls.append({"name": tool_map.get(call["name"], call["name"]),
                      "arguments": {param_map.get(key, key): value for key, value in call["arguments"].items()}})
    # Prose does not say which tool a parameter name belongs to: a name renamed differently across tools
    # follows the called tools, and stays unchanged when those disagree too.
    called = {call["name"] for call in item["expected_calls"]}
    renames, ambiguous = dict(tool_map), []
    for param in {param for param_map in param_maps.values() for param in param_map}:
        options = {param_map[param] for name, param_map in param_maps.items() if param in param_map}
        if len(options) > 1:
            options = {param_maps[name][param] for name in called if param in param_maps.get(name, {})}
        if len(options) == 1:
            renames.setdefault(param, options.pop())
        elif IDENTIFIER.search(param):
            ambiguous.append(param)
    olds = sorted((name for name in renames if IDENTIFIER.search(name)), key=len, reverse=True)
    # One pass, so a new name that equals an old one is never renamed again.
    pattern = re.compile(r"(?<![\w])(" + "|".join(map(re.escape, olds)) + r")(?![\w])") if olds else None
    turns = [{**turn, "content": pattern.sub(lambda match: renames[match.group(1)], turn["content"])
              if pattern else turn["content"]} for turn in item["turns"]]
    meta = {**item["meta"], "renamed_from": item["id"], "tool_map": tool_map}
    if ambiguous:
        meta["ambiguous_params"] = sorted(ambiguous)
    suffix = ":rename" if synonyms is None else ":synonym"
    return {**item, "id": item["id"] + suffix, "tools": tools, "turns": turns, "expected_calls": calls,
            "meta": meta}, undeclared


def synonym_table(rows: list) -> dict:
    """rename_synonyms labels: {"id": "tool:<name>" | "param:<name>" | "param:<tool>.<name>", "synonym": new}."""
    table = {"tools": {}, "params": {}}
    for row in rows:
        kind, name = row["id"].split(":", 1)
        table["tools" if kind == "tool" else "params"][name] = row["synonym"]
    return table


def _words(name):
    return set(re.findall(r"[a-z]+|\d+", re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower()))


def synonym_problems(items: list, synonyms: dict) -> Counter:
    """Every way the synonym table fails the items: a name without a synonym, a synonym equal to or no more
    than a respelling of the name, and two tools of one item (or two parameters of one tool) sharing one."""
    problems = Counter()
    for item in items:
        new_tools = {}  # Keyed by the old name: a source item may already offer two tools with one name.
        for tool in item["tools"]:
            new_tool = synonyms["tools"].get(tool["name"])
            if new_tool is None:
                problems[f"tool without synonym: {tool['name']}"] += 1
            elif _words(new_tool) == _words(tool["name"]):
                problems[f"tool unchanged: {tool['name']}"] += 1
            new_tools[tool["name"]] = new_tool or tool["name"]
            new_params = []
            for param in ((tool.get("parameters") or {}).get("properties") or {}):
                new_param = synonym_of(synonyms, tool["name"], param)
                if new_param is None:
                    problems[f"parameter without synonym: {param}"] += 1
                elif _words(new_param) == _words(param):
                    problems[f"parameter unchanged: {param}"] += 1
                new_params.append(new_param or param)
            if len(set(new_params)) < len(new_params):
                problems[f"parameters share a synonym in {tool['name']}"] += 1
        if len(set(new_tools.values())) < len(new_tools):
            problems[f"tools share a synonym in {item['id']}"] += 1
    return problems


# ---- Sampling ----------------------------------------------------------------------------

class GrowingIndex(NearDuplicateIndex):
    """Near-duplicate index that items join once picked, so no two eval items repeat a prompt."""

    def __init__(self, **minhash):
        super().__init__([], **minhash)
        self.signatures, self.buckets = np.zeros((0, self.hasher.a.shape[0]), dtype=np.uint64), {}

    def add(self, text: str):
        self.exact.add(normalize_text(text))
        signature = self.hasher.signature(shingles(text, self.ngram))
        position = len(self.signatures)
        self.signatures = np.vstack([self.signatures, signature])
        for key in self._band_keys(signature):
            self.buckets[key] = np.append(self.buckets.get(key, np.array([], dtype=int)), position)


def user_text(item: dict) -> str:
    return "\n".join(turn["content"] for turn in item["turns"] if turn["role"] == "user")


def pick(candidates: list, n: int, index: GrowingIndex, *, rng, text=user_text) -> tuple[list, Counter]:
    """n items in random order, skipping prompts that repeat one already picked in any set."""
    order = list(candidates)
    rng.shuffle(order)
    picked, skipped = [], Counter()
    for item in order:
        if len(picked) == n:
            break
        prompt = text(item)
        verdict = index.match(prompt)
        if verdict:
            skipped[verdict] += 1
            continue
        index.add(prompt)
        picked.append(item)
    return picked, skipped


def called_clusters(item: dict, cluster_of: dict) -> set:
    """Clusters of the tools the answer calls: the domain a sample tests (offered distractors aside)."""
    tools = {tool["name"]: tool for tool in item["tools"]}
    return {cluster_of[tool_text(tools[call["name"]])] for call in item["expected_calls"]
            if call["name"] in tools and tool_text(tools[call["name"]]) in cluster_of}


def cap_clusters(splits: dict, candidates: list, sizes: dict, cap: float, index: GrowingIndex, *, rng,
                 cluster_of: dict) -> tuple[dict, dict]:
    """Per split, at most floor(cap x size) samples (at least 1) per called cluster. Picked items stay in order
    while they fit; the rest is topped up from candidates in random order, skipping repeats like pick."""
    order = list(candidates)
    rng.shuffle(order)
    taken = {item["id"] for items in splits.values() for item in items}
    order = [item for item in order if item["id"] not in taken]
    result, stats = {}, {"kept": 0, "dropped": Counter(), "added": 0, "skipped": Counter()}
    for split_name, size in sizes.items():
        limit, counts, kept = max(1, int(cap * size)), Counter(), []

        def fits(item):
            return all(counts[cluster] < limit for cluster in called_clusters(item, cluster_of))

        for item in splits[split_name]:
            if len(kept) < size and fits(item):
                counts.update(called_clusters(item, cluster_of))
                kept.append(item)
            else:
                stats["dropped"].update(called_clusters(item, cluster_of))
        stats["kept"] += len(kept)
        for item in order:
            if len(kept) == size:
                break
            if not fits(item):
                continue
            verdict = index.match(user_text(item))
            if verdict:
                stats["skipped"][verdict] += 1
                continue
            index.add(user_text(item))
            counts.update(called_clusters(item, cluster_of))
            kept.append({**item, "split": split_name})
            stats["added"] += 1
        result[split_name] = kept
    return result, stats


def split_dev_test(items: list, n_test: int) -> dict:
    return {"test": items[:n_test], "dev": items[n_test:]}


def with_id(example: dict, item: dict, set_name: str) -> dict:
    meta = {"source_id": example["id"], "source": example["source"], "clusters": example["meta"].get("clusters", [])}
    return {"id": f"{set_name}:{example['id']}", "set": set_name, **item, "meta": meta}


# ---- Labelled groups ---------------------------------------------------------------------

LABEL_FILES = ("missing_info_glaive", "missing_info_rewrite", "multi_turn", "no_tool_prompts", "no_tool",
               "rename_synonyms")
LABELLED_GROUPS = ("missing_info", "multi_turn", "no_tool")


def glaive_missing_info(draft: dict, label: dict) -> dict:
    """A Glaive Missing-info draft with its labelled missing arguments; asking for any of them is right."""
    params = label["missing_params"]
    return {"id": draft["id"], "set": draft["set"], "group": "missing_info", "tools": draft["tools"],
            "turns": draft["turns"], "expected_calls": [], "missing_param": params[0],
            "synonyms": list(label["synonyms"]), "meta": {**draft["meta"], "label_source": "glaive"}}


def multi_turn_item(draft: dict, label: dict) -> dict:
    return {"id": draft["id"], "set": draft["set"], "group": "multi_turn", "tools": draft["tools"],
            "turns": draft["turns"], "expected_calls": [], "reference": draft["reference"],
            "required": label["required"], "forbidden": label["forbidden"],
            "meta": {**draft["meta"], "derivable": bool(label.get("derivable"))}}


def mix(first: dict, second: dict, n_first: dict, *, n_test: int, want: int) -> dict:
    """Per side, n_first[side] items of `first` and the rest from `second`; test and dev keep that ratio."""
    parts = {}
    for side in first:
        test_first = round(n_test * n_first[side] / want)
        a, b = first[side][:n_first[side]], second[side][:want - n_first[side]]
        parts[side] = {"test": a[:test_first] + b[:n_test - test_first], "dev": a[test_first:] + b[n_test - test_first:]}
    return parts


def assemble_labelled(drafts: dict, labels: dict, *, n_test: int, n_dev: int,
                      index: GrowingIndex) -> tuple[dict, dict]:
    """Missing-info and Multi-turn per side from labelled drafts, in draft (random) order.
    Missing-info takes the same number of Glaive items on both sides and fills up with xLAM rewrites,
    so seen and unseen share one source mix. Multi-turn likewise aims at one mix: as many items that need the
    tool result as the poorer side has, the rest labelled derivable (the answer can be worked out without the
    tool result; meta.derivable marks them for a sub-score). A side short of derivable items fills up with
    items that need the tool result. Test and dev keep both mixes. Rewritten queries
    are new prompts, so they are checked against `index` (the prompts already in the eval) and each other."""
    want, sets, stats = n_test + n_dev, defaultdict(lambda: defaultdict(list)), defaultdict(dict)
    by_id = {name: {label["id"]: label for label in labels.get(name, [])} for name in LABEL_FILES}
    sides = ("t_seen", "t_unseen")
    glaive, multi, derivable = {}, {}, {}
    for side in sides:
        glaive[side] = [glaive_missing_info(draft, by_id["missing_info_glaive"][draft["id"]])
                        for draft in drafts.get(f"{side}_missing_info", [])
                        if "missing_params" in by_id["missing_info_glaive"].get(draft["id"], {})]
        labelled = [(draft, by_id["multi_turn"][draft["id"]]) for draft in drafts.get(f"{side}_multi_turn", [])
                    if "required" in by_id["multi_turn"].get(draft["id"], {})]
        derivable[side] = [multi_turn_item(draft, label) for draft, label in labelled if label.get("derivable")]
        multi[side] = [multi_turn_item(draft, label) for draft, label in labelled if not label.get("derivable")]
    target = want - min(want, *(len(items) for items in multi.values()))
    n_derivable = {side: min(target, len(derivable[side])) for side in sides}
    multi_parts = mix(derivable, multi, n_derivable, n_test=n_test, want=want)
    n_glaive = min(want, *(len(items) for items in glaive.values()))
    for side in sides:
        for item in glaive[side][:n_glaive] + multi_parts[side]["test"] + multi_parts[side]["dev"]:
            index.add(user_text(item))
    rewrites, repeated = {}, Counter()
    for side in sides:
        rewrites[side] = []
        for draft in drafts.get(f"{side}_missing_info_rewrite", []):
            if draft["id"] not in by_id["missing_info_rewrite"] or len(rewrites[side]) == want - n_glaive:
                continue
            item = apply_rewrite(draft, by_id["missing_info_rewrite"][draft["id"]])
            verdict = index.match(user_text(item))
            if verdict:
                repeated[side] += 1
                continue
            index.add(user_text(item))
            rewrites[side].append(item)
    info_parts = mix(glaive, rewrites, dict.fromkeys(sides, n_glaive), n_test=n_test, want=want)
    for side in sides:
        for parts in (info_parts[side], multi_parts[side]):
            for split_name, items in parts.items():
                sets[split_name][side] += [{**item, "split": split_name} for item in items]
        labelled_rewrites = sum(draft["id"] in by_id["missing_info_rewrite"]
                                for draft in drafts.get(f"{side}_missing_info_rewrite", []))
        stats[side]["missing_info"] = {"picked": n_glaive + len(rewrites[side]), "wanted": want, "glaive": n_glaive,
                                       "rewrite": len(rewrites[side]), "rewrite_repeated": repeated[side],
                                       "labelled": len(glaive[side]) + labelled_rewrites}
        stats[side]["multi_turn"] = {"picked": sum(len(items) for items in multi_parts[side].values()), "wanted": want,
                                     "derivable": n_derivable[side], "labelled": len(multi[side]) + len(derivable[side])}
    draft_ids = {draft["id"] for items in drafts.values() for draft in items}
    # no_tool_prompts is keyed by chat example, not by draft.
    stats["unmatched_labels"] = {name: sum(label["id"] not in draft_ids for label in labels.get(name, []))
                                 for name in LABEL_FILES if name not in ("no_tool_prompts", "rename_synonyms")}
    return sets, stats


# ---- No-tool (PLAN 4.3) ------------------------------------------------------------------

NO_TOOL_KINDS = ("plain", "related", "no_fitting_tool")
NO_FITTING_TOOL_REPLY = "None of the tools I have can help with this request, so I can't look it up or do it for you."


def side_tools(examples: list, tools: list, vectors: np.ndarray, near: dict, *, unseen: bool) -> dict:
    """The tools a side may offer: those its eval pool offers, with schema, cluster and embedding."""
    row = {tool["text"]: index for index, tool in enumerate(tools)}
    found = {}
    for example in examples:
        for tool in example["tools"]:
            text = tool_text(tool)
            if text in row and tools[row[text]]["unseen"] == unseen and text not in near:
                found.setdefault(text, tool)
    # One tool per name: an offer listing two tools with one name would be ambiguous.
    by_name = {}
    for text in sorted(found):
        by_name.setdefault(found[text]["name"], text)
    texts = sorted(by_name.values())
    return {"tools": [found[text] for text in texts], "clusters": np.array([tools[row[text]]["cluster"] for text in texts]),
            "vectors": vectors[[row[text] for text in texts]]}


def centroids(offer: dict) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(offer["clusters"])
    sums = np.stack([offer["vectors"][offer["clusters"] == cluster].sum(axis=0) for cluster in ids])
    return ids, sums / np.linalg.norm(sums, axis=1, keepdims=True)


def offer_tools(offer: dict, members: np.ndarray, rng, *, first=None) -> list:
    """1–4 tools drawn from `members` (row indices into the side's tools), `first` included, in random order."""
    k = min(len(members), rng.randint(1, 4))
    chosen = list(rng.sample([int(m) for m in members if m != first], k - (first is not None)))
    if first is not None:
        chosen.append(int(first))
    rng.shuffle(chosen)
    return [offer["tools"][m] for m in chosen]


def no_tool_drafts(side: str, offer: dict, prompts: list, prompt_vectors: np.ndarray, requests: list, catalog: dict,
                   settings: dict, index: GrowingIndex, rng, *, want: int | None = None,
                   offer_rng=None) -> tuple[list, dict]:
    """Drafts of the three No-tool kinds for one side (PLAN 4.3), for `want` items (default: test + dev):
    plain — a general-knowledge prompt with tools from the clusters farthest from it;
    related — a prompt close to one cluster with tools from that cluster, so a tool looks relevant but is not needed;
    no_fitting_tool — an xLAM request that needs a tool, offered only tools far from the tools its example offered
    (looked up in `catalog`, text -> (cluster, vector), since they may belong to the other side).
    Every draft goes to the labelling pass, which checks that no offered tool helps. offer_rng(draft id), when
    given, draws each offer on its own, so a change elsewhere in the pick does not reshuffle every offer."""
    def offer_with(draft_id):
        return offer_rng(draft_id) if offer_rng else rng

    ids, centers = centroids(offer)
    if want is None:
        want = settings["test_per_group"] + round(settings["test_per_group"] * settings["dev_fraction"])
    n_hard = round(want * settings["no_tool"]["hard_fraction"] / 2)
    margin = settings["no_tool"]["draft_margin"]
    similarity = prompt_vectors @ centers.T                       # prompt x cluster
    nearest = similarity.argmax(axis=1)
    closeness = similarity.max(axis=1)
    cut = np.quantile(closeness, 1 - settings["no_tool"]["related_quantile"])
    drafts, stats = [], {}

    related = [(prompt, row) for row, prompt in enumerate(prompts) if closeness[row] >= cut]
    picked, skipped = pick(related, round(n_hard * margin), index, rng=rng, text=lambda pair: pair[0]["turns"][0]["content"])
    for prompt, row in picked:
        members = np.flatnonzero(offer["clusters"] == ids[nearest[row]])
        best = members[np.argmax(offer["vectors"][members] @ prompt_vectors[row])]
        drafts.append(with_id(prompt, {"group": "no_tool", "kind": "related", "tools": offer_tools(offer, members, offer_with(f"{side}:{prompt['id']}"), first=best),
                                       "turns": prompt["turns"][:1], "expected_calls": [],
                                       "reference": prompt["turns"][1]["content"],
                                       "draft": {"cluster": int(ids[nearest[row]]), "similarity": round(float(closeness[row]), 3)}},
                              side))
    stats["related"] = {"picked": len(picked), "wanted": round(n_hard * margin), "skipped": dict(skipped)}

    plain = [(prompt, row) for row, prompt in enumerate(prompts)]
    n_plain = round((want - 2 * n_hard) * margin)
    picked, skipped = pick(plain, n_plain, index, rng=rng, text=lambda pair: pair[0]["turns"][0]["content"])
    for prompt, row in picked:
        far = [cluster for cluster in ids[np.argsort(similarity[row])] if cluster != ids[nearest[row]]][:3]
        members = np.flatnonzero(np.isin(offer["clusters"], far))
        if not len(members):
            skipped["no_far_tool"] += 1
            continue
        drafts.append(with_id(prompt, {"group": "no_tool", "kind": "plain", "tools": offer_tools(offer, members, offer_with(f"{side}:{prompt['id']}")),
                                       "turns": prompt["turns"][:1], "expected_calls": [],
                                       "reference": prompt["turns"][1]["content"], "draft": {"clusters": [int(cluster) for cluster in far]}},
                              side))
    stats["plain"] = {"picked": len(picked), "wanted": n_plain, "skipped": dict(skipped)}

    n_requests = round(n_hard * settings["no_tool"]["no_fitting_margin"])
    picked, skipped = pick(requests, n_requests, index, rng=rng)
    for request in picked:
        called = request["expected_calls"][0]["name"]
        own = [catalog[tool_text(tool)] for tool in request["tools"] if tool_text(tool) in catalog]
        if not own:
            skipped["unknown_tool"] += 1
            continue
        cosine = offer["vectors"] @ np.mean([vector for _, vector in own], axis=0)
        # Far from every tool the request offered, and outside their clusters.
        mask = (cosine < np.quantile(cosine, settings["no_tool"]["far_quantile"])) & \
            ~np.isin(offer["clusters"], [cluster for cluster, _ in own])
        if not mask.any():
            skipped["no_far_tool"] += 1
            continue
        drafts.append({**request, "group": "no_tool", "kind": "no_fitting_tool",
                       "tools": offer_tools(offer, np.flatnonzero(mask), offer_with(request["id"])), "expected_calls": [],
                       "reference": NO_FITTING_TOOL_REPLY, "draft": {"needed_tool": called}})
    stats["no_fitting_tool"] = {"picked": len(picked), "wanted": n_requests, "skipped": dict(skipped)}
    return drafts, stats


def no_tool_item(draft: dict) -> dict:
    return {key: draft[key] for key in ("id", "set", "group", "kind", "tools", "turns", "expected_calls", "reference", "meta")}


def assemble_no_tool(drafts: dict, labels: list, *, n_test: int, n_dev: int, hard_fraction: float) -> tuple[dict, dict]:
    """No-tool per side from drafts the labelling pass kept (ok), in draft order. Both sides take the same
    number of each kind (the hard kinds split hard_fraction evenly) and test and dev keep that mix."""
    want, sets, stats = n_test + n_dev, defaultdict(lambda: defaultdict(list)), {}
    ok = {label["id"] for label in labels if label.get("ok")}
    n_hard = round(want * hard_fraction / 2)
    counts = {"related": n_hard, "no_fitting_tool": n_hard, "plain": want - 2 * n_hard}
    tests = {"related": round(n_test * n_hard / want), "no_fitting_tool": round(n_test * n_hard / want)}
    tests["plain"] = n_test - 2 * tests["related"]
    for side in ("t_seen", "t_unseen"):
        kept = [no_tool_item(draft) for draft in drafts.get(f"{side}_no_tool", []) if draft["id"] in ok]
        stats[side] = {"no_tool": {"picked": 0, "wanted": want, "labelled": len(kept)}}
        for kind in NO_TOOL_KINDS:
            items = [item for item in kept if item["kind"] == kind][:counts[kind]]
            for split_name, part in split_dev_test(items, tests[kind]).items():
                sets[split_name][side] += [{**item, "split": split_name} for item in part]
            stats[side]["no_tool"][kind] = len(items)
            stats[side]["no_tool"]["picked"] += len(items)
    return sets, stats


# ---- Pipeline ----------------------------------------------------------------------------

def run(pools: dict, tools: list, config: dict, *, labels=None, embed=embed_texts) -> dict:
    settings, seed = config["eval_sets"], config["seed"]
    n_test, n_dev = settings["test_per_group"], round(settings["test_per_group"] * settings["dev_fraction"])
    rng = random.Random(seed)
    index = GrowingIndex(seed=seed, **config["split"]["minhash"])

    print(f"[eval_sets] embedding {len(tools)} tools for the T-unseen filter…", flush=True)
    vectors = embed([tool["text"] for tool in tools], config["split"]["embedder"])
    near = near_seen_tools(tools, vectors, threshold=settings["near_seen_threshold"])
    dropped, kept_unseen = Counter(), []
    for example in pools["unseen_eval"]:
        if any(tool_text(tool) in near for tool in example["tools"]):
            dropped[example["group"]] += 1
        else:
            kept_unseen.append(example)

    sets, drafts, stats = defaultdict(lambda: defaultdict(list)), defaultdict(list), defaultdict(dict)
    available = defaultdict(Counter)
    for set_name, examples in (("t_seen", pools["seen_eval"]), ("t_unseen", kept_unseen)):
        by_group = defaultdict(list)
        for example in examples:
            if example["group"] in CALL_GROUPS:
                item = call_item(example)
            elif example["group"] == "clarify_candidate":
                item = clarify_draft(example)
            elif example["group"] == "multi_turn":
                item = multi_turn_draft(example)
            else:
                continue
            if item is not None:
                by_group[item["group"]].append(with_id(example, item, set_name))
        for group in sorted(by_group):
            available[set_name][group] = len(by_group[group])
            # Drafts get extra room: the labelling pass will drop some.
            want = n_test + n_dev if group in CALL_GROUPS else round((n_test + n_dev) * settings["draft_margin"])
            picked, skipped = pick(by_group[group], want, index, rng=rng)
            stats[set_name][group] = {"picked": len(picked), "wanted": want, "skipped": dict(skipped)}
            if group in CALL_GROUPS:
                for split_name, items in split_dev_test(picked, n_test).items():
                    sets[split_name][set_name] += [{**item, "split": split_name} for item in items]
            else:
                drafts[f"{set_name}_{group}"] = picked

    undeclared = Counter()
    for split_name in SPLITS:
        for item in sets[split_name]["t_seen"]:
            renamed, missing = rename_item(item, seed=seed)
            undeclared.update(missing)
            sets[split_name]["t_rename_nonsense"].append({**renamed, "set": "t_rename_nonsense"})
    # Synonym level: labelled names, no randomness and no sampling, so no other set depends on it.
    synonyms = synonym_table((labels or {}).get("rename_synonyms", []))
    renamable = [item for split_name in SPLITS for item in sets[split_name]["t_seen"]]
    synonym_issues = synonym_problems(renamable, synonyms) if synonyms["tools"] else Counter()
    if synonyms["tools"]:
        for split_name in SPLITS:
            for item in sets[split_name]["t_seen"]:
                renamed, _ = rename_item(item, seed=seed, synonyms=synonyms)
                sets[split_name]["t_rename_synonym"].append({**renamed, "set": "t_rename_synonym"})
    for group in CALL_GROUPS:
        stats["t_rename_synonym"][group] = {"wanted": n_test + n_dev, "skipped": {}, "picked": sum(
            item["group"] == group for split_name in SPLITS for item in sets[split_name]["t_rename_synonym"])}

    n_chat_test = settings["chat_test"]
    chat = [with_id(example, chat_item(example), "c") for example in pools["chat_eval"]]
    want = n_chat_test + round(n_chat_test * settings["dev_fraction"])
    picked, skipped = pick(chat, want, index, rng=rng)
    stats["c"]["chat"] = {"picked": len(picked), "wanted": want, "skipped": dict(skipped)}
    available["c"]["chat"] = len(chat)
    for split_name, items in split_dev_test(picked, n_chat_test).items():
        sets[split_name]["c"] = [{**item, "split": split_name} for item in items]

    # Glaive repeats its Missing-info prompts, so xLAM requests rewritten without one required value top
    # the group up. Drawn last, from examples no other set uses, so the sets above stay unchanged.
    for set_name, examples in (("t_seen", pools["seen_eval"]), ("t_unseen", kept_unseen)):
        candidates = [with_id(example, item, set_name) for example in examples
                      for item in [rewrite_draft(example)] if item is not None]
        picked, skipped = pick(candidates, settings["rewrite_drafts"][set_name], index, rng=rng)
        stats[set_name]["missing_info_rewrite"] = {"picked": len(picked), "wanted": settings["rewrite_drafts"][set_name],
                                                   "skipped": dict(skipped)}
        drafts[f"{set_name}_missing_info_rewrite"] = [{**item, "id": item["id"] + ":rewrite"} for item in picked]

    # Unseen Multi-turn is mostly arithmetic the model can redo without the tool result; more drafts give the
    # labelling pass room to find the rest. Drawn last and appended, so earlier drafts and sets stay unchanged.
    for set_name, examples in (("t_seen", pools["seen_eval"]), ("t_unseen", kept_unseen)):
        extra = settings["multi_turn_extra_drafts"].get(set_name, 0)
        if extra:
            candidates = [with_id(example, item, set_name) for example in examples if example["group"] == "multi_turn"
                          for item in [multi_turn_draft(example)] if item is not None]
            picked, skipped = pick(candidates, extra, index, rng=rng)
            stats[set_name]["multi_turn_extra"] = {"picked": len(picked), "wanted": extra, "skipped": dict(skipped)}
            drafts[f"{set_name}_multi_turn"] += picked

    # No-tool drafts come last too. Prompts are held-out chat the labelling pass kept (no_tool_prompts).
    # Drafts keep only the first turn, so repeats are checked on it, against C's first turns too.
    keep = {label["id"] for label in (labels or {}).get("no_tool_prompts", []) if label.get("keep")}
    in_c = {normalize_text(item["turns"][0]["content"]) for split_name in SPLITS for item in sets[split_name]["c"]}
    prompts = [example for example in pools["chat_eval"] if example["id"] in keep and len(example["turns"]) > 1
               and example["turns"][0]["role"] == "user" and example["turns"][1]["role"] == "assistant"
               and normalize_text(example["turns"][0]["content"]) not in in_c]
    if prompts:
        prompt_vectors = embed([example["turns"][0]["content"] for example in prompts], config["split"]["embedder"])
        # A side is set by the tools it offers. Both sides take their requests from the seen pool: the unseen
        # pool asks mostly for arithmetic a model can do without any tool, so it has few requests that need one.
        singles = [(example, item) for example in pools["seen_eval"] if example["group"] == "single"
                   for item in [call_item(example)] if item is not None and len(item["expected_calls"]) == 1]
        catalog = {tool["text"]: (tool["cluster"], vectors[row]) for row, tool in enumerate(tools)}
        for set_name, examples, unseen in (("t_seen", pools["seen_eval"], False), ("t_unseen", kept_unseen, True)):
            requests = [with_id(example, item, set_name) for example, item in singles]
            offer = side_tools(examples, tools, vectors, near, unseen=unseen)
            drafts[f"{set_name}_no_tool"], stats[set_name]["no_tool_drafts"] = no_tool_drafts(
                set_name, offer, prompts, prompt_vectors, requests, catalog, settings, index, rng)

    # T-unseen's pool is mostly one cluster of arithmetic; a cap per called cluster keeps the call groups from
    # testing one domain. Drawn last, so the drafts and every other set stay unchanged.
    cap = settings.get("unseen_cluster_cap")
    cluster_of = {tool["text"]: tool["cluster"] for tool in tools}
    if cap:
        unseen_items = [with_id(example, item, "t_unseen") for example in kept_unseen
                        if example["group"] in CALL_GROUPS for item in [call_item(example)] if item is not None]
        capped = {}
        for group in sorted(CALL_GROUPS):  # The order the groups were picked in.
            picked = {split_name: [item for item in sets[split_name]["t_unseen"] if item["group"] == group]
                      for split_name in SPLITS}
            capped[group], cap_stats = cap_clusters(picked, [item for item in unseen_items if item["group"] == group],
                                                    {"test": n_test, "dev": n_dev}, cap, index, rng=rng,
                                                    cluster_of=cluster_of)
            stat = stats["t_unseen"].setdefault(group, {"picked": 0, "wanted": n_test + n_dev, "skipped": {}})
            stat["picked"] = sum(len(items) for items in capped[group].values())
            stat["capped"] = cap_stats
            stat["available_by_cluster"] = Counter(cluster for item in unseen_items if item["group"] == group
                                                   for cluster in called_clusters(item, cluster_of))
        for split_name in SPLITS:
            sets[split_name]["t_unseen"] = [item for group in sorted(CALL_GROUPS) for item in capped[group][split_name]]

    labelled_stats = {}
    if labels:
        # A fresh index of the prompts the eval already holds: the drafting index also has every draft,
        # which a rewrite of that same draft would match.
        final = GrowingIndex(seed=seed, **config["split"]["minhash"])
        for split_name in SPLITS:
            for set_name in ("t_seen", "t_unseen", "c"):
                for item in sets[split_name][set_name]:
                    final.add(user_text(item))
        labelled, labelled_stats = assemble_labelled(drafts, labels, n_test=n_test, n_dev=n_dev, index=final)
        no_tool, no_tool_stats = assemble_no_tool(drafts, labels.get("no_tool", []), n_test=n_test, n_dev=n_dev,
                                                  hard_fraction=settings["no_tool"]["hard_fraction"])
        for side, groups in no_tool_stats.items():
            labelled_stats[side].update(groups)
        for split_name in SPLITS:
            for set_name, items in no_tool[split_name].items():
                labelled[split_name][set_name] += items
        for split_name in SPLITS:
            for set_name, items in labelled[split_name].items():
                sets[split_name][set_name] += items

    called = Counter(call["name"] for item in sets["test"]["t_unseen"] for call in item["expected_calls"])
    clusters = Counter(cluster for split_name in SPLITS for item in sets[split_name]["t_unseen"]
                       for cluster in item["meta"]["clusters"])
    called_by_group = {(set_name, group): Counter(cluster for split_name in SPLITS
                                                  for item in sets[split_name][set_name] if item["group"] == group
                                                  for cluster in called_clusters(item, cluster_of))
                       for set_name in ("t_seen", "t_unseen") for group in CALL_GROUPS}
    return {"sets": sets, "drafts": drafts, "stats": stats, "available": available, "near": near,
            "called_clusters": called_by_group, "cluster_cap": cap, "synonym_problems": synonym_issues,
            "dropped_unseen": dropped, "undeclared": undeclared, "unseen_top_tools": called.most_common(15),
            "unseen_clusters": clusters, "n_test": n_test, "n_dev": n_dev, "labelled": labelled_stats,
            "short": short_groups(stats, labelled_stats, n_test + n_dev)}


def short_groups(stats: dict, labelled_stats: dict, want: int) -> list:
    """Every group the eval needs, with fewer samples than wanted; an absent group counts as 0. Draft groups
    share names with the labelled ones but want a margin, so labelled groups are read from labelled_stats."""
    required = [(stats, set_name, group) for set_name in ("t_seen", "t_unseen", "t_rename_synonym")
                for group in CALL_GROUPS]
    required += [(stats, "c", "chat")]
    required += [(labelled_stats, set_name, group) for set_name in ("t_seen", "t_unseen") for group in LABELLED_GROUPS]
    short = []
    for source, set_name, group in required:
        stat = source.get(set_name, {}).get(group, {"picked": 0, "wanted": want})
        if stat["picked"] < stat["wanted"]:
            short.append((set_name, group, stat["picked"]))
    return short


def render_report(result: dict, config: dict, hashes: dict) -> str:
    settings = config["eval_sets"]
    lines = ["# Báo cáo bộ eval", "",
             "Sinh tự động bởi `python -m src.eval.build_sets`. Đừng sửa tay; chạy lại để cập nhật.", "",
             f"Seed {config['seed']}. Mỗi nhóm có {result['n_test']} mẫu test và {result['n_dev']} mẫu dev "
             f"(PLAN 3.2); bộ C có {settings['chat_test']} mẫu test. Không có prompt nào lặp lại giữa các bộ: "
             "mẫu trùng exact hoặc gần trùng (MinHash, cùng tham số với bước tách) với mẫu đã chọn thì bị bỏ qua.", "",
             "## Số mẫu", "", "| Bộ | Nhóm | Có sẵn | test | dev | Bỏ vì trùng |", "|---|---|---|---|---|---|"]
    for set_name in ("t_seen", "t_unseen", "t_rename_nonsense", "t_rename_synonym", "c"):
        groups = sorted({item["group"] for split_name in SPLITS for item in result["sets"][split_name][set_name]})
        for group in groups:
            counts = {split_name: sum(item["group"] == group for item in result["sets"][split_name][set_name])
                      for split_name in SPLITS}
            stat = result["stats"].get(set_name, {}).get(group, {})
            skipped = ", ".join(f"{kind}: {count}" for kind, count in stat.get("skipped", {}).items()) or "–"
            available = result["available"].get(set_name, {}).get(group, "–")
            lines.append(f"| {set_name} | {group} | {available} | {counts['test']} | {counts['dev']} | {skipped} |")
    if result["short"]:
        lines += ["", "**Thiếu mẫu:** " + "; ".join(f"{s} / {g}: {n}" for s, g, n in result["short"])]
    near = result["near"]
    lines += ["", "## Lọc T-unseen (PLAN 4.2 ý 6)", "",
              f"{len(near)} tool unseen có cosine ≥ {settings['near_seen_threshold']} tới một tool seen. "
              "Mọi mẫu `unseen_eval` có cung cấp một trong các tool này bị bỏ:", "",
              "| Nhóm | Mẫu bị bỏ |", "|---|---|",
              *[f"| {group} | {count} |" for group, count in sorted(result["dropped_unseen"].items())], "",
              "Cụm của các mẫu T-unseen còn lại (dev + test, một mẫu tính cho mọi cụm của nó): "
              + ", ".join(f"cụm {cluster}: {count}" for cluster, count in sorted(result["unseen_clusters"].items())) + ".",
              "", "Tool được gọi nhiều nhất trong T-unseen test: "
              + ", ".join(f"`{name}` ({count})" for name, count in result["unseen_top_tools"]) + "."]
    if result.get("called_clusters"):
        cap = result.get("cluster_cap")
        lines += ["", "## Cụm của tool được gọi", "",
                  (f"T-unseen đặt trần {cap:.0%} mẫu mỗi cụm cho từng nhóm, tính riêng test và dev: pool unseen phần lớn "
                   "là một cụm tính toán (cụm 29). Mẫu đã chọn được giữ nếu còn trong trần, phần thiếu rút thêm từ pool "
                   "sau mọi bước khác, nên nháp và các bộ khác không đổi. " if cap else "")
                  + "Một mẫu tính cho mọi cụm của các tool nó gọi (tool nhiễu không tính); dev + test.", "",
                  "| Bộ | Nhóm | Ba cụm nhiều nhất (số mẫu) | Cụm lớn nhất / số mẫu |", "|---|---|---|---|"]
        for (set_name, group), counts in sorted(result["called_clusters"].items()):
            total = sum(item["group"] == group for split_name in SPLITS for item in result["sets"][split_name][set_name])
            top = ", ".join(f"cụm {cluster}: {count}" for cluster, count in counts.most_common(3))
            share = counts.most_common(1)[0][1] / total if counts and total else 0
            lines.append(f"| {set_name} | {group} | {top} | {share:.0%} |")
        if cap:
            lines += ["", "| Nhóm | Mẫu cũ giữ lại | Rút thêm | Có sẵn trong pool theo cụm (ba cụm lớn nhất) |",
                      "|---|---|---|---|"]
            for group in sorted(CALL_GROUPS):
                stat = result["stats"]["t_unseen"][group]
                capped = stat.get("capped", {})
                pool = stat.get("available_by_cluster", Counter())
                lines.append(f"| {group} | {capped.get('kept', '–')} | {capped.get('added', '–')} | "
                             + ", ".join(f"cụm {cluster}: {count}" for cluster, count in pool.most_common(3))
                             + f" (pool có {result['available']['t_unseen'].get(group, 0)} mẫu) |")
    undeclared = sum(result["undeclared"].values())
    lines += ["", "## T-rename mức tên vô nghĩa (PLAN 4.4)", "",
              "Tool đổi thành `fn_xxx`, tham số cấp ngoài cùng thành `p1, p2…` theo thứ tự trong schema; description "
              "giữ nguyên. Tên dạng định danh (có `_`, chữ số hoặc camelCase) xuất hiện trong prompt cũng được thay.",
              "", f"Tham số trong đáp án mà schema không khai báo, nên giữ tên cũ: {undeclared}"
              + (" (" + ", ".join(f"`{key}`" for key, _ in result["undeclared"].most_common(10)) + ")" if undeclared else "")
              + "."]
    synonym_items = [item for split_name in SPLITS for item in result["sets"][split_name].get("t_rename_synonym", [])]
    if synonym_items:
        renamed_tools = {old: new for item in synonym_items for old, new in item["meta"]["tool_map"].items()}
        lines += ["", "## T-rename mức đồng nghĩa (PLAN 4.4)", "",
                  "Tool và tham số cấp ngoài cùng đổi sang tên cùng nghĩa theo bảng `rename_synonyms` do session gán "
                  "(một tên tham số dùng chung một tên mới, trừ khi tool đó có tên riêng); description giữ nguyên, tên "
                  "dạng định danh trong prompt cũng được thay. Bảng bị từ chối nếu thiếu tên, tên mới chỉ là cách viết "
                  "khác của tên cũ, hoặc hai tool (hai tham số của một tool) trong một mẫu trùng tên mới.", "",
                  f"Tool được đổi tên: {len(renamed_tools)}. Lỗi bảng: "
                  + (", ".join(f"{name} ({count})" for name, count in result["synonym_problems"].most_common(10))
                     or "0") + ".", "",
                  "Ví dụ: " + ", ".join(f"`{old}` → `{new}`" for old, new in sorted(renamed_tools.items())[:8]) + "."]
    labelled = result["labelled"]
    if labelled:
        lines += ["", "## Missing-info và Multi-turn (nhãn do session gán)", "",
                  "Nhãn nằm ở `eval_sets/labels/` (dataset trên Hub). Missing-info lấy cùng số mẫu Glaive ở hai phía, "
                  "phần còn lại là câu hỏi xLAM viết lại cho thiếu một tham số bắt buộc, để seen và unseen có cùng tỉ lệ "
                  "nguồn; test và dev giữ tỉ lệ đó. Multi-turn nhắm cùng một tỉ lệ ở hai phía: lấy số mẫu cần đọc kết quả "
                  "tool bằng mức phía nghèo hơn có, phần còn lại là mẫu tự tính được (đáp án suy ra được từ câu hỏi, "
                  "đánh dấu `meta.derivable` để tính điểm phụ trên phần còn lại); phía thiếu mẫu tự tính được thì bù bằng "
                  "mẫu cần đọc kết quả tool. Đáp án đúng của Missing-info là không gọi tool.", "",
                  "| Bộ | Nhóm | Có nhãn | Dùng | Glaive | Viết lại | Tự tính được |", "|---|---|---|---|---|---|---|"]
        for side in ("t_seen", "t_unseen"):
            for group in ("missing_info", "multi_turn"):
                stat = labelled[side][group]
                lines.append(f"| {side} | {group} | {stat['labelled']} | {stat['picked']} | {stat.get('glaive', '–')} | "
                             f"{stat.get('rewrite', '–')} | {stat.get('derivable', '–')} |")
        if any("no_tool" in labelled.get(side, {}) for side in ("t_seen", "t_unseen")):
            lines += ["", "No-tool (PLAN 4.3): prompt chat giữ riêng mà lượt gán nhãn giữ lại (`no_tool_prompts`), cùng "
                      "câu hỏi xLAM cần tool. `plain` ghép tool ở các cụm xa chủ đề nhất; `related` ghép tool cùng cụm "
                      "với prompt (có vẻ liên quan nhưng không cần dùng); `no_fitting_tool` là câu hỏi cần tool nhưng chỉ "
                      "có tool ở cụm khác. Lượt gán nhãn kiểm từng nháp rằng không tool nào giúp được (`no_tool`). Hai phía "
                      "có cùng số mẫu mỗi loại; test và dev giữ tỉ lệ đó. Đáp án đúng là không gọi tool.", "",
                      "| Bộ | Có nhãn | Dùng | plain | related | no_fitting_tool |", "|---|---|---|---|---|---|"]
            for side in ("t_seen", "t_unseen"):
                stat = labelled[side]["no_tool"]
                lines.append(f"| {side} | {stat['labelled']} | {stat['picked']} | "
                             + " | ".join(str(stat.get(kind, 0)) for kind in NO_TOOL_KINDS) + " |")
        unmatched = {name: count for name, count in labelled["unmatched_labels"].items() if count}
        lines += ["", "Nhãn không khớp nháp nào: "
                  + (", ".join(f"`{name}`: {count}" for name, count in unmatched.items()) or "0") + "."]
    lines += ["", "## Nháp chờ gán nhãn", "",
              "Chưa phải bộ eval: cần một lượt gán nhãn rồi mới chia dev/test (PLAN 3.2).", "",
              "| File | Số mẫu | Nháp |", "|---|---|---|"]
    for name, items in sorted(result["drafts"].items()):
        kind = {"missing_info": "`missing_param_candidates`", "missing_info_rewrite": "`removable` (viết lại câu hỏi)",
                "multi_turn": "`required_candidates`", "no_tool": "`kind` (rà: không tool nào giúp được)"}[name.split("_", 2)[2]]
        lines.append(f"| `drafts/{name}.jsonl` | {len(items)} | {kind} |")
    lines += ["", "## Hash", "", "| File | SHA-256 |", "|---|---|",
              *[f"| `{name}` | `{digest}` |" for name, digest in sorted(hashes.items())], ""]
    return "\n".join(lines)


def load_pools(directory: Path) -> tuple[dict, list]:
    def read(name):
        with (directory / f"{name}.jsonl").open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream]

    return {name: read(name) for name in ("seen_eval", "unseen_eval", "chat_eval")}, read("tools")


def fetch_split(repo_id: str, directory: Path):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, repo_type="dataset", local_dir=str(directory.parent),
                      allow_patterns=[f"{directory.name}/{name}.jsonl"
                                      for name in ("seen_eval", "unseen_eval", "chat_eval", "tools")])


def load_labels(directory: Path) -> dict:
    labels = {}
    for name in LABEL_FILES:
        path = directory / f"{name}.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                labels[name] = [json.loads(line) for line in stream]
    return labels


def fetch_labels(repo_id: str, output_dir: Path):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, repo_type="dataset", local_dir=str(output_dir.parent),
                      allow_patterns=[f"{output_dir.name}/labels/{name}.jsonl" for name in LABEL_FILES])


def push(directory: Path, repo_id: str):
    from huggingface_hub import upload_folder

    upload_folder(folder_path=str(directory), repo_id=repo_id, repo_type="dataset", path_in_repo="eval_sets",
                  allow_patterns=["test/*.jsonl", "dev/*.jsonl", "drafts/*.jsonl"],
                  commit_message="Rule-labelled eval sets and label drafts (src.eval.build_sets)")


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--from-hub", help="Download data/split from this dataset repo when it is missing.")
    parser.add_argument("--push-to", help="Private HF dataset repo; files go under eval_sets/.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    split_dir, output_dir = Path(config["split"]["output_dir"]), Path(config["eval_sets"]["output_dir"])
    if args.from_hub and not (split_dir / "tools.jsonl").exists():
        fetch_split(args.from_hub, split_dir)
    if args.from_hub:
        fetch_labels(args.from_hub, output_dir)
    pools, tools = load_pools(split_dir)
    result = run(pools, tools, config, labels=load_labels(output_dir / "labels"))
    hashes = {}
    for split_name in SPLITS:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)
        for set_name, items in sorted(result["sets"][split_name].items()):
            hashes[f"{split_name}/{set_name}.jsonl"] = write_jsonl(output_dir / split_name / f"{set_name}.jsonl", items)
    (output_dir / "drafts").mkdir(parents=True, exist_ok=True)
    for name, items in sorted(result["drafts"].items()):
        hashes[f"drafts/{name}.jsonl"] = write_jsonl(output_dir / "drafts" / f"{name}.jsonl", items)
    (output_dir / "REPORT.md").write_text(render_report(result, config, hashes), encoding="utf-8")
    for name, digest in hashes.items():
        print(f"[eval_sets] {name}: {digest[:12]}", flush=True)
    if result["synonym_problems"]:
        sys.exit(f"[eval_sets] synonym table problems: {dict(result['synonym_problems'].most_common(20))}; not uploaded")
    if result["short"]:
        sys.exit(f"[eval_sets] groups short of samples: {result['short']} (see {output_dir}/REPORT.md); not uploaded")
    if args.push_to:
        push(output_dir, args.push_to)


if __name__ == "__main__":
    main()
