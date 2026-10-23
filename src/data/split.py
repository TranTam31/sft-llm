"""Cluster tools into functional groups, hold out unseen clusters and dedupe train against eval (PLAN 4.2, 4.5).

Run: python -m src.data.split --config configs/data.yaml [--from-hub USER/REPO] [--push-to USER/REPO]
Reads data/processed/*.jsonl, writes data/split/*.jsonl (Git-ignored) and data/SPLIT.md (tracked).

Pools: an example whose offered tools all sit in seen clusters is seen; all in unseen clusters is
unseen (T-unseen candidates); a mix of both is dropped, since it would leak an unseen tool into
train. Seen tool examples and SmolTalk chat lend a fixed number per group to the eval pools (T-seen,
P, C); the rest is train, minus anything that duplicates an eval example.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import numpy as np

from src.tool_calls import ToolCallParseError, parse_tool_calls


TOOL_GROUPS = ("single", "select", "parallel", "multi_turn", "clarify_candidate", "no_call", "call_no_response")
# Groups that T-unseen draws from directly; No-tool eval is built later from chat prompts (PLAN 4.3).
UNSEEN_EVAL_GROUPS = ("single", "select", "parallel", "multi_turn", "clarify_candidate")
SEEN_EVAL_GROUPS = UNSEEN_EVAL_GROUPS + ("no_call",)  # groups T-seen and P draw from
GENERIC_PREFIXES = {"get", "fetch", "retrieve", "api"}  # PLAN 4.2: get_weather and weather mean the same


# ---- Tools -------------------------------------------------------------------------------

def name_words(name: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name))
    words = [word for word in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if word]
    while len(words) > 1 and words[0] in GENERIC_PREFIXES:
        words = words[1:]
    return words


def tool_text(tool: dict) -> str:
    """What gets embedded: normalized name, description and parameter names."""
    params = list(((tool.get("parameters") or {}).get("properties") or {}))
    text = f"{' '.join(name_words(tool['name']))}. {(tool.get('description') or '').strip()}"
    return text + (f" Parameters: {', '.join(params)}." if params else "")


def n_params(tool: dict) -> int:
    return len((tool.get("parameters") or {}).get("properties") or {})


def called_names(example: dict) -> list[str]:
    names = []
    for turn in example["turns"]:
        if turn["role"] == "assistant" and turn["content"].startswith("<tool_call>"):
            try:
                names += [call["name"] for call in parse_tool_calls(turn["content"])]
            except ToolCallParseError:
                pass
    return names


def embed_texts(texts, embedder: dict, *, batch_size=128):
    """CLS pooling, L2-normalized: the pooling bge-small-en-v1.5 was trained with."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(embedder["id"], revision=embedder["revision"])
    model = AutoModel.from_pretrained(embedder["id"], revision=embedder["revision"]).to(device).eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=256,
                              return_tensors="pt").to(device)
            cls = model(**batch).last_hidden_state[:, 0]
            chunks.append(torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy())
    return np.concatenate(chunks)


def spherical_kmeans(vectors: np.ndarray, k: int, *, seed: int, iterations=100) -> np.ndarray:
    """k-means on unit vectors with cosine similarity; k-means++ seeding. Returns a label per row."""
    rng = np.random.default_rng(seed)
    centers = [vectors[rng.integers(len(vectors))]]
    closest = 1 - vectors @ centers[0]
    for _ in range(1, k):
        weights = np.clip(closest, 0, None) ** 2
        choice = rng.choice(len(vectors), p=weights / weights.sum()) if weights.sum() > 0 else rng.integers(len(vectors))
        centers.append(vectors[choice])
        closest = np.minimum(closest, 1 - vectors @ vectors[choice])
    centers = np.stack(centers)
    labels = None
    for _ in range(iterations):
        new_labels = np.argmax(vectors @ centers.T, axis=1)
        if labels is not None and np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = vectors[labels == cluster]
            if len(members):
                mean = members.sum(axis=0)
                centers[cluster] = mean / np.linalg.norm(mean)
    return labels


def merge_near_identical(vectors: np.ndarray, labels: np.ndarray, *, threshold: float, block=2048):
    """Give tools that are near-identical (cosine >= threshold) one cluster: the majority label of
    their connected component. Glaive repeats a tool with reworded descriptions, and k-means alone
    can put the copies on both sides of the seen/unseen line."""
    parent = list(range(len(vectors)))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for start in range(0, len(vectors), block):
        similarity = vectors[start:start + block] @ vectors.T
        for row, column in zip(*np.nonzero(similarity >= threshold)):
            left, right = find(start + row), find(column)
            if left != right:
                parent[left] = right
    components = defaultdict(list)
    for item in range(len(vectors)):
        components[find(item)].append(item)
    merged = labels.copy()
    for members in components.values():
        if len(members) > 1:
            merged[members] = Counter(labels[members].tolist()).most_common(1)[0][0]
    largest = max(len(members) for members in components.values())
    return merged, {"relabelled": int((merged != labels).sum()), "largest_component": largest}


def compact_name(name: str) -> str:
    """team_image, teamImage and teamimage all become teamimage."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def trigrams(text: str) -> set:
    return {text[i:i + 3] for i in range(max(1, len(text) - 2))}


def name_similarity(left: set, right: set) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def absorb_neighbors(vectors: np.ndarray, unseen: np.ndarray, *, threshold: float, block=2048,
                     names=None, name_threshold=None, alias_similarity=1.0):
    """Move to unseen every seen tool within the cosine threshold of an unseen tool, until none is left.
    Cluster borders cut through families of near-synonymous tools (get_recipe on both sides). An alias,
    a tool whose compact name is this close (character-trigram Jaccard) to an unseen tool's name, needs
    only name_threshold: teamimage vs team_image, check_file_exists vs check_file_existence. The name
    requirement keeps the lower cosine from cascading."""
    unseen = unseen.copy()
    frontier = np.flatnonzero(unseen)
    grams = [trigrams(compact_name(name)) for name in names] if names is not None else None
    rounds = 0
    while len(frontier):
        rounds += 1
        reached = np.zeros(len(vectors), dtype=bool)
        for start in range(0, len(frontier), block):
            rows = frontier[start:start + block]
            similarity = vectors @ vectors[rows].T
            reached |= (similarity >= threshold).any(axis=1)
            if grams is not None:
                for candidate, column in zip(*np.nonzero((similarity >= name_threshold) & ~unseen[:, None])):
                    if not reached[candidate] and name_similarity(grams[candidate],
                                                                  grams[rows[column]]) >= alias_similarity:
                        reached[candidate] = True
        frontier = np.flatnonzero(reached & ~unseen)
        unseen[frontier] = True
    return unseen, rounds


# ---- Choosing unseen clusters ------------------------------------------------------------

def bucket(value: int, edges) -> int:
    """Index of the first edge the value does not exceed; the last bucket is open-ended."""
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


OFFERED_EDGES, PARAM_EDGES = (1, 2, 3, 5), (0, 1, 2, 3)


def difficulty(example: dict) -> tuple:
    """(group, offered-tool bucket, parameter bucket) — the strata PLAN 4.2 matches seen and unseen on.
    Parameters are counted on the example's own schemas: many unrelated APIs share a name like search."""
    offered = {tool["name"]: tool for tool in example["tools"]}
    called = called_names(example) or list(offered)
    params = max((n_params(offered[name]) for name in called if name in offered), default=0)
    return example["group"], bucket(len(example["tools"]), OFFERED_EDGES), bucket(params, PARAM_EDGES)


def strata_distance(seen: Counter, unseen: Counter) -> float:
    """Sum over the three strata of the total-variation distance between two Counters of strata tuples."""
    distance = 0.0
    for position in range(3):
        a, b = Counter(), Counter()
        for strata, count in seen.items():
            a[strata[position]] += count
        for strata, count in unseen.items():
            b[strata[position]] += count
        if not a or not b:
            distance += 1.0
            continue
        total_a, total_b = sum(a.values()), sum(b.values())
        distance += 0.5 * sum(abs(a[key] / total_a - b[key] / total_b) for key in set(a) | set(b))
    return distance


def cluster_adjacency(vectors: np.ndarray, labels: np.ndarray, n_clusters: int, *, threshold: float, block=2048):
    """Count of tool pairs across each pair of clusters with cosine >= threshold: how leaky a border is."""
    adjacency = np.zeros((n_clusters, n_clusters))
    for start in range(0, len(vectors), block):
        rows, columns = np.nonzero(vectors[start:start + block] @ vectors.T >= threshold)
        np.add.at(adjacency, (labels[start + rows], labels[columns]), 1)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def cluster_sums(vectors: np.ndarray, labels: np.ndarray, n_clusters: int):
    sums = np.zeros((n_clusters, vectors.shape[1]))
    np.add.at(sums, labels, vectors)
    return sums, np.bincount(labels, minlength=n_clusters)


def coherence(sums: np.ndarray, sizes: np.ndarray, members) -> float:
    """Mean pairwise cosine between the tools of the given clusters (unit vectors, so from the summed vector)."""
    total, n = sums[list(members)].sum(axis=0), sizes[list(members)].sum()
    return float((total @ total - n) / (n * (n - 1))) if n > 1 else 1.0


def cluster_blocks(sums: np.ndarray, *, threshold: float) -> list:
    """Clusters whose centroids are within the cosine threshold, joined transitively: one domain split by
    k-means (sports fixtures vs standings) must be held out whole, or its other half leaks into train."""
    centroids = sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)
    similarity = centroids @ centroids.T
    parent = list(range(len(sums)))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left, right in zip(*np.nonzero(np.triu(similarity, 1) >= threshold)):
        parent[find(left)] = find(right)
    blocks = defaultdict(list)
    for cluster in range(len(sums)):
        blocks[find(cluster)].append(cluster)
    return sorted(blocks.values())


def check_exclusions(entries: list, top_tool_of: dict, *, key="excluded_clusters") -> dict:
    """Hand-picked clusters (excluded or pinned) are named by id AND their most used tool, so a changed
    clustering fails loudly instead of acting on the wrong cluster. Returns {id: reason}."""
    reasons = {}
    for entry in entries or []:
        actual = top_tool_of.get(entry["id"])
        if actual != entry["top_tool"]:
            raise ValueError(f"cluster {entry['id']} should start with {entry['top_tool']!r} but starts "
                             f"with {actual!r}: the clustering changed, review split.{key}")
        reasons[entry["id"]] = entry.get("reason", "")
    return reasons


class UnseenSearch:
    """Rank candidate unseen-cluster sets on aggregated (cluster set, strata) counts, so a trial is a few matmuls.

    Hard limits (enough unseen examples per eval group, bounded loss of train data) decide feasibility;
    among feasible sets the score adds the strata distance and the border leak, each divided by its
    mean over all trials so neither dominates by scale."""

    def __init__(self, records, n_clusters: int, *, adjacency=None, tools_per_cluster=None):
        counts = Counter((frozenset(clusters), strata) for clusters, strata in records)
        keys = list(counts)
        self.n_clusters = n_clusters
        self.count = np.array([counts[key] for key in keys], dtype=float)
        self.touch = np.zeros((len(keys), n_clusters), dtype=float)
        for row, (clusters, _) in enumerate(keys):
            self.touch[row, list(clusters)] = 1
        self.size = self.touch.sum(axis=1)
        self.features, self.values = [], []
        for position in range(3):
            values = sorted({strata[position] for _, strata in keys}, key=str)
            onehot = np.zeros((len(keys), len(values)))
            for row, (_, strata) in enumerate(keys):
                onehot[row, values.index(strata[position])] = 1
            self.features.append(onehot)
            self.values.append(values)
        self.group_onehot, self.group_names = self.features[0], self.values[0]
        self.adjacency = adjacency if adjacency is not None else np.zeros((n_clusters, n_clusters))
        self.tools_per_cluster = tools_per_cluster if tools_per_cluster is not None else np.ones(n_clusters)

    def pools(self, unseen: np.ndarray):
        hits = self.touch @ unseen
        return hits == 0, (hits == self.size) & (self.size > 0)

    def evaluate(self, unseen: np.ndarray, *, min_per_group: int, max_loss: float, eval_groups) -> dict:
        seen, held = self.pools(unseen)
        distance = 0.0
        for onehot in self.features:
            a, b = (self.count * seen) @ onehot, (self.count * held) @ onehot
            distance += 0.5 * np.abs(a / a.sum() - b / b.sum()).sum() if a.sum() and b.sum() else 1.0
        per_group = dict(zip(self.group_names, (self.count * held) @ self.group_onehot))
        loss = (self.count * ~seen).sum() / self.count.sum()
        leak = float(unseen @ self.adjacency @ (1 - unseen)) / max(1.0, float(unseen @ self.tools_per_cluster))
        short = [group for group in eval_groups if per_group.get(group, 0) < min_per_group]
        return {"distance": float(distance), "leak": leak, "loss": float(loss), "short_groups": short,
                "feasible": not short and loss <= max_loss,
                "unseen_per_group": {group: int(count) for group, count in per_group.items()}}

    def rank(self, n_unseen: int, *, trials: int, seed: int, leak_weight=1.0, blocks=None, pinned=(),
             **limits) -> list:
        """Distinct candidates, feasible ones first by score, then infeasible ones by how far they miss.
        A candidate is the pinned clusters plus whole blocks (default: one block per cluster) up to
        n_unseen clusters, or one more when a block does not fit exactly."""
        rng = np.random.default_rng(seed)
        blocks = blocks if blocks is not None else [[cluster] for cluster in range(self.n_clusters)]
        blocks = [block for block in blocks if not set(block) & set(pinned)]
        seen_sets, candidates = set(), []
        for _ in range(trials):
            chosen = list(pinned)
            for index in rng.permutation(len(blocks)):
                if len(chosen) + len(blocks[index]) <= n_unseen + 1:
                    chosen += blocks[index]
                if len(chosen) >= n_unseen:
                    break
            chosen = tuple(sorted(chosen))
            if len(chosen) < n_unseen or chosen in seen_sets:
                continue
            seen_sets.add(chosen)
            unseen = np.zeros(self.n_clusters)
            unseen[list(chosen)] = 1
            candidates.append((list(chosen), self.evaluate(unseen, **limits)))
        mean_distance = np.mean([result["distance"] for _, result in candidates]) or 1.0
        mean_leak = np.mean([result["leak"] for _, result in candidates]) or 1.0
        for _, result in candidates:
            result["score"] = result["distance"] / mean_distance + leak_weight * result["leak"] / mean_leak
            result["miss"] = len(result["short_groups"]) + max(0.0, result["loss"] - limits["max_loss"])
        return sorted(candidates, key=lambda item: (not item[1]["feasible"], item[1]["miss"], item[1]["score"]))


# ---- Near-duplicates ---------------------------------------------------------------------

MERSENNE = (1 << 61) - 1


def prompt_text(example: dict) -> str:
    return "\n".join(turn["content"] for turn in example["turns"] if turn["role"] == "user")


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def shingles(text: str, n: int) -> set[str]:
    words = normalize_text(text).split()
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


class MinHasher:
    def __init__(self, num_perm: int, *, seed: int):
        rng = np.random.default_rng(seed)
        # a < 2^31 and 32-bit shingle hashes keep a*x + b below 2^64.
        self.a = rng.integers(1, 1 << 31, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, MERSENNE, num_perm, dtype=np.uint64)

    def signature(self, items: set[str]) -> np.ndarray:
        hashes = np.array([int.from_bytes(hashlib.blake2b(item.encode("utf-8"), digest_size=4).digest(), "little")
                           for item in items], dtype=np.uint64)
        return ((np.outer(hashes, self.a) + self.b) % MERSENNE).min(axis=0)


class NearDuplicateIndex:
    """Exact (normalized prompt) and MinHash-LSH near-duplicate lookup against a reference set."""

    def __init__(self, texts, *, num_perm=128, bands=32, ngram=3, threshold=0.7, seed=0):
        self.hasher, self.bands, self.ngram, self.threshold = MinHasher(num_perm, seed=seed), bands, ngram, threshold
        self.rows = num_perm // bands
        self.exact = {normalize_text(text) for text in texts}
        unique = sorted(self.exact)  # Templated prompts repeat a lot; index each distinct prompt once.
        self.signatures = np.stack([self.hasher.signature(shingles(text, ngram)) for text in unique]) if unique else None
        buckets = defaultdict(list)
        for position in range(len(unique)):
            for band, key in self._band_keys(self.signatures[position]):
                buckets[(band, key)].append(position)
        self.buckets = {key: np.array(positions) for key, positions in buckets.items()}

    def _band_keys(self, signature):
        for band in range(self.bands):
            yield band, signature[band * self.rows:(band + 1) * self.rows].tobytes()

    def match(self, text: str) -> str | None:
        if normalize_text(text) in self.exact:
            return "exact"
        signature = self.hasher.signature(shingles(text, self.ngram))
        for key in self._band_keys(signature):
            positions = self.buckets.get(key)
            if positions is not None and ((self.signatures[positions] == signature).mean(axis=1) >= self.threshold).any():
                return "near"
        return None


# ---- Pipeline ----------------------------------------------------------------------------

def load_processed(directory: Path) -> list[dict]:
    examples = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            examples += [json.loads(line) for line in stream]
    return examples


def collect_tools(examples):
    """Unique tools keyed by their embedding text, with how many examples offer each."""
    texts, names, uses = {}, {}, Counter()
    for example in examples:
        for tool in example["tools"]:
            text = tool_text(tool)
            texts.setdefault(text, len(texts))
            names.setdefault(text, tool["name"])
            uses[text] += 1
    return list(texts), names, uses


def split_examples(examples, cluster_of_text, unseen_texts: set, *, holdout_per_group: int, holdout_chat: int,
                   seed: int):
    pools = {"train": [], "seen_eval": [], "unseen_eval": [], "chat_eval": []}
    mixed = Counter()
    seen_by_group = defaultdict(list)
    chat = []
    for example in examples:
        texts = [tool_text(tool) for tool in example["tools"]]
        example["meta"]["clusters"] = sorted({cluster_of_text[text] for text in texts})
        hidden = [text in unseen_texts for text in texts]
        if hidden and all(hidden):
            pools["unseen_eval"].append(example)
        elif any(hidden):
            mixed[example["group"]] += 1
        elif example["group"] in SEEN_EVAL_GROUPS:
            seen_by_group[example["group"]].append(example)
        elif example["source"].startswith("smoltalk"):
            chat.append(example)
        else:
            pools["train"].append(example)  # Glaive chat without tools and other leftovers: train only.
    rng = random.Random(seed)
    for group in sorted(seen_by_group):
        items = seen_by_group[group]
        rng.shuffle(items)
        pools["seen_eval"] += items[:holdout_per_group]
        pools["train"] += items[holdout_per_group:]
    rng.shuffle(chat)
    pools["chat_eval"], pools["train"] = chat[:holdout_chat], pools["train"] + chat[holdout_chat:]
    return pools, mixed


def dedupe_train(pools, minhash: dict, *, seed: int):
    eval_examples = [example for name in ("seen_eval", "unseen_eval", "chat_eval") for example in pools[name]]
    index = NearDuplicateIndex([prompt_text(example) for example in eval_examples], seed=seed, **minhash)
    kept, removed = [], Counter()
    for example in pools["train"]:
        verdict = index.match(prompt_text(example))
        if verdict:
            removed[(example["group"], verdict)] += 1
        else:
            kept.append(example)
    pools["train"] = kept
    return removed


def leakage_pairs(vectors, hidden: np.ndarray, texts, names, labels, *, threshold: float):
    """Unseen tools whose nearest seen tool is above the cosine threshold: candidates for a manual look.
    Returns (cosine, unseen name, unseen cluster, seen name, seen cluster), most similar first."""
    if not hidden.any() or hidden.all():
        return []
    seen_index = np.flatnonzero(~hidden)
    pairs = []
    for row in np.flatnonzero(hidden):
        similarity = vectors[seen_index] @ vectors[row]
        best = int(np.argmax(similarity))
        if similarity[best] >= threshold:
            other = seen_index[best]
            pairs.append((float(similarity[best]), names[texts[row]], int(labels[row]), names[texts[other]],
                          int(labels[other])))
    return sorted(pairs, reverse=True)


def final_checks(pools, mixed, removed, settings, examples_by_pool_strata) -> list:
    """Every split condition measured on the final pools: (condition, value, limit, passed or None)."""
    unseen = Counter(example["group"] for example in pools["unseen_eval"])
    checks = [(f"Mẫu unseen nhóm {group}", unseen[group], f"≥ {settings['min_unseen_per_group']}",
               unseen[group] >= settings["min_unseen_per_group"]) for group in UNSEEN_EVAL_GROUPS]
    tool_total = (sum(example["group"] in TOOL_GROUPS for pool in pools.values() for example in pool)
                  + sum(mixed.values()) + sum(count for (group, _), count in removed.items() if group in TOOL_GROUPS))
    lost = sum(unseen.values()) + sum(mixed.values())
    loss = lost / max(tool_total, 1)
    checks.append(("Tỉ lệ mẫu tool mất khỏi train (unseen + lẫn)", f"{loss:.1%}", f"≤ {settings['max_tool_loss']:.0%}",
                   loss <= settings["max_tool_loss"]))
    seen_strata, unseen_strata = examples_by_pool_strata
    checks.append(("Khoảng cách phân phối seen/unseen (nhóm, số tool, số tham số)",
                   f"{strata_distance(seen_strata, unseen_strata):.3f}", "chỉ báo", None))
    return checks


def render_split_report(*, config, clusters, unseen, choice, merge, absorbed, pools, mixed, removed, leaks,
                        checks, hashes, blocks=(), pinned=()) -> str:
    settings = config["split"]
    lines = ["# Báo cáo tách seen/unseen", "",
             "Sinh tự động bởi `python -m src.data.split`. Đừng sửa tay; chạy lại để cập nhật.", "",
             f"Embedder `{settings['embedder']['id']}` @ `{settings['embedder']['revision']}`, "
             f"{settings['n_clusters']} cụm (spherical k-means, seed {config['seed']}). "
             f"Tool gần như trùng nhau (cosine ≥ {settings['merge_threshold']}) được dồn về cùng một cụm: "
             f"{merge['relabelled']} tool đổi cụm, khối lớn nhất {merge['largest_component']} tool.", "",
             f"Chọn {len(unseen)} cụm unseen qua {settings['search_trials']} lần thử: chỉ nhận phương án đạt mọi "
             f"ràng buộc cứng, xếp theo khoảng cách phân phối cộng rò rỉ ranh giới (cặp tool hai phía có cosine ≥ "
             f"{settings['leak_threshold']}, trọng số {settings['leak_weight']}). Phương án được dùng đứng thứ "
             f"{choice['rank']} trong bảng xếp hạng ({choice['feasible_candidates']} phương án đạt ràng buộc lúc "
             f"tìm); {choice['rank'] - 1} phương án trước bị loại khi kiểm lại trên pool cuối.", "",
             f"Sau khi chọn cụm, tool seen có cosine ≥ {settings['absorb_threshold']} với một tool unseen, hoặc có tên "
             f"gần giống (trigram ký tự ≥ {settings['alias_similarity']} sau khi bỏ dấu gạch và chữ hoa) và cosine ≥ "
             f"{settings['name_absorb_threshold']}, bị kéo sang unseen (lặp tới khi hết): "
             f"{absorbed['tools']} tool qua {absorbed['rounds']} vòng.", "",
             f"Ứng viên unseen là các khối cụm: cụm có tâm gần nhau (cosine ≥ {settings['block_threshold']}) gộp "
             f"thành một khối và chỉ được chọn nguyên khối. Khối có độ đồng nhất (cosine trung bình giữa các tool) "
             f"< {settings['min_coherence']} hoặc bị loại tay trong `split.excluded_clusters` không được chọn.", "",
             (f"Cụm ghim unseen sau khi duyệt tay (`split.pinned_unseen`): {', '.join(map(str, pinned))}; "
              f"thuật toán chỉ chọn thêm phần còn lại." if pinned else "Không có cụm ghim unseen."), "",
             "## Kiểm tra trên pool cuối", "", "| Điều kiện | Giá trị | Ngưỡng | Kết quả |", "|---|---|---|---|"]
    for condition, value, limit, passed in checks:
        lines.append(f"| {condition} | {value} | {limit} | "
                     f"{'–' if passed is None else ('ĐẠT' if passed else '**KHÔNG ĐẠT**')} |")
    failed = [condition for condition, _, _, passed in checks if passed is False]
    lines += ["", "**Kết luận: " + ("đạt mọi ràng buộc cứng.**" if not failed else
                                    f"KHÔNG đạt {len(failed)} ràng buộc; chưa dùng được làm eval.**"), ""]
    lines += ["## Số mẫu theo pool", "", "| Nhóm | train | seen_eval | unseen_eval | chat_eval | Bỏ vì lẫn seen/unseen |",
              "|---|---|---|---|---|---|"]
    groups = sorted({example["group"] for pool in pools.values() for example in pool} | set(mixed))
    counts = {name: Counter(example["group"] for example in pool) for name, pool in pools.items()}
    for group in groups:
        lines.append(f"| {group} | {counts['train'][group]} | {counts['seen_eval'][group]} | "
                     f"{counts['unseen_eval'][group]} | {counts['chat_eval'][group]} | {mixed[group]} |")
    lines += ["", "## Miền unseen cần duyệt", "",
              "Mỗi cụm unseen là một miền chức năng bị giữ khỏi train. Duyệt: tool trong miền có cùng chức năng "
              "không, và các tool seen gần ranh giới có thực ra cùng chức năng (cần gộp hoặc loại) không.", ""]
    by_id = {cluster["id"]: cluster for cluster in clusters}
    for cluster_id in sorted(unseen, key=lambda cid: -by_id[cid]["uses"]):
        cluster = by_id[cluster_id]
        origin = "ghim" if cluster_id in pinned else "chọn thêm"
        lines += [f"### Cụm {cluster_id} ({origin}): {cluster['n_tools']} tool, được cung cấp {cluster['uses']} lần", "",
                  "Tool hay gặp nhất: " + ", ".join(f"`{name}`" for name in cluster["top"]) + ".", ""]
        neighbours = sorted(cluster["borders"].items(), key=lambda item: -item[1])[:3]
        if neighbours:
            lines.append("Cụm seen giáp ranh (số cặp tool gần nhau): " + "; ".join(
                f"cụm {other} ({count:.0f} cặp: {', '.join(f'`{name}`' for name in by_id[other]['top'][:3])})"
                for other, count in neighbours) + ".")
            lines.append("")
        border = [pair for pair in leaks if pair[2] == cluster_id][:5]
        if border:
            lines += ["| Cosine | Tool unseen | Tool seen gần nhất | Cụm seen |", "|---|---|---|---|"]
            lines += [f"| {score:.3f} | `{hidden}` | `{seen}` | {other} |" for score, hidden, _, seen, other in border]
            lines.append("")
    absorbed_names = absorbed.get("top", [])
    if absorbed_names:
        lines += ["Tool bị kéo từ cụm seen sang unseen (hay gặp nhất): "
                  + ", ".join(f"`{name}`" for name in absorbed_names) + ".", ""]
    by_id = {cluster["id"]: cluster for cluster in clusters}
    multi = [block for block in blocks if len(block["clusters"]) > 1 or block["excluded"]]
    if multi:
        lines += ["## Khối gộp và khối bị loại khỏi ứng viên", "",
                  "| Khối (cụm) | Độ đồng nhất | Bị loại vì | Tool hay gặp nhất |", "|---|---|---|---|"]
        for block in sorted(multi, key=lambda block: (not block["excluded"], block["clusters"])):
            sample = [name for cluster in block["clusters"] for name in by_id[cluster]["top"][:2]][:8]
            lines.append(f"| {', '.join(map(str, block['clusters']))} | {block['coherence']:.3f} | "
                         f"{block['excluded'] or '–'} | {', '.join(f'`{name}`' for name in sample)} |")
        lines.append("")
    lines += ["## Khử trùng lặp train với eval", "",
              f"MinHash {settings['minhash']['num_perm']} hoán vị, shingle {settings['minhash']['ngram']} từ, "
              f"ngưỡng Jaccard {settings['minhash']['threshold']}; so trên các lượt user.", "",
              "| Nhóm | Trùng exact | Gần trùng |", "|---|---|---|"]
    for group in sorted({group for group, _ in removed}):
        lines.append(f"| {group} | {removed[(group, 'exact')]} | {removed[(group, 'near')]} |")
    if not removed:
        lines.append("| – | 0 | 0 |")
    lines += ["", "## Cụm chức năng", "",
              "| Cụm | Unseen | Độ đồng nhất | Số tool | Tool bị kéo sang unseen | Số lần được cung cấp | Tool hay gặp nhất |",
              "|---|---|---|---|---|---|---|"]
    for cluster in clusters:
        lines.append(f"| {cluster['id']} | {'✔' if cluster['id'] in unseen else ''} | {cluster['coherence']:.3f} | "
                     f"{cluster['n_tools']} | "
                     f"{'' if cluster['id'] in unseen else cluster['absorbed'] or ''} | "
                     f"{cluster['uses']} | {', '.join(f'`{name}`' for name in cluster['top'][:8])} |")
    lines += ["", f"## Tool unseen gần tool seen nhất (cosine ≥ {settings['leak_threshold']}, {len(leaks)} tool)", ""]
    if leaks:
        lines += ["| Cosine | Tool unseen | Cụm | Tool seen gần nhất | Cụm |", "|---|---|---|---|---|"]
        lines += [f"| {score:.3f} | `{hidden}` | {a} | `{seen}` | {b} |" for score, hidden, a, seen, b in leaks[:40]]
    else:
        lines.append("Không có.")
    lines += ["", "## Hash", "", "| File | SHA-256 |", "|---|---|",
              *[f"| `{name}.jsonl` | `{digest}` |" for name, digest in hashes.items()], ""]
    return "\n".join(lines)


def write_jsonl(path: Path, rows) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(examples, config, *, embed=embed_texts):
    settings, seed = config["split"], config["seed"]
    texts, names, uses = collect_tools(examples)
    print(f"[split] embedding {len(texts)} unique tools…", flush=True)
    vectors = embed(texts, settings["embedder"])
    labels, merge = merge_near_identical(vectors, spherical_kmeans(vectors, settings["n_clusters"], seed=seed),
                                         threshold=settings["merge_threshold"])
    cluster_of_text = dict(zip(texts, labels.tolist()))
    adjacency = cluster_adjacency(vectors, labels, settings["n_clusters"], threshold=settings["leak_threshold"])
    tools_per_cluster = np.bincount(labels, minlength=settings["n_clusters"]).astype(float)

    top_tool_of = {}
    for cluster in range(settings["n_clusters"]):
        used = Counter()
        for row in np.flatnonzero(labels == cluster):
            used[names[texts[row]]] += uses[texts[row]]
        top_tool_of[cluster] = used.most_common(1)[0][0] if used else None
    excluded = check_exclusions(settings.get("excluded_clusters"), top_tool_of)
    pinned = sorted(check_exclusions(settings.get("pinned_unseen"), top_tool_of, key="pinned_unseen"))
    sums, sizes = cluster_sums(vectors, labels, settings["n_clusters"])
    candidates = []
    for block in cluster_blocks(sums, threshold=settings["block_threshold"]):
        score = coherence(sums, sizes, block)
        reason = ("; ".join(excluded[cluster] for cluster in block if cluster in excluded)
                  or (f"độ đồng nhất {score:.3f} < {settings['min_coherence']}" if score < settings["min_coherence"]
                      else ""))
        candidates.append({"clusters": block, "coherence": score, "excluded": reason})
    eligible = [block["clusters"] for block in candidates if not block["excluded"]]
    for block in candidates:  # A pinned cluster drags its whole block along, or that domain would leak.
        if set(block["clusters"]) & set(pinned) and not set(block["clusters"]) <= set(pinned):
            raise ValueError(f"pinned_unseen covers part of block {block['clusters']}; pin the whole block")

    records = [([cluster_of_text[tool_text(tool)] for tool in example["tools"]], difficulty(example))
               for example in examples if example["group"] in TOOL_GROUPS]
    search = UnseenSearch(records, settings["n_clusters"], adjacency=adjacency, tools_per_cluster=tools_per_cluster)
    n_unseen = max(1, round(settings["unseen_fraction"] * settings["n_clusters"]))
    print(f"[split] ranking {n_unseen}-cluster unseen sets from {len(eligible)} eligible blocks over "
          f"{settings['search_trials']} trials…", flush=True)
    ranked = search.rank(n_unseen, trials=settings["search_trials"], seed=seed, leak_weight=settings["leak_weight"],
                         blocks=eligible, pinned=pinned, min_per_group=settings["min_unseen_per_group"],
                         max_loss=settings["max_tool_loss"], eval_groups=UNSEEN_EVAL_GROUPS)
    feasible = sum(result["feasible"] for _, result in ranked)
    tool_names = [names[text] for text in texts]
    # Absorbing border tools moves examples after the search, so the hard limits are checked again on
    # the actual pools; the first candidate that still passes is used, else the top-ranked one.
    def attempt(unseen_list):
        in_cluster = np.array([label in set(unseen_list) for label in labels.tolist()])
        hidden, rounds = absorb_neighbors(vectors, in_cluster, threshold=settings["absorb_threshold"],
                                          names=tool_names, name_threshold=settings["name_absorb_threshold"],
                                          alias_similarity=settings["alias_similarity"])
        unseen_texts = {texts[row] for row in np.flatnonzero(hidden)}
        pools, mixed = split_examples(examples, cluster_of_text, unseen_texts,
                                      holdout_per_group=settings["holdout_per_group"],
                                      holdout_chat=settings["holdout_chat"], seed=seed)
        passed = all(ok is not False for _, _, _, ok in final_checks(pools, mixed, Counter(), settings,
                                                                     (Counter(), Counter())))
        return passed, (in_cluster, hidden, rounds, unseen_texts, pools, mixed)

    for rank, (unseen_list, choice) in enumerate(ranked[:settings["final_checks"]], start=1):
        passed, state = attempt(unseen_list)
        if passed:
            break
        print(f"[split] candidate {rank} fails on the final pools; trying the next", flush=True)
    else:
        # Nothing passes: keep the top-ranked attempt only so the report can show why; main() then
        # refuses to write or upload the pools.
        rank, (unseen_list, choice) = 1, ranked[0]
        passed, state = attempt(unseen_list)
    unseen = set(unseen_list)
    in_unseen_cluster, hidden, rounds, unseen_texts, pools, mixed = state
    choice = {**choice, "rank": rank, "feasible_candidates": feasible}
    absorbed_texts = [texts[row] for row in np.flatnonzero(hidden & ~in_unseen_cluster)]
    absorbed_uses = Counter()
    for text in absorbed_texts:
        absorbed_uses[names[text]] += uses[text]  # Variants of one name count once.
    absorbed = {"tools": len(absorbed_texts), "rounds": rounds,
                "top": [name for name, _ in absorbed_uses.most_common(15)]}
    print("[split] deduplicating train against eval pools…", flush=True)
    removed = dedupe_train(pools, settings["minhash"], seed=seed)
    seen_strata = Counter(difficulty(example) for name in ("train", "seen_eval") for example in pools[name]
                          if example["group"] in TOOL_GROUPS)
    unseen_strata = Counter(difficulty(example) for example in pools["unseen_eval"])
    checks = final_checks(pools, mixed, removed, settings, (seen_strata, unseen_strata))

    clusters = []
    for cluster in range(settings["n_clusters"]):
        members = [text for text in texts if cluster_of_text[text] == cluster]
        top = Counter()
        for text in members:
            top[names[text]] += uses[text]
        borders = {other: float(adjacency[cluster, other]) for other in range(settings["n_clusters"])
                   if other not in unseen and adjacency[cluster, other] > 0}
        clusters.append({"id": cluster, "n_tools": len(members), "uses": sum(uses[text] for text in members),
                         "coherence": coherence(sums, sizes, [cluster]),
                         "absorbed": sum(text in unseen_texts for text in members), "borders": borders,
                         "top": [name for name, _ in top.most_common(12)]})
    clusters.sort(key=lambda cluster: -cluster["uses"])
    leaks = leakage_pairs(vectors, hidden, texts, names, labels, threshold=settings["leak_threshold"])
    tools = [{"text": text, "name": names[text], "cluster": cluster_of_text[text], "uses": uses[text],
              "unseen": text in unseen_texts} for text in texts]
    return {"pools": pools, "mixed": mixed, "removed": removed, "unseen": unseen, "choice": choice, "merge": merge,
            "absorbed": absorbed, "clusters": clusters, "leaks": leaks, "checks": checks, "tools": tools,
            "blocks": candidates, "pinned": pinned,
            "passed": all(ok is not False for _, _, _, ok in checks)}


def fetch_processed(repo_id: str, directory: Path):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, repo_type="dataset", local_dir=str(directory), allow_patterns=["*.jsonl"],
                      ignore_patterns=["split/*"])


def push(directory: Path, repo_id: str, names):
    from huggingface_hub import upload_folder

    upload_folder(folder_path=str(directory), repo_id=repo_id, repo_type="dataset", path_in_repo="split",
                  allow_patterns=[f"{name}.jsonl" for name in names],
                  commit_message="Seen/unseen split (src.data.split)")


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--from-hub", help="Download data/processed from this dataset repo when it is missing.")
    parser.add_argument("--report", default="data/SPLIT.md")
    parser.add_argument("--push-to", help="Private HF dataset repo; files go under split/.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    processed = Path(config["output_dir"])
    if args.from_hub and not any(processed.glob("*.jsonl")):
        fetch_processed(args.from_hub, processed)
    examples = load_processed(processed)
    print(f"[split] {len(examples)} examples from {processed}", flush=True)
    result = run(examples, config)
    output_dir = Path(config["split"]["output_dir"])
    hashes = {}
    if result["passed"]:
        output_dir.mkdir(parents=True, exist_ok=True)
        hashes = {name: write_jsonl(output_dir / f"{name}.jsonl", rows) for name, rows in result["pools"].items()}
        hashes["tools"] = write_jsonl(output_dir / "tools.jsonl", result["tools"])
    report = render_split_report(config=config, clusters=result["clusters"], unseen=result["unseen"],
                                 choice=result["choice"], merge=result["merge"], absorbed=result["absorbed"],
                                 checks=result["checks"], blocks=result["blocks"], pinned=result["pinned"],
                                 pools=result["pools"],
                                 mixed=result["mixed"],
                                 removed=result["removed"], leaks=result["leaks"], hashes=hashes)
    Path(args.report).write_text(report, encoding="utf-8")
    for name, rows in result["pools"].items():
        print(f"[split] {name}: {len(rows)}", flush=True)
    if not result["passed"]:
        sys.exit(f"[split] the split fails its hard limits (see {args.report}); nothing written or uploaded")
    if args.push_to:
        push(output_dir, args.push_to, hashes)


if __name__ == "__main__":
    main()
