from collections import Counter
import unittest

import numpy as np

from src.data.split import (NearDuplicateIndex, UnseenSearch, absorb_neighbors, called_names, check_exclusions,
                            cluster_blocks, cluster_sums, coherence, compact_name, dedupe_train,
                            difficulty, final_checks, merge_near_identical, name_words, run, spherical_kmeans, split_examples,
                            name_similarity, strata_distance, tool_text, trigrams)
from src.tool_calls import format_tool_calls


def tool(name, description="", params=()):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {param: {"type": "string"} for param in params},
                           "required": list(params)}}


def example(id_, group, tools, prompt, calls=(), source="xlam"):
    turns = [{"role": "user", "content": prompt}]
    turns.append({"role": "assistant", "content": format_tool_calls(list(calls)) if calls else "Sure."})
    return {"id": id_, "source": source, "group": group, "tools": tools, "turns": turns, "meta": {}}


TOPICS = {"weather": 0, "forecast": 0, "stock": 1, "price": 1, "sum": 2, "math": 2}


def fake_embed(texts, _embedder):
    """Unit vector on the axis of the first topic word in the text."""
    vectors = np.zeros((len(texts), 3))
    for row, text in enumerate(texts):
        axis = next((TOPICS[word] for word in text.lower().replace(".", " ").split() if word in TOPICS), 2)
        vectors[row, axis] = 1
    return vectors + 1e-3 * np.arange(len(texts))[:, None] / len(texts)


class ToolTextTests(unittest.TestCase):
    def test_names_are_split_lowercased_and_lose_generic_prefixes(self):
        self.assertEqual(name_words("getWeatherForecast"), ["weather", "forecast"])
        self.assertEqual(name_words("fetch_stock_price"), ["stock", "price"])
        self.assertEqual(name_words("get"), ["get"])
        self.assertEqual(name_words("HTTPRequest_v2"), ["http", "request", "v2"])

    def test_tool_text_keeps_description_and_parameter_names(self):
        text = tool_text(tool("get_weather", "Current weather.", ["city", "unit"]))
        self.assertEqual(text, "weather. Current weather. Parameters: city, unit.")

    def test_called_names_reads_every_call_turn(self):
        ex = example("a", "parallel", [tool("a"), tool("b")], "q", [{"name": "a", "arguments": {}},
                                                                     {"name": "b", "arguments": {}}])
        self.assertEqual(called_names(ex), ["a", "b"])


class ClusteringTests(unittest.TestCase):
    def test_kmeans_separates_well_separated_directions(self):
        rng = np.random.default_rng(0)
        vectors = np.concatenate([np.eye(3)[axis] + 0.05 * rng.normal(size=(20, 3)) for axis in range(3)])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        labels = spherical_kmeans(vectors, 3, seed=0)
        self.assertEqual({len(set(labels[i * 20:(i + 1) * 20])) for i in range(3)}, {1})
        self.assertEqual(len(set(labels)), 3)

    def test_near_identical_tools_share_the_majority_cluster(self):
        vectors = np.array([[1, 0], [0.999, 0.045], [0.998, 0.063], [0, 1]], dtype=float)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        merged, stats = merge_near_identical(vectors, np.array([0, 0, 1, 1]), threshold=0.99)
        self.assertEqual(merged.tolist(), [0, 0, 0, 1])
        self.assertEqual(stats, {"relabelled": 1, "largest_component": 3})

    def test_seen_tools_near_an_unseen_tool_are_pulled_in_transitively(self):
        angles = np.radians([0, 20, 40, 90])  # cos(20 degrees) ~ 0.94: 0 -> 1 -> 2, but 3 stays seen
        vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        hidden, rounds = absorb_neighbors(vectors, np.array([True, False, False, False]), threshold=0.93)
        self.assertEqual(hidden.tolist(), [True, True, True, False])
        self.assertEqual(rounds, 3)

    def test_an_alias_needs_only_the_lower_name_threshold(self):
        angles = np.radians([0, 30, -30, -30])  # all ~0.87 from the unseen tool, 1 and 3 far apart
        vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        names = ["team_image", "teamimage", "team_roster", "check_file_existence"]
        hidden, _ = absorb_neighbors(vectors, np.array([True, False, False, False]), threshold=0.93,
                                     names=names, name_threshold=0.85, alias_similarity=0.6)
        self.assertEqual(hidden.tolist(), [True, True, False, False])

    def test_aliases_are_spotted_through_spelling_variants(self):
        def similar(a, b):
            return name_similarity(trigrams(compact_name(a)), trigrams(compact_name(b)))
        self.assertEqual(similar("team_image", "teamImage"), 1.0)
        self.assertGreaterEqual(similar("check_file_existence", "check_file_exists"), 0.6)
        self.assertLess(similar("getteamroster", "getteamstats"), 0.6)

    def test_rank_puts_feasible_sets_first_and_prefers_clean_borders(self):
        # Clusters 0 and 1 share near-identical tools; 2 and 3 are isolated. Every cluster has 5 examples.
        records = [([cluster], ("single", 0, 1)) for cluster in range(4) for _ in range(5)]
        adjacency = np.zeros((4, 4))
        adjacency[0, 1] = adjacency[1, 0] = 10
        search = UnseenSearch(records, 4, adjacency=adjacency, tools_per_cluster=np.ones(4))
        ranked = search.rank(1, trials=50, seed=0, min_per_group=5, max_loss=0.3, eval_groups=("single",))
        self.assertIn(ranked[0][0], ([2], [3]))
        self.assertEqual(ranked[0][1]["leak"], 0)
        strict = search.rank(1, trials=50, seed=0, min_per_group=6, max_loss=0.3, eval_groups=("single",))
        self.assertFalse(any(result["feasible"] for _, result in strict))
        self.assertEqual(strict[0][1]["short_groups"], ["single"])

    def test_rank_keeps_blocks_whole(self):
        records = [([cluster], ("single", 0, 1)) for cluster in range(5) for _ in range(5)]
        search = UnseenSearch(records, 5)
        ranked = search.rank(2, trials=40, seed=0, blocks=[[0, 1], [2], [3]], min_per_group=0, max_loss=1.0,
                             eval_groups=("single",))
        for clusters, _ in ranked:
            self.assertEqual(0 in clusters, 1 in clusters)
            self.assertNotIn(4, clusters)  # not in any eligible block

    def test_pinned_clusters_are_in_every_candidate(self):
        records = [([cluster], ("single", 0, 1)) for cluster in range(5) for _ in range(5)]
        ranked = UnseenSearch(records, 5).rank(3, trials=30, seed=0, pinned=[0, 1], min_per_group=0, max_loss=1.0,
                                               eval_groups=("single",))
        self.assertTrue(ranked)
        for clusters, _ in ranked:
            self.assertEqual(len(clusters), 3)
            self.assertTrue({0, 1} <= set(clusters))

    def test_blocks_join_close_centroids_and_coherence_spots_catch_alls(self):
        vectors = np.array([[1, 0, 0], [0.99, 0.14, 0], [0.98, 0.2, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        labels = np.array([0, 0, 1, 2, 2])  # cluster 2 mixes two unrelated tools
        sums, sizes = cluster_sums(vectors, labels, 3)
        self.assertEqual(cluster_blocks(sums, threshold=0.95), [[0, 1], [2]])
        self.assertAlmostEqual(coherence(sums, sizes, [2]), 0.0)
        self.assertGreater(coherence(sums, sizes, [0, 1]), 0.95)

    def test_excluded_clusters_must_still_start_with_their_pinned_tool(self):
        entry = {"id": 3, "top_tool": "search", "reason": "catch-all"}
        self.assertEqual(check_exclusions([entry], {3: "search"}), {3: "catch-all"})
        with self.assertRaises(ValueError):
            check_exclusions([entry], {3: "get_recipe"})

    def test_parameters_are_counted_on_the_example_own_schema(self):
        wide = tool("search", "Search hotels.", ["city", "checkin", "checkout", "guests", "rooms"])
        ex = example("w", "single", [wide], "q", [{"name": "search", "arguments": {}}])
        self.assertEqual(difficulty(ex), ("single", 0, 4))  # 5 parameters: the open-ended bucket

    def test_distance_is_zero_for_equal_mixes_and_grows_with_difference(self):
        a = Counter({("single", 0, 1): 2, ("select", 1, 2): 2})
        self.assertEqual(strata_distance(a, Counter({("single", 0, 1): 1, ("select", 1, 2): 1})), 0)
        self.assertAlmostEqual(strata_distance(a, Counter({("single", 0, 1): 1})), 1.5)


class DedupeTests(unittest.TestCase):
    def test_exact_and_near_duplicates_are_found_but_different_prompts_are_not(self):
        index = NearDuplicateIndex(["What is the weather in Paris today and tomorrow for my long trip?"], seed=0)
        self.assertEqual(index.match("what is the WEATHER in Paris today, and tomorrow for my long trip"), "exact")
        self.assertEqual(index.match("What is the weather in Paris today and tomorrow for my long trip please?"),
                         "near")
        self.assertIsNone(index.match("Compute the sum of 3 and 4."))

    def test_train_loses_examples_that_duplicate_eval(self):
        pools = {"train": [example("t1", "single", [], "Sum 3 and 4."), example("t2", "single", [], "Hello there.")],
                 "seen_eval": [example("e", "single", [], "sum 3 and 4")], "unseen_eval": [], "chat_eval": []}
        removed = dedupe_train(pools, {"num_perm": 64, "bands": 16, "ngram": 3, "threshold": 0.7}, seed=0)
        self.assertEqual([ex["id"] for ex in pools["train"]], ["t2"])
        self.assertEqual(dict(removed), {("single", "exact"): 1})


class RunTests(unittest.TestCase):
    def test_end_to_end_with_a_fake_embedder(self):
        tools = [tool("get_weather", "Weather now.", ["city"]), tool("forecast", "Forecast.", ["city", "days"]),
                 tool("stock_price", "Stock price.", ["ticker"]), tool("sum", "Math sum.", ["a", "b"])]
        examples = []
        for index in range(60):
            chosen = tools[index % 4]
            call = {"name": chosen["name"], "arguments": {}}
            examples.append(example(f"x{index}", "single", [chosen], f"request number {index} about {chosen['name']}",
                                    [call]))
        config = {"seed": 0, "split": {"embedder": {"id": "fake", "revision": "0"}, "n_clusters": 3,
                                        "unseen_fraction": 0.34, "search_trials": 20, "min_unseen_per_group": 0,
                                        "max_tool_loss": 0.5, "holdout_per_group": 5, "holdout_chat": 0,
                                        "merge_threshold": 0.999, "absorb_threshold": 0.999, "name_absorb_threshold": 0.999,
                                        "alias_similarity": 1.0,
                                        "leak_threshold": 0.99, "leak_weight": 1.0, "final_checks": 5,
                                        "block_threshold": 0.999, "min_coherence": 0.0,
                                        "minhash": {"num_perm": 64, "bands": 16, "ngram": 3, "threshold": 0.7}}}
        result = run(examples, config, embed=fake_embed)
        self.assertEqual(len(result["unseen"]), 1)
        unseen_names = {t["name"] for t in result["tools"] if t["unseen"]}
        train_names = {t["name"] for ex in result["pools"]["train"] for t in ex["tools"]}
        self.assertTrue(unseen_names)
        self.assertFalse(unseen_names & train_names)
        self.assertEqual(len(result["pools"]["seen_eval"]), 5)
        total = sum(len(pool) for pool in result["pools"].values()) + sum(result["mixed"].values())
        self.assertEqual(total + sum(result["removed"].values()), 60)
        self.assertTrue(all(passed is not False for _, _, _, passed in result["checks"]))
        self.assertTrue(result["passed"])

    def test_a_split_that_misses_its_limits_is_flagged(self):
        tools = [tool("get_weather", "Weather now.", ["city"]), tool("sum", "Math sum.", ["a", "b"])]
        examples = [example(f"x{index}", "single", [tools[index % 2]], f"request {index}",
                            [{"name": tools[index % 2]["name"], "arguments": {}}]) for index in range(20)]
        config = {"seed": 0, "split": {"embedder": {"id": "fake", "revision": "0"}, "n_clusters": 2,
                                        "unseen_fraction": 0.5, "search_trials": 10, "min_unseen_per_group": 500,
                                        "max_tool_loss": 0.9, "holdout_per_group": 2, "holdout_chat": 0,
                                        "merge_threshold": 0.999, "absorb_threshold": 0.999,
                                        "name_absorb_threshold": 0.999, "alias_similarity": 1.0,
                                        "leak_threshold": 0.99, "leak_weight": 1.0, "final_checks": 3,
                                        "block_threshold": 0.999, "min_coherence": 0.0,
                                        "minhash": {"num_perm": 64, "bands": 16, "ngram": 3, "threshold": 0.7}}}
        self.assertFalse(run(examples, config, embed=fake_embed)["passed"])

    def test_final_checks_flag_short_groups_and_excess_loss(self):
        settings = {"min_unseen_per_group": 2, "max_tool_loss": 0.3}
        pools = {"train": [example("t", "single", [], "q")], "seen_eval": [], "chat_eval": [],
                 "unseen_eval": [example("u", "single", [], "q")]}
        checks = {condition: passed for condition, _, _, passed in
                  final_checks(pools, Counter(), Counter(), settings, (Counter(), Counter()))}
        self.assertFalse(checks["Mẫu unseen nhóm single"])
        self.assertFalse(checks["Tỉ lệ mẫu tool mất khỏi train (unseen + lẫn)"])  # 1 of 2 = 50%


if __name__ == "__main__":
    unittest.main()
