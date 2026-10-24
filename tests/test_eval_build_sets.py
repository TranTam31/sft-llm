from collections import Counter
import json
import random
import unittest

import numpy as np

from src.data.split import tool_text
from src.eval.build_sets import (NO_FITTING_TOOL_REPLY, GrowingIndex, apply_rewrite, assemble_labelled, assemble_no_tool,
                                 call_item, called_clusters, cap_clusters, chat_item, clarify_draft, multi_turn_draft,
                                 near_seen_tools, pick, rename_item, rewrite_draft, run, synonym_problems,
                                 synonym_table)
from src.eval.scorers import score_clarify, score_tool_sample
from src.tool_calls import format_tool_calls, format_tool_responses


def tool(name, params=("city",), description="A tool."):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {param: {"type": "string"} for param in params},
                           "required": list(params)}}


def call_example(id_, prompt, calls, tools, group="single", clusters=(0,)):
    return {"id": id_, "source": "xlam", "group": group, "tools": tools,
            "turns": [{"role": "user", "content": prompt}, {"role": "assistant", "content": format_tool_calls(calls)}],
            "meta": {"clusters": list(clusters)}}


WEATHER = tool("get_weather", ("city", "unit"))


class ItemTests(unittest.TestCase):
    def test_call_item_keeps_the_user_turn_and_the_calls(self):
        call = {"name": "get_weather", "arguments": {"city": "Hanoi", "unit": "c"}}
        item = call_item(call_example("x", "Weather in Hanoi?", [call], [WEATHER]))
        self.assertEqual(item["turns"], [{"role": "user", "content": "Weather in Hanoi?"}])
        self.assertEqual(item["expected_calls"], [call])

    def test_call_item_writes_decimals_as_plain_json_numbers(self):
        call = {"name": "get_weather", "arguments": {"city": "Hanoi", "unit": 0.1, "days": 3}}
        item = call_item(call_example("x", "Weather?", [call], [WEATHER]))
        self.assertEqual(json.loads(json.dumps(item["expected_calls"])), [call])
        output = format_tool_calls([call])
        self.assertTrue(score_tool_sample(output, expected_calls=item["expected_calls"], tools=[WEATHER])["call_exact"])

    def test_call_item_skips_a_prose_first_reply(self):
        example = call_example("x", "Hi", [{"name": "get_weather", "arguments": {}}], [WEATHER])
        example["turns"][1]["content"] = "Hello!"
        self.assertIsNone(call_item(example))

    def test_chat_item_prompts_with_every_turn_before_the_last_reply(self):
        turns = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                 {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]
        item = chat_item({"turns": turns})
        self.assertEqual(item["turns"], turns[:3])
        self.assertEqual(item["reference"], "d")

    def test_clarify_draft_proposes_arguments_absent_from_the_first_turn(self):
        call = {"name": "get_weather", "arguments": {"city": "Hanoi", "unit": "celsius"}}
        example = {"id": "g", "source": "glaive", "group": "clarify_candidate", "tools": [WEATHER], "meta": {},
                   "turns": [{"role": "user", "content": "What's the weather in celsius?"},
                             {"role": "assistant", "content": "Which city?"},
                             {"role": "user", "content": "Hanoi"},
                             {"role": "assistant", "content": format_tool_calls([call])}]}
        draft = clarify_draft(example)
        self.assertEqual(draft["turns"], example["turns"][:1])
        self.assertEqual(draft["draft"]["missing_param_candidates"], ["city"])
        self.assertEqual(draft["draft"]["later_call"], call)

    def test_multi_turn_draft_keeps_result_values_the_reference_repeats(self):
        call = {"name": "get_weather", "arguments": {"city": "Hanoi"}}
        result = format_tool_responses([{"name": "get_weather", "content": {"temp": -3, "wind": 12, "ok": True}}])
        example = {"id": "g", "source": "glaive", "group": "multi_turn", "tools": [WEATHER], "meta": {},
                   "turns": [{"role": "user", "content": "Weather in Hanoi?"},
                             {"role": "assistant", "content": format_tool_calls([call])},
                             {"role": "tool", "content": result},
                             {"role": "assistant", "content": "It is -3 degrees in Hanoi."}]}
        draft = multi_turn_draft(example)
        self.assertEqual(draft["turns"], example["turns"][:3])
        self.assertEqual(draft["draft"]["required_candidates"], [-3])  # 12 is not repeated; booleans never count


class RewriteDraftTests(unittest.TestCase):
    def test_required_values_stated_in_the_query_are_removable(self):
        call = {"name": "get_weather", "arguments": {"city": "Hanoi", "unit": "celsius"}}
        weather = tool("get_weather", ("city", "unit"))
        weather["parameters"]["required"] = ["city"]
        draft = rewrite_draft(call_example("x", "Weather in Hanoi in celsius?", [call], [weather]))
        self.assertEqual(draft["draft"]["removable"], [{"param": "city", "value": "Hanoi"}])
        self.assertEqual(draft["draft"]["original_call"], call)

    def test_requests_without_a_stated_required_value_or_with_several_calls_are_skipped(self):
        call = {"name": "get_weather", "arguments": {"city": "Hanoi", "unit": "c"}}
        self.assertIsNone(rewrite_draft(call_example("x", "Weather where I live?", [call], [WEATHER])))
        self.assertIsNone(rewrite_draft(call_example("x", "Hanoi c, twice", [call, call], [WEATHER], group="parallel")))


class ApplyRewriteTests(unittest.TestCase):
    def draft(self):
        call = {"name": "get_weather", "arguments": {"city": "Hanoi", "unit": "c"}}
        item = rewrite_draft(call_example("x", "Weather in Hanoi?", [call], [WEATHER]))
        return {"id": "t_seen:x:rewrite", "set": "t_seen", **item, "meta": {"source_id": "x"}}

    def test_a_rewrite_becomes_a_no_call_item_that_asks_for_the_parameter(self):
        item = apply_rewrite(self.draft(), {"query": "What's the weather like?", "missing_param": "city",
                                            "synonyms": ["location"]})
        self.assertEqual(item["turns"], [{"role": "user", "content": "What's the weather like?"}])
        self.assertEqual((item["expected_calls"], item["missing_param"], item["synonyms"]), ([], "city", ["location"]))
        self.assertEqual(item["meta"]["rewritten_from"], "Weather in Hanoi?")
        self.assertTrue(score_clarify("Which location?", missing_param="city", synonyms=item["synonyms"])["clarify_correct"])

    def test_a_query_that_still_states_the_value_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_rewrite(self.draft(), {"query": "Weather in hanoi please", "missing_param": "city"})
        with self.assertRaises(ValueError):
            apply_rewrite(self.draft(), {"query": "Weather?", "missing_param": "country"})


WORDS = ("amber basil cedar delta ember fjord grove harbor island jasper kettle lantern meadow nectar orchid "
         "pepper quartz raven saffron timber umber violet willow yarrow zephyr anchor bramble copper dune falcon").split()


def phrase(seed):
    """A request unlike any other seed's, so the near-duplicate check keeps it."""
    return " ".join(random.Random(seed).sample(WORDS, 8)) + "?"


def index():
    return GrowingIndex(num_perm=64, bands=16, ngram=3, threshold=0.7, seed=0)


class AssembleTests(unittest.TestCase):
    def assemble(self, drafts, labels, prompts=()):
        taken = index()
        for prompt in prompts:
            taken.add(prompt)
        return assemble_labelled(drafts, labels, n_test=3, n_dev=1, index=taken)

    def drafts(self, side, n_glaive, n_rewrite, n_multi):
        def draft(kind, index, extra):
            return {"id": f"{side}:{kind}{index}", "set": side, "tools": [WEATHER], "meta": {},
                    "turns": [{"role": "user", "content": f"{kind} {index}"}], **extra}
        call = {"name": "get_weather", "arguments": {"city": "Hanoi"}}
        result = {f"{side}_missing_info": [draft("g", i, {}) for i in range(n_glaive)],
                f"{side}_missing_info_rewrite": [draft("r", i, {"draft": {"removable": [{"param": "city", "value": "Hanoi"}],
                                                                           "original_call": call}})
                                                 for i in range(n_rewrite)],
                f"{side}_multi_turn": [draft("m", i, {"reference": "It is 31 C."}) for i in range(n_multi)]}
        # Prompts differ per draft so the near-duplicate check keeps them all.
        for name, items in result.items():
            for index_, item in enumerate(items):
                item["turns"] = [{"role": "user", "content": phrase(f"{name}{index_}")}]
        return result

    def labels(self, drafts):
        return {"missing_info_glaive": [{"id": d["id"], "missing_params": ["city"], "synonyms": ["location"]}
                                        for name, items in drafts.items() if name.endswith("_missing_info") for d in items],
                "missing_info_rewrite": [{"id": d["id"], "query": phrase(d["id"]), "missing_param": "city"}
                                         for name, items in drafts.items() if name.endswith("_rewrite") for d in items],
                "multi_turn": [{"id": d["id"], "required": [31], "forbidden": []}
                               for name, items in drafts.items() if name.endswith("_multi_turn") for d in items]
                              + [{"id": "t_seen:m0", "drop": "x"}, {"id": "nowhere", "drop": "x"}]}

    def test_both_sides_share_the_glaive_count_and_test_dev_keep_the_mix(self):
        drafts = {**self.drafts("t_seen", 8, 8, 6), **self.drafts("t_unseen", 2, 8, 6)}
        labels = self.labels(drafts)
        labels["multi_turn"] = [label for label in labels["multi_turn"] if label["id"] != "t_seen:m0"] + [
            {"id": "t_seen:m0", "drop": "x"}]
        sets, stats = self.assemble(drafts, labels)
        for side in ("t_seen", "t_unseen"):
            self.assertEqual((stats[side]["missing_info"]["glaive"], stats[side]["missing_info"]["rewrite"]), (2, 2))
            test = [item for item in sets["test"][side] if item["group"] == "missing_info"]
            self.assertEqual(sorted(item["meta"]["label_source"] for item in test), ["glaive", "glaive", "xlam_rewrite"])
            self.assertTrue(all(item["expected_calls"] == [] for item in test))
        multi = [item["id"] for split in ("test", "dev") for item in sets[split]["t_seen"] if item["group"] == "multi_turn"]
        self.assertNotIn("t_seen:m0", multi)
        self.assertEqual(len(multi), 4)
        self.assertEqual(stats["unmatched_labels"]["multi_turn"], 1)

    def test_a_side_short_of_rewrites_is_reported(self):
        drafts = {**self.drafts("t_seen", 1, 1, 4), **self.drafts("t_unseen", 1, 1, 4)}
        _, stats = self.assemble(drafts, self.labels(drafts))
        self.assertEqual(stats["t_seen"]["missing_info"]["picked"], 2)

    def test_a_rewrite_repeating_a_prompt_already_in_the_eval_is_skipped(self):
        drafts = {**self.drafts("t_seen", 1, 5, 4), **self.drafts("t_unseen", 1, 5, 4)}
        labels = self.labels(drafts)
        rewrites = labels["missing_info_rewrite"]
        rewrites[1]["query"] = rewrites[0]["query"]   # two drafts rewritten to one request
        rewrites[2]["query"] = "What is the weather today in my town, please?"
        sets, stats = self.assemble(drafts, labels, prompts=["What is the weather today in my town please"])
        self.assertEqual(stats["t_seen"]["missing_info"]["rewrite_repeated"], 2)
        prompts = [item["turns"][0]["content"] for split in ("test", "dev") for name in sets[split]
                   for item in sets[split][name]]
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertEqual(stats["t_seen"]["missing_info"]["picked"], 4)

    def test_both_sides_take_as_many_tool_dependent_items_as_the_poorer_side_has(self):
        drafts = {**self.drafts("t_seen", 4, 0, 8), **self.drafts("t_unseen", 4, 0, 8)}
        labels = self.labels(drafts)
        # Unseen has one item that needs the tool result (m0); seen has five derivable ones.
        for label in labels["multi_turn"]:
            if "required" in label:
                side, kind = label["id"].split(":")
                label["derivable"] = kind != "m0" if side == "t_unseen" else kind in ("m1", "m2", "m3", "m4", "m5")
        sets, stats = self.assemble(drafts, labels)
        for side in ("t_seen", "t_unseen"):
            self.assertEqual((stats[side]["multi_turn"]["picked"], stats[side]["multi_turn"]["derivable"]), (4, 3))
            used = [item for split in ("test", "dev") for item in sets[split][side] if item["group"] == "multi_turn"]
            self.assertEqual(sum(item["meta"]["derivable"] for item in used), 3)
            # Test keeps the 3:1 mix of the group, dev takes the rest.
            test = [item for item in sets["test"][side] if item["group"] == "multi_turn"]
            self.assertEqual(sum(item["meta"]["derivable"] for item in test), 2)

    def test_a_side_short_of_derivable_items_fills_up_with_the_others(self):
        drafts = {**self.drafts("t_seen", 4, 0, 8), **self.drafts("t_unseen", 4, 0, 8)}
        labels = self.labels(drafts)
        # Unseen needs three derivable items (one other item only); seen has two.
        for label in labels["multi_turn"]:
            if "required" in label:
                side, kind = label["id"].split(":")
                label["derivable"] = kind != "m0" if side == "t_unseen" else kind in ("m1", "m2")
        _, stats = self.assemble(drafts, labels)
        self.assertEqual([stats[side]["multi_turn"]["derivable"] for side in ("t_seen", "t_unseen")], [2, 3])
        self.assertEqual([stats[side]["multi_turn"]["picked"] for side in ("t_seen", "t_unseen")], [4, 4])


class FilterTests(unittest.TestCase):
    def test_unseen_tools_close_to_a_seen_tool_are_flagged(self):
        tools = [{"text": "a", "name": "a", "unseen": False}, {"text": "b", "name": "b", "unseen": True},
                 {"text": "c", "name": "c", "unseen": True}]
        vectors = np.array([[1.0, 0.0], [0.9, np.sqrt(1 - 0.81)], [0.0, 1.0]])
        near = near_seen_tools(tools, vectors, threshold=0.85)
        self.assertEqual(set(near), {"b"})
        self.assertAlmostEqual(near["b"][0], 0.9)
        self.assertEqual(near["b"][1], "a")


class RenameTests(unittest.TestCase):
    def item(self):
        call = {"name": "get_weather", "arguments": {"city": "Hanoi", "unit": "c"}}
        prompt = "Use get_weather for the city Hanoi."
        return {"id": "t_seen:x", "set": "t_seen", **call_item(call_example("x", prompt, [call], [WEATHER, tool("search")])),
                "meta": {}}

    def test_names_change_consistently_and_the_original_answer_still_scores(self):
        renamed, undeclared = rename_item(self.item(), seed=0)
        self.assertEqual(undeclared, [])
        new_name = renamed["meta"]["tool_map"]["get_weather"]
        self.assertRegex(new_name, r"^fn_[a-z0-9]{3}$")
        self.assertEqual(renamed["expected_calls"], [{"name": new_name, "arguments": {"p1": "Hanoi", "p2": "c"}}])
        self.assertEqual(list(renamed["tools"][0]["parameters"]["properties"]), ["p1", "p2"])
        self.assertEqual(renamed["tools"][0]["parameters"]["required"], ["p1", "p2"])
        self.assertEqual(renamed["tools"][0]["description"], WEATHER["description"])
        # Identifier-like names are replaced in the prompt; plain words such as "city" are not.
        self.assertEqual(renamed["turns"][0]["content"], f"Use {new_name} for the city Hanoi.")
        output = format_tool_calls(renamed["expected_calls"])
        self.assertTrue(score_tool_sample(output, expected_calls=renamed["expected_calls"],
                                          tools=renamed["tools"])["call_exact"])

    def test_renaming_is_deterministic_and_names_are_unique(self):
        first, _ = rename_item(self.item(), seed=0)
        second, _ = rename_item(self.item(), seed=0)
        self.assertEqual(first, second)
        self.assertEqual(len({tool["name"] for tool in first["tools"]}), 2)

    def test_a_parameter_shared_by_two_tools_follows_the_called_tool(self):
        read, write = tool("read_file", ("file_path", "encoding")), tool("write_file", ("content", "file_path"))
        call = {"name": "read_file", "arguments": {"file_path": "/a"}}
        item = {"id": "t_seen:f", "set": "t_seen", "meta": {},
                **call_item(call_example("f", "Call read_file with file_path=/a.", [call], [read, write]))}
        renamed, _ = rename_item(item, seed=0)
        new_name = renamed["meta"]["tool_map"]["read_file"]
        self.assertEqual(renamed["expected_calls"], [{"name": new_name, "arguments": {"p1": "/a"}}])
        self.assertEqual(renamed["turns"][0]["content"], f"Call {new_name} with p1=/a.")

    def test_names_are_replaced_in_one_pass(self):
        # p2 becomes p1 and x_y becomes p2; one pass keeps the new p2 from being renamed again to p1.
        item = {"id": "t_seen:g", "set": "t_seen", "meta": {},
                **call_item(call_example("g", "Set p2 and x_y.", [{"name": "set_opts", "arguments": {"p2": 1, "x_y": 2}}],
                                         [tool("set_opts", ("p2", "x_y"))]))}
        renamed, _ = rename_item(item, seed=0)
        self.assertEqual(renamed["turns"][0]["content"], "Set p1 and p2.")

    def test_a_parameter_renamed_differently_by_two_called_tools_is_left_unchanged(self):
        read, write = tool("read_file", ("file_path", "encoding")), tool("write_file", ("content", "file_path"))
        calls = [{"name": "read_file", "arguments": {"file_path": "/a"}},
                 {"name": "write_file", "arguments": {"content": "x", "file_path": "/b"}}]
        item = {"id": "t_seen:h", "set": "t_seen", "meta": {},
                **call_item(call_example("h", "Copy file_path /a to /b.", calls, [read, write], group="parallel"))}
        renamed, _ = rename_item(item, seed=0)
        self.assertEqual(renamed["turns"][0]["content"], "Copy file_path /a to /b.")
        self.assertEqual(renamed["meta"]["ambiguous_params"], ["file_path"])

    def test_arguments_missing_from_the_schema_are_reported(self):
        item = self.item()
        item["expected_calls"][0]["arguments"]["extra"] = 1
        _, undeclared = rename_item(item, seed=0)
        self.assertEqual(undeclared, ["get_weather.extra"])

    SYNONYMS = [{"id": "tool:get_weather", "synonym": "fetch_forecast"}, {"id": "tool:search", "synonym": "look_up"},
                {"id": "param:city", "synonym": "location"}, {"id": "param:unit", "synonym": "temperature_scale"},
                {"id": "param:search.city", "synonym": "town"}]

    def test_synonym_level_uses_the_labelled_names(self):
        synonyms = synonym_table(self.SYNONYMS)
        renamed, undeclared = rename_item(self.item(), seed=0, synonyms=synonyms)
        self.assertEqual(undeclared, [])
        self.assertEqual(renamed["id"], "t_seen:x:synonym")
        self.assertEqual(renamed["expected_calls"],
                         [{"name": "fetch_forecast", "arguments": {"location": "Hanoi", "temperature_scale": "c"}}])
        self.assertEqual([tool["name"] for tool in renamed["tools"]], ["fetch_forecast", "look_up"])
        # The per-tool entry wins over the shared one.
        self.assertEqual(list(renamed["tools"][1]["parameters"]["properties"]), ["town"])
        self.assertEqual(renamed["tools"][0]["parameters"]["required"], ["location", "temperature_scale"])
        self.assertEqual(renamed["turns"][0]["content"], "Use fetch_forecast for the city Hanoi.")
        self.assertEqual(synonym_problems([self.item()], synonyms), Counter())
        # The nonsense level is unchanged by the new argument.
        self.assertEqual(rename_item(self.item(), seed=0)[0]["meta"]["tool_map"]["get_weather"][:3], "fn_")

    def test_synonym_table_problems_are_reported(self):
        rows = [{"id": "tool:get_weather", "synonym": "getWeather"}, {"id": "tool:search", "synonym": "look_up"},
                {"id": "param:unit", "synonym": "scale"}, {"id": "param:search.city", "synonym": "scale"}]
        problems = synonym_problems([self.item()], synonym_table(rows))
        self.assertEqual(problems, Counter({"tool unchanged: get_weather": 1, "parameter without synonym: city": 1}))
        clash = synonym_table([*self.SYNONYMS, {"id": "tool:search", "synonym": "fetch_forecast"},
                               {"id": "param:unit", "synonym": "location"}])
        self.assertEqual(synonym_problems([self.item()], clash),
                         Counter({"parameters share a synonym in get_weather": 1, "tools share a synonym in t_seen:x": 1}))
        # Two tools that already share a name in the source keep sharing one: not a clash.
        twice = {**self.item(), "tools": [WEATHER, WEATHER, tool("search")]}
        self.assertEqual(synonym_problems([twice], synonym_table(self.SYNONYMS)), Counter())


class SamplingTests(unittest.TestCase):
    def test_pick_skips_repeated_prompts(self):
        index = GrowingIndex(num_perm=64, bands=16, ngram=3, threshold=0.7, seed=0)
        items = [{"turns": [{"role": "user", "content": text}]} for text in
                 ["what is the weather in hanoi today please", "What is the weather in Hanoi today, please?",
                  "convert ten dollars to euros right now", "sort these numbers from small to large"]]
        picked, skipped = pick(items, 10, index, rng=random.Random(0))
        self.assertEqual(len(picked), 3)
        self.assertEqual(skipped["exact"], 1)

    def test_cluster_cap_keeps_picked_items_that_fit_and_tops_up_the_rest(self):
        math_tool, text_tool = tool("add", ("a",)), tool("reverse", ("s",))
        cluster_of = {tool_text(math_tool): 29, tool_text(text_tool): 7}
        words = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo")

        def item(id_, word, called, split=None):
            # Both tools are offered; only the called one sets the cluster.
            entry = {"id": id_, "group": "single", "tools": [math_tool, text_tool],
                     "turns": [{"role": "user", "content": f"{id_} {phrase(word)}"}],
                     "expected_calls": [{"name": called["name"], "arguments": {"a": 1}}]}
            return {**entry, "split": split} if split else entry

        splits = {"test": [item(f"t{i}", words[i], math_tool, "test") for i in range(4)],
                  "dev": [item("d0", words[4], math_tool, "dev"), item("d1", words[5], text_tool, "dev")]}
        candidates = [item(f"m{i}", words[6 + i], math_tool) for i in range(2)]
        candidates += [item(f"r{i}", words[8 + i], text_tool) for i in range(1)]
        candidates.append({**item("dup", words[0], text_tool), "turns": splits["test"][0]["turns"]})
        candidates.append(item("r1", words[9], text_tool))
        index = GrowingIndex(num_perm=64, bands=16, ngram=3, threshold=0.7, seed=0)
        for items in splits.values():
            for entry in items:
                index.add(entry["turns"][0]["content"])
        capped, stats = cap_clusters(splits, candidates, {"test": 4, "dev": 2}, 0.5, index, rng=random.Random(0),
                                     cluster_of=cluster_of)
        self.assertEqual([entry["id"] for entry in capped["test"][:2]], ["t0", "t1"])
        self.assertEqual([entry["id"] for entry in capped["dev"][:2]], ["d0", "d1"])
        for split_name, limit in (("test", 2), ("dev", 1)):
            clusters = Counter(cluster for entry in capped[split_name] for cluster in called_clusters(entry, cluster_of))
            self.assertLessEqual(max(clusters.values()), limit)
            self.assertTrue(all(entry["split"] == split_name for entry in capped[split_name]))
        self.assertEqual(len(capped["test"]), 4)
        self.assertEqual((stats["kept"], stats["added"], stats["dropped"]), (4, 2, Counter({29: 2})))
        # "dup" fits the cap but repeats t0's prompt, so the top-up takes r0 and r1 whatever the order.
        self.assertEqual(sorted(entry["id"] for entry in capped["test"][2:]), ["r0", "r1"])


TOPIC = {"weather": 0, "stock": 1, "sports": 2}


def fake_embed(texts, _embedder):
    vectors = np.zeros((len(texts), 3))
    for row, text in enumerate(texts):
        vectors[row, next(axis for word, axis in TOPIC.items() if word in text)] = 1
    return vectors


def multi_turn_example(id_, city, clusters=(0,)):
    call = {"name": "get_weather", "arguments": {"city": city}}
    result = format_tool_responses([{"name": "get_weather", "content": {"temp": 30 + len(city)}}])
    return {"id": id_, "source": "glaive", "group": "multi_turn", "tools": [WEATHER], "meta": {"clusters": list(clusters)},
            "turns": [{"role": "user", "content": f"how warm is it over in {city} at the moment"},
                      {"role": "assistant", "content": format_tool_calls([call])},
                      {"role": "tool", "content": result},
                      {"role": "assistant", "content": f"It is {30 + len(city)} degrees."}]}


class RunTests(unittest.TestCase):
    def setUp(self):
        weather, stock = tool("get_weather", description="weather"), tool("stock_price", ("ticker",), "stock")
        sports, near = tool("score", ("team",), "sports"), tool("weather_alerts", description="weather alerts")
        seen = [call_example(f"s{i}", f"weather in city number {i} {'x' * i}", [{"name": "get_weather", "arguments": {"city": str(i)}}],
                             [weather, stock], group="select") for i in range(12)]
        unseen = [call_example(f"u{i}", f"score of team number {i} {'y' * i}", [{"name": "score", "arguments": {"team": str(i)}}],
                               [sports], clusters=(5,)) for i in range(8)]
        unseen += [call_example(f"n{i}", f"alerts for city {i} {'z' * i}", [{"name": "weather_alerts", "arguments": {"city": str(i)}}],
                                [near], clusters=(6,)) for i in range(5)]
        chat = [{"id": f"c{i}", "source": "smoltalk:x", "group": "chat", "tools": [], "meta": {},
                 "turns": [{"role": "user", "content": f"tell me fact {i} {'w' * i}"}, {"role": "assistant", "content": "ok"}]}
                for i in range(10)]
        tools = [{"text": tool_text(item), "name": item["name"], "unseen": unseen_flag, "cluster": cluster}
                 for cluster, (item, unseen_flag) in enumerate(((weather, False), (stock, False), (sports, True), (near, True)))]
        config = {"seed": 0, "split": {"embedder": {}, "minhash": {"num_perm": 64, "bands": 16, "ngram": 3, "threshold": 0.7}},
                  "eval_sets": {"test_per_group": 3, "dev_fraction": 0.3333, "chat_test": 3, "near_seen_threshold": 0.85,
                                "draft_margin": 1.5, "rewrite_drafts": {"t_seen": 5, "t_unseen": 5},
                                "multi_turn_extra_drafts": {"t_unseen": 10},
                                "no_tool": {"hard_fraction": 0.3333, "draft_margin": 2.0, "related_quantile": 0.33,
                                            "far_quantile": 0.5, "no_fitting_margin": 2.0}}}
        self.pools, self.tools, self.config = {"seen_eval": seen, "unseen_eval": unseen, "chat_eval": chat}, tools, config

    def test_sets_are_split_filtered_and_renamed(self):
        result = run(self.pools, self.tools, self.config, embed=fake_embed)
        # The fixture only has select requests on the seen side and single ones on the unseen side, and no
        # labels: every absent group is reported, so nothing is uploaded.
        self.assertEqual(result["short"], [("t_seen", "single", 0), ("t_seen", "parallel", 0), ("t_unseen", "select", 0),
                                           ("t_unseen", "parallel", 0), ("t_rename_synonym", "single", 0),
                                           ("t_rename_synonym", "select", 0), ("t_rename_synonym", "parallel", 0),
                                           ("t_seen", "missing_info", 0),
                                           ("t_seen", "multi_turn", 0), ("t_seen", "no_tool", 0),
                                           ("t_unseen", "missing_info", 0), ("t_unseen", "multi_turn", 0),
                                           ("t_unseen", "no_tool", 0)])
        self.assertEqual(result["dropped_unseen"], {"single": 5})
        test, dev = result["sets"]["test"], result["sets"]["dev"]
        self.assertEqual((len(test["t_seen"]), len(dev["t_seen"])), (3, 1))
        self.assertEqual((len(test["t_unseen"]), len(dev["t_unseen"])), (3, 1))
        self.assertTrue(all(item["meta"]["source_id"].startswith("u") for item in test["t_unseen"] + dev["t_unseen"]))
        self.assertEqual([item["meta"]["renamed_from"] for item in test["t_rename_nonsense"]],
                         [item["id"] for item in test["t_seen"]])
        self.assertEqual((len(test["c"]), len(dev["c"])), (3, 1))
        rewrites = result["drafts"]["t_seen_missing_info_rewrite"]
        self.assertTrue(rewrites)
        used = {item["meta"]["source_id"] for split in (test, dev) for item in split["t_seen"]}
        self.assertFalse(used & {item["meta"]["source_id"] for item in rewrites})
        ids = [item["meta"]["source_id"] for split in (test, dev) for name in ("t_seen", "t_unseen", "c") for item in split[name]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_cluster_cap_changes_only_the_unseen_call_groups(self):
        lottery = tool("draw", ("game",), "sports lottery")
        self.tools.append({"text": tool_text(lottery), "name": "draw", "unseen": True, "cluster": 4})
        self.pools["unseen_eval"] += [call_example(f"l{i}", f"lottery {phrase(i)}", [{"name": "draw", "arguments": {"game": str(i)}}],
                                                   [lottery], clusters=(4,)) for i in range(6)]
        plain_run = run(self.pools, self.tools, self.config, embed=fake_embed)
        self.config["eval_sets"]["unseen_cluster_cap"] = 0.67
        capped = run(self.pools, self.tools, self.config, embed=fake_embed)
        self.assertEqual(capped["drafts"], plain_run["drafts"])
        for split_name, size in (("test", 3), ("dev", 1)):
            for name in ("t_seen", "t_rename_nonsense", "c"):
                self.assertEqual(capped["sets"][split_name][name], plain_run["sets"][split_name][name])
            unseen = capped["sets"][split_name]["t_unseen"]
            self.assertEqual(len(unseen), size)
            per_cluster = Counter(item["meta"]["clusters"][0] for item in unseen)
            self.assertLessEqual(max(per_cluster.values()), max(1, int(0.67 * size)))
        self.assertEqual(capped["called_clusters"][("t_unseen", "single")], Counter({2: 2, 4: 2}))
        self.assertEqual(capped["short"], plain_run["short"])
        self.assertEqual(capped["stats"]["t_unseen"]["single"]["picked"], 4)

    def test_synonym_level_renames_t_seen_and_leaves_the_rest(self):
        plain_run = run(self.pools, self.tools, self.config, embed=fake_embed)
        rows = [{"id": "tool:get_weather", "synonym": "fetch_forecast"}, {"id": "tool:stock_price", "synonym": "share_quote"},
                {"id": "param:city", "synonym": "location"}, {"id": "param:ticker", "synonym": "symbol"}]
        result = run(self.pools, self.tools, self.config, labels={"rename_synonyms": rows}, embed=fake_embed)
        self.assertEqual(result["synonym_problems"], Counter())
        for split_name in ("test", "dev"):
            for name in ("t_seen", "t_unseen", "t_rename_nonsense", "c"):
                self.assertEqual(result["sets"][split_name][name], plain_run["sets"][split_name][name])
            synonym = result["sets"][split_name]["t_rename_synonym"]
            self.assertEqual([item["meta"]["renamed_from"] for item in synonym],
                             [item["id"] for item in result["sets"][split_name]["t_seen"]])
            self.assertTrue(all(call["name"] == "fetch_forecast" and list(call["arguments"]) == ["location"]
                                for item in synonym for call in item["expected_calls"]))
        self.assertEqual(result["drafts"], plain_run["drafts"])
        self.assertNotIn(("t_rename_synonym", "select", 0), result["short"])
        missing = run(self.pools, self.tools, self.config, labels={"rename_synonyms": rows[1:]}, embed=fake_embed)
        self.assertEqual(missing["synonym_problems"], Counter({"tool without synonym: get_weather": 4}))

    def test_labelled_groups_are_checked_on_labels_not_on_the_draft_margin(self):
        cities = ["Hanoi", "Hue", "Da Nang", "Can Tho", "Vinh", "Sapa", "Dalat", "Quy Nhon", "Nha Trang", "Ha Long"]
        self.pools["seen_eval"] += [multi_turn_example(f"ms{i}", city) for i, city in enumerate(cities[:5])]
        self.pools["unseen_eval"] += [multi_turn_example(f"mu{i}", city, clusters=(5,)) for i, city in enumerate(cities[5:])]
        drafts = run(self.pools, self.tools, self.config, embed=fake_embed)["drafts"]
        # Five drafts a side fall short of the 6 the margin asks for, but four labels fill the group.
        self.assertEqual([len(drafts[f"{side}_multi_turn"]) for side in ("t_seen", "t_unseen")], [5, 5])
        labels = {"multi_turn": [{"id": draft["id"], "required": [33], "forbidden": []}
                                 for side in ("t_seen", "t_unseen") for draft in drafts[f"{side}_multi_turn"][:4]]}
        result = run(self.pools, self.tools, self.config, labels=labels, embed=fake_embed)
        labelled_short = [entry for entry in result["short"] if entry[1] in ("missing_info", "multi_turn")]
        self.assertEqual(labelled_short, [("t_seen", "missing_info", 0), ("t_unseen", "missing_info", 0)])

    def test_no_tool_drafts_offer_tools_that_do_not_fit(self):
        stock = tool("stock_price", ("ticker",), "stock")
        self.pools["seen_eval"] += [call_example(f"q{i}", f"stock {phrase(word)}", [{"name": "stock_price", "arguments": {"ticker": word}}],
                                                 [stock]) for i, word in enumerate(("alpha", "bravo", "charlie", "delta", "echo",
                                                                                    "foxtrot", "golf", "hotel", "india", "juliet"))]
        # The Missing-info rewrite drafts draw from the same requests first.
        topics = ["why does the weather change with the {} season", "explain how sports leagues began in {}"]
        places = ("spring", "autumn", "winter", "summer", "Peru", "Chile")
        chat = [{"id": f"p{t}{i}", "source": "smoltalk:x", "group": "chat", "tools": [], "meta": {},
                 "turns": [{"role": "user", "content": topic.format(place)}, {"role": "assistant", "content": "Because."}]}
                for t, topic in enumerate(topics) for i, place in enumerate(places)]
        # A longer chat that opens like another prompt: its first turn alone would repeat.
        chat.append({**chat[1], "id": "p99", "turns": chat[1]["turns"] + [{"role": "user", "content": "And then?"},
                                                                           {"role": "assistant", "content": "Then."}]})
        self.pools["chat_eval"] += chat
        labels = {"no_tool_prompts": [{"id": example["id"], "keep": example["id"] != "p00"} for example in chat]}
        result = run(self.pools, self.tools, self.config, labels=labels, embed=fake_embed)
        seen = result["drafts"]["t_seen_no_tool"]
        self.assertEqual(Counter(draft["kind"] for draft in seen), {"plain": 4, "related": 2, "no_fitting_tool": 2})
        for draft in seen:
            names = {offered["name"] for offered in draft["tools"]}
            self.assertEqual(len(names), len(draft["tools"]))
            prompt = draft["turns"][0]["content"]
            self.assertTrue(names)
            self.assertNotEqual(prompt, chat[0]["turns"][0]["content"])   # dropped by the prompt labels
            if draft["kind"] == "related":
                self.assertIn("weather", prompt)
                self.assertEqual(names, {"get_weather"})
            elif draft["kind"] == "plain" and "weather" in prompt:
                self.assertEqual(names, {"stock_price"})
            elif draft["kind"] == "no_fitting_tool":
                self.assertEqual((names, draft["reference"]), ({"get_weather"}, NO_FITTING_TOOL_REPLY))
        # Unseen offers one cluster only, so plain and no_fitting_tool have no far tool to offer.
        unseen = result["drafts"]["t_unseen_no_tool"]
        self.assertEqual({draft["kind"] for draft in unseen}, {"related"})
        prompts = [draft["turns"][0]["content"] for draft in seen + unseen]
        self.assertEqual(len(prompts), len(set(prompts)))


class NoToolAssembleTests(unittest.TestCase):
    def test_both_sides_take_the_same_mix_and_test_keeps_it(self):
        def draft(side, kind, index):
            return {"id": f"{side}:{kind}{index}", "set": side, "group": "no_tool", "kind": kind, "tools": [WEATHER],
                    "turns": [{"role": "user", "content": "q"}], "expected_calls": [], "reference": "a", "meta": {}}
        drafts = {f"{side}_no_tool": [draft(side, kind, i) for kind in ("plain", "related", "no_fitting_tool") for i in range(8)]
                  for side in ("t_seen", "t_unseen")}
        labels = [{"id": item["id"], "ok": not item["id"].endswith("related0")} for items in drafts.values() for item in items]
        sets, stats = assemble_no_tool(drafts, labels, n_test=6, n_dev=3, hard_fraction=0.3333)
        for side in ("t_seen", "t_unseen"):
            # A third of 9 split over two hard kinds rounds to one each.
            self.assertEqual(stats[side]["no_tool"], {"picked": 9, "wanted": 9, "labelled": 23, "plain": 7, "related": 1,
                                                      "no_fitting_tool": 1})
            test = Counter(item["kind"] for item in sets["test"][side])
            self.assertEqual(test, {"plain": 4, "related": 1, "no_fitting_tool": 1})
            self.assertNotIn(f"{side}:related0", [item["id"] for split in ("test", "dev") for item in sets[split][side]])


if __name__ == "__main__":
    unittest.main()
