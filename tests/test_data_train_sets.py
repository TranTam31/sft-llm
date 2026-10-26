from collections import Counter
import unittest

from src.data.split import tool_text
from src.data.train_sets import asks_for_missing, run, spread, targets
from src.eval.build_sets import NO_FITTING_TOOL_REPLY
from src.tool_calls import format_tool_calls, format_tool_responses
from test_eval_build_sets import call_example, fake_embed, phrase, tool  # tests/ is on sys.path under discover


WEATHER = tool("get_weather", ("city", "unit"), "weather")
STOCK = tool("stock_price", ("ticker",), "stock")
SPORTS = tool("score", ("team",), "sports")


def label(draft, ok=True):
    return {"id": draft["id"], "tools": sorted(offered["name"] for offered in draft["tools"]), "ok": ok}


def chat(id_, topic):
    return {"id": id_, "source": "smoltalk:x", "group": "chat", "tools": [], "meta": {"n_tokens": 10},
            "turns": [{"role": "user", "content": f"{topic} {phrase(id_)}"}, {"role": "assistant", "content": "Because."}]}


def clarify(id_, first_turn, city="Hanoi"):
    call = {"name": "get_weather", "arguments": {"city": city, "unit": "celsius"}}
    return {"id": id_, "source": "glaive", "group": "clarify_candidate", "tools": [WEATHER], "meta": {},
            "turns": [{"role": "user", "content": first_turn},
                      {"role": "assistant", "content": "Which city?"},
                      {"role": "user", "content": city},
                      {"role": "assistant", "content": format_tool_calls([call])}]}


def multi_turn(id_):
    call = {"name": "get_weather", "arguments": {"city": "Hue", "unit": "c"}}
    return {"id": id_, "source": "glaive", "group": "multi_turn", "tools": [WEATHER], "meta": {},
            "turns": [{"role": "user", "content": f"weather {phrase(id_)}"},
                      {"role": "assistant", "content": format_tool_calls([call])},
                      {"role": "tool", "content": format_tool_responses([{"name": "get_weather", "content": {"t": 30}}])},
                      {"role": "assistant", "content": "It is 30 degrees."}]}


class TargetTests(unittest.TestCase):
    def test_the_decided_shares_give_the_group_sizes(self):
        settings = {"n_tool": 10000, "shares": {"multi_turn": 0.2, "missing_info": 0.1, "no_tool": 0.2},
                    "call_split": {"single": 0.34, "select": 0.34}}
        self.assertEqual(targets(settings), {"multi_turn": 2000, "missing_info": 1000, "no_tool": 2000,
                                             "single": 1700, "select": 1700, "parallel": 1600})

    def test_missing_info_shortfall_goes_to_the_call_groups(self):
        counts = {"multi_turn": 2000, "missing_info": 1000, "no_tool": 2000, "single": 1700, "select": 1700,
                  "parallel": 1600}
        spread_counts = spread(300, counts)
        self.assertEqual(sum(spread_counts[group] for group in ("single", "select", "parallel")), 5300)
        self.assertEqual(spread_counts["multi_turn"], 2000)

    def test_missing_info_needs_a_required_value_the_first_turn_lacks(self):
        self.assertTrue(asks_for_missing(clarify("a", "What's the weather in celsius?")))
        # The city is already in the first turn, and unit is stated too: nothing was missing.
        self.assertFalse(asks_for_missing(clarify("b", "What's the weather in Hanoi in celsius?")))


class RunTests(unittest.TestCase):
    def setUp(self):
        self.pool = [chat(f"c{i}", ("weather", "stock")[i % 2]) for i in range(12)]
        self.pool += [call_example(f"s{i}", f"stock {phrase(f's{i}')}", [{"name": "stock_price", "arguments": {"ticker": str(i)}}],
                                   [STOCK]) for i in range(8)]
        self.pool += [call_example(f"e{i}", f"weather {phrase(f'e{i}')}", [{"name": "get_weather", "arguments": {"city": str(i)}}],
                                   [WEATHER, STOCK], group="select") for i in range(4)]
        self.pool += [call_example(f"p{i}", f"weather {phrase(f'p{i}')}",
                                   [{"name": "get_weather", "arguments": {"city": str(i)}},
                                    {"name": "stock_price", "arguments": {"ticker": str(i)}}],
                                   [WEATHER, STOCK], group="parallel") for i in range(5)]
        self.pool += [multi_turn(f"m{i}") for i in range(5)]
        self.pool += [clarify(f"g{i}", f"weather in celsius {phrase(f'g{i}')}") for i in range(3)]
        self.pool.append(clarify("g_full", "What's the weather in Hanoi in celsius?"))
        self.tools = [{"text": tool_text(item), "name": item["name"], "unseen": unseen, "cluster": cluster}
                      for cluster, (item, unseen) in enumerate(((WEATHER, False), (STOCK, False), (SPORTS, True)))]
        self.eval_items = [{"turns": [{"role": "user", "content": self.pool[0]["turns"][0]["content"]}]},
                           {"turns": [{"role": "user", "content": self.pool[2]["turns"][0]["content"]},
                                      {"role": "assistant", "content": "Sure."},
                                      {"role": "user", "content": f"and also {phrase('later')}"}]}]
        self.config = {"seed": 0, "split": {"embedder": {}, "minhash": {"num_perm": 64, "bands": 16, "ngram": 3,
                                                                         "threshold": 0.7}},
                       "train_sets": {"n_chat": 4, "n_tool": 20,
                                      "shares": {"multi_turn": 0.2, "missing_info": 0.1, "no_tool": 0.2},
                                      "call_split": {"single": 0.34, "select": 0.34},
                                      "no_tool": {"hard_fraction": 0.3333, "draft_margin": 2.0, "related_quantile": 0.5,
                                                  "far_quantile": 0.5, "no_fitting_margin": 2.0}}}

    def build(self, labels=None):
        return run(self.pool, self.tools, self.eval_items, self.config, labels=labels, embed=fake_embed,
                   measure=lambda example: 50)

    def test_without_labels_only_d_chat_and_the_drafts_are_built(self):
        result = self.build()
        self.assertIsNone(result["d_tool"])
        self.assertEqual(len(result["d_chat"]), 4)
        self.assertNotIn("c0", {item["id"] for item in result["d_chat"]})       # repeats an eval prompt
        self.assertNotIn("c2", {item["id"] for item in result["d_chat"]})       # opens like a multi-turn eval item
        self.assertEqual(result["stats"]["missing_info"]["filtered"], 3)        # g_full asks for nothing missing
        used = {item["id"] for item in result["d_chat"]}
        for draft in result["drafts"]:
            self.assertNotIn(draft["meta"]["source_id"], used)
            self.assertTrue(draft["tools"])
            self.assertFalse({offered["name"] for offered in draft["tools"]} & {"score"})  # seen tools only
        self.assertEqual(Counter(draft["kind"] for draft in result["drafts"]),
                         {"plain": 4, "related": 2, "no_fitting_tool": 2})

    def test_labelled_drafts_complete_d_tool_with_the_decided_mix(self):
        drafts = self.build()["drafts"]
        result = self.build(labels=[label(draft) for draft in drafts])
        d_tool = result["d_tool"]
        self.assertEqual(Counter(item["group"] for item in d_tool),
                         {"single": 3, "select": 3, "parallel": 4, "multi_turn": 4, "missing_info": 2, "no_tool": 4})
        self.assertEqual(Counter(item.get("kind") for item in d_tool if item["group"] == "no_tool"),
                         {"plain": 2, "related": 1, "no_fitting_tool": 1})
        no_fitting = next(item for item in d_tool if item.get("kind") == "no_fitting_tool")
        self.assertEqual(no_fitting["turns"][-1], {"role": "assistant", "content": NO_FITTING_TOOL_REPLY})
        singles = {item["id"] for item in d_tool if item["group"] == "single"}
        self.assertNotIn(no_fitting["meta"]["source_id"], singles)
        firsts = [item["turns"][0]["content"] for item in d_tool + result["d_chat"]]
        self.assertEqual(len(firsts), len(set(firsts)))
        self.assertEqual(self.build(labels=[label(draft) for draft in drafts])["d_tool"], d_tool)

    def test_too_few_good_labels_leave_d_tool_unwritten(self):
        drafts = self.build()["drafts"]
        labels = [label(draft, draft["kind"] != "related") for draft in drafts]
        result = self.build(labels=labels)
        self.assertIsNone(result["d_tool"])
        self.assertEqual(result["stats"]["no_tool"]["related"]["picked"], 0)

    def rewrite_setup(self):
        """Requests that state their ticker, a Missing-info share the Glaive pool cannot fill, and rewrite settings."""
        self.pool += [call_example(f"r{i}", f"stock price for ticker QX{i}Z {phrase(f'r{i}')}",
                                   [{"name": "stock_price", "arguments": {"ticker": f"QX{i}Z"}}], [STOCK])
                      for i in range(6)]
        settings = self.config["train_sets"]
        settings["shares"] = {"multi_turn": 0.2, "missing_info": 0.25, "no_tool": 0.2}
        settings["missing_info_rewrite"] = {"draft_margin": 1.5}

    def rewrite_label(self, draft, **fields):
        return {"id": draft["id"], "ok": True, "missing_param": "ticker", "synonyms": ["ticker symbol", "which stock"],
                "query": f"stock price for a company please {phrase(draft['id'])}",
                "reply": "Sure. Which ticker symbol should I look up?", **fields}

    def test_rewrites_fill_the_missing_info_shortfall_and_take_the_last_call_items(self):
        self.rewrite_setup()
        first = self.build()
        rewrites = first["rewrite_drafts"]
        self.assertEqual(len(rewrites), 3)                       # shortfall 2, margin 1.5
        self.assertTrue(all(draft["id"].startswith("train:r") for draft in rewrites))
        no_tool = [label(draft) for draft in first["drafts"]]
        self.assertIsNone(self.build(labels=no_tool)["d_tool"])  # rewrites not labelled yet
        before = {group: [item["id"] for item in first["d_chat"]] for group in ("chat",)}
        result = run(self.pool, self.tools, self.eval_items, self.config, labels=no_tool,
                     rewrite_labels=[self.rewrite_label(draft) for draft in rewrites], embed=fake_embed,
                     measure=lambda example: 50)
        d_tool = result["d_tool"]
        counts = Counter(item["group"] for item in d_tool)
        self.assertEqual(counts["missing_info"], 5)              # 3 Glaive + 2 rewrites
        self.assertEqual(sum(counts[group] for group in ("single", "select", "parallel")),
                         20 - 5 - 4 - 4)
        added = [item for item in d_tool if item["meta"].get("label_source") == "xlam_rewrite"]
        self.assertEqual(len(added), 2)
        self.assertEqual(added[0]["turns"][-1]["role"], "assistant")
        self.assertNotIn("QX", added[0]["turns"][0]["content"])
        self.assertEqual(result["drafts"], first["drafts"])      # No-tool drafts stay as labelled
        self.assertEqual(before["chat"], [item["id"] for item in result["d_chat"]])

    def test_a_rewrite_label_that_still_states_the_value_or_does_not_ask_is_an_error(self):
        self.rewrite_setup()
        first = self.build()
        no_tool = [label(draft) for draft in first["drafts"]]
        drafts = first["rewrite_drafts"]
        still_states = [self.rewrite_label(draft, query=draft["turns"][0]["content"]) for draft in drafts]
        no_question = [self.rewrite_label(draft, reply="Sure, here it is.") for draft in drafts]
        for bad, labels in (("query", still_states), ("reply", no_question)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run(self.pool, self.tools, self.eval_items, self.config, labels=no_tool, rewrite_labels=labels,
                    embed=fake_embed, measure=lambda example: 50)

    def test_a_label_holds_only_for_the_tools_it_saw(self):
        drafts = self.build()["drafts"]
        labels = [label(draft) for draft in drafts]
        for item, draft in zip(labels, drafts):
            if draft["kind"] == "related":
                item["tools"] = ["some_other_tool"]   # labelled for an offer the draft no longer makes
        result = self.build(labels=labels)
        self.assertIsNone(result["d_tool"])
        self.assertEqual(result["stats"]["no_tool"]["unmatched_labels"], 2)
        self.assertEqual(result["stats"]["no_tool"]["related"]["picked"], 0)


if __name__ == "__main__":
    unittest.main()
