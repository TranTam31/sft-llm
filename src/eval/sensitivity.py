"""Scorer sensitivity check (PLAN 6, Phase 2): for every eval item, a correct output built from its labels and outputs
that each make one known error. Every error must score wrong and every correct output right.

Run: python -m src.eval.sensitivity [--split dev test]
Prints, per group and variant, how many items scored as expected, and up to a few failures for each.
"""

import argparse
import copy
import json

from src.eval.run_eval import ROOT, load_items, score
from src.tool_calls import format_tool_calls


END = "<|im_end|>"
CALL_GROUPS = ("single", "select", "parallel")


def calls_output(calls: list) -> str:
    return format_tool_calls(calls) + END


def _first_with_args(calls: list) -> int | None:
    return next((index for index, call in enumerate(calls) if call["arguments"]), None)


def _retyped(value):
    """The same content under another JSON type: 5 -> "5", "Hanoi" -> ["Hanoi"], True -> "true"."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return [value]
    return json.dumps(value)


def _changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + " x"
    return [] if value else ["x"]


def call_variants(item: dict) -> dict:
    """(output, stop_reason, expectations) per variant for Single/Select/Parallel and T-rename."""
    expected = item["expected_calls"]
    right = {"call_exact": True, "format_valid": True, "missed_call": False}
    wrong = {"call_exact": False}
    out = {"correct": (calls_output(expected), "im_end", right),
           "prose_around": ("Let me check.\n" + format_tool_calls(expected) + END, "im_end",
                            {**right, "strict_format": False}),
           "no_call": ("I can help with that.", "im_end", {"call_exact": False, "missed_call": True}),
           "broken_json": (calls_output(expected)[:-len(END)].replace("}\n</tool_call>", "\n</tool_call>", 1) + END,
                           "im_end", {"call_exact": False, "format_valid": False})}
    others = [tool["name"] for tool in item["tools"] if tool["name"] not in {call["name"] for call in expected}]
    renamed = copy.deepcopy(expected)
    renamed[0]["name"] = others[0] if others else renamed[0]["name"] + "_v2"
    out["wrong_name"] = (calls_output(renamed), "im_end", {**wrong, "tool_name_acc": False})
    index = _first_with_args(expected)
    if index is not None:
        key = next(iter(expected[index]["arguments"]))
        for name, change in (("missing_arg", None), ("wrong_type", _retyped), ("wrong_value", _changed)):
            calls = copy.deepcopy(expected)
            if change is None:
                del calls[index]["arguments"][key]
            else:
                calls[index]["arguments"][key] = change(calls[index]["arguments"][key])
            out[name] = (calls_output(calls), "im_end", wrong)
    extra = copy.deepcopy(expected)
    extra[0]["arguments"]["unexpected_param"] = 1
    out["extra_arg"] = (calls_output(extra), "im_end", wrong)
    out["duplicate_call"] = (calls_output(expected + expected[:1]), "im_end", wrong)
    if len(expected) > 1:
        out["dropped_call"] = (calls_output(expected[:-1]), "im_end", wrong)
    return out


def no_tool_variants(item: dict) -> dict:
    out = {"correct": (item["reference"], None, {"false_call": False, "called": False}),
           "calls_tool": (calls_output([{"name": item["tools"][0]["name"], "arguments": {}}]), "im_end",
                          {"false_call": True})}
    if item.get("kind") == "no_fitting_tool":
        out["correct"][2]["no_fitting_said"] = True
    return out


def clarify_variants(item: dict) -> dict:
    words = " ".join([item["missing_param"], *item["synonyms"]]).lower().replace("_", " ").split()
    other = next((name for tool in item["tools"] for name in tool["parameters"].get("properties", {})
                  if not set(name.lower().replace("_", " ").split()) & set(words)), "favourite colour")
    asked = item["missing_param"].replace("_", " ")
    right, wrong = {"clarify_correct": True}, {"clarify_correct": False}
    return {"correct": (f"Sure. Could you give me the {asked}?" + END, "im_end", right),
            "wrong_param": (f"Sure. Could you give me the {other.replace('_', ' ')}?" + END, "im_end", wrong),
            "no_question": (f"Sure, I will look up the {asked} for you." + END, "im_end", wrong),
            "calls_tool": (calls_output([{"name": item["tools"][0]["name"], "arguments": {}}]), "im_end", wrong),
            "truncated": (f"Sure. Could you give me the {asked}? Also", "max_new_tokens", wrong)}


def grounded_variants(item: dict) -> dict:
    required, forbidden = [str(v) for v in item["required"]], [str(v) for v in item["forbidden"]]
    right, wrong = {"grounded_rule": True}, {"grounded_rule": False}
    stated = "Here is what I found: " + ", ".join(required) + "."
    out = {"reference": (item["reference"], None, right),
           "correct": (stated + END, "im_end", right),
           "truncated": (stated, "max_new_tokens", wrong),
           "calls_again": (calls_output([{"name": item["tools"][0]["name"], "arguments": {}}]), "im_end", wrong)}
    if required:
        out["missing_value"] = ("Here is what I found: " + ", ".join(required[1:]) + "." + END, "im_end", wrong)
        out["negated_value"] = (f"It is not {required[0]}. " + ", ".join(required[1:]) + "." + END, "im_end", wrong)
    if forbidden:
        out["forbidden_value"] = (stated + f" Also {forbidden[0]}." + END, "im_end", wrong)
    return out


VARIANTS = {**{group: call_variants for group in CALL_GROUPS}, "no_tool": no_tool_variants,
            "missing_info": clarify_variants, "multi_turn": grounded_variants}


def check(items: list[dict]) -> dict:
    """{(group, variant): {"n": items, "failures": [(id, metric, expected, got, output)]}}."""
    report = {}
    for item in items:
        build = VARIANTS.get(item["group"])
        if build is None:
            continue
        for variant, (output, stop_reason, expectations) in build(item).items():
            scores = score(item, output, stop_reason)
            entry = report.setdefault((item["group"], variant), {"n": 0, "failures": []})
            entry["n"] += 1
            for metric, want in expectations.items():
                if scores[metric] != want:
                    entry["failures"].append((item["id"], metric, want, scores[metric], output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", nargs="+", default=["dev", "test"])
    parser.add_argument("--show", type=int, default=3, help="Failures printed per variant.")
    args = parser.parse_args()
    items = [item for split in args.split for item in load_items(split, ROOT, probe=False)]
    report = check(items)
    for (group, variant), entry in report.items():
        failed = len({failure[0] for failure in entry["failures"]})
        print(f"{group:13} {variant:16} {entry['n'] - failed:4}/{entry['n']:4} as expected")
        for failure in entry["failures"][:args.show]:
            print(f"    {failure[0]}: {failure[1]} expected {failure[2]}, got {failure[3]} | {failure[4][:160]!r}")
    total = sum(len({f[0] for f in entry["failures"]}) for entry in report.values())
    print(f"{total} (item, variant) pairs scored against expectation.")


if __name__ == "__main__":
    main()
