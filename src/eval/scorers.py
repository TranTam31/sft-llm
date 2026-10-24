"""Rule-based per-sample metrics from PLAN 3.3; group denominators are always every sample.

An output is read two ways:
- extract_tool_calls (lenient): finds every <tool_call> block anywhere, so prose around
  a call does not hide the decision to call. Drives format_valid, call_exact and the
  call/no-call decision metrics, like BFCL's AST checking.
- tool_calls.parse_tool_calls (strict): the whole output must be call blocks only.
  Reported as strict_format.

clarify_correct and grounded_correct are rules only: no judge is re-run per checkpoint
(PLAN 3.3), so the labels carry synonyms and required values strict enough for rules. Both also
require a generated output to end with <|im_end|>: one that never ends is not a turn, and its tail
is often an invented user turn whose text would otherwise earn credit (seen on base, E0 2026-10-24).
"""

from collections import Counter
from dataclasses import dataclass, field
import re

from src.tool_calls import ToolCallParseError, canonical, json_decoder, parse_tool_calls, validate_call


OPEN, CLOSE = "<tool_call>", "</tool_call>"
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


@dataclass
class Extraction:
    has_marker: bool                 # any opening or closing tool_call text at all
    calls: list = field(default_factory=list)  # well-formed calls, in output order
    invalid_blocks: int = 0          # broken JSON, wrong shape, unclosed or stray tags
    prose_outside: bool = False      # non-whitespace text outside call blocks


def _strip_stop(text):
    body = text.strip()
    for stop in STOP_TOKENS:
        if body.endswith(stop):
            return body[:-len(stop)].rstrip()
    return body


def extract_tool_calls(text: str) -> Extraction:
    body = _strip_stop(text)
    result = Extraction(has_marker="<tool_call" in body or "</tool_call" in body)
    decoder, position, outside = json_decoder(), 0, []
    while True:
        start = body.find(OPEN, position)
        if start == -1:
            outside.append(body[position:])
            break
        outside.append(body[position:start])
        cursor = start + len(OPEN)
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        try:
            call, end = decoder.raw_decode(body, cursor)
            validate_call(call)
        except (ValueError, ToolCallParseError):
            result.invalid_blocks += 1
            close = body.find(CLOSE, cursor)
            position = close + len(CLOSE) if close != -1 else len(body)
            continue
        while end < len(body) and body[end].isspace():
            end += 1
        if body.startswith(CLOSE, end):
            result.calls.append(call)
            position = end + len(CLOSE)
        else:
            result.invalid_blocks += 1
            position = end
    for segment in outside:
        # Leftover tags outside a block are broken calls too.
        result.invalid_blocks += segment.count(CLOSE) + segment.count("<tool_call")
    result.prose_outside = any(segment.replace(CLOSE, "").strip() for segment in outside)
    return result


def _pair_overlap(predicted, expected):
    if predicted["name"] != expected["name"]:
        return 0
    return len({(key, canonical(value)) for key, value in predicted["arguments"].items()}
               & {(key, canonical(value)) for key, value in expected["arguments"].items()})


def max_weight_assignment(matrix: list) -> int:
    """Hungarian algorithm for rows <= columns: the largest total taking one cell per row, each column
    at most once. Exact for any size, so the score never depends on the order of the calls."""
    rows, columns = len(matrix), len(matrix[0])
    infinity = float("inf")
    u, v, owner, way = [0] * (rows + 1), [0] * (columns + 1), [0] * (columns + 1), [0] * (columns + 1)
    for row in range(1, rows + 1):
        owner[0], free = row, 0
        slack, used = [infinity] * (columns + 1), [False] * (columns + 1)
        while True:
            used[free] = True
            current, delta, nearest = owner[free], infinity, 0
            for column in range(1, columns + 1):
                if not used[column]:
                    reduced = -matrix[current - 1][column - 1] - u[current] - v[column]
                    if reduced < slack[column]:
                        slack[column], way[column] = reduced, free
                    if slack[column] < delta:
                        delta, nearest = slack[column], column
            for column in range(columns + 1):
                if used[column]:
                    u[owner[column]] += delta
                    v[column] -= delta
                else:
                    slack[column] -= delta
            free = nearest
            if owner[free] == 0:
                break
        while free:
            previous = way[free]
            owner[free] = owner[previous]
            free = previous
    return sum(matrix[owner[column] - 1][column - 1] for column in range(1, columns + 1) if owner[column])


def matched_arguments(predicted: list, expected: list) -> int:
    """Correct (argument, value) pairs when each predicted call is paired with at most one expected
    call, choosing the pairing with the most overlap; swapping values between calls then costs."""
    matrix = [[_pair_overlap(p, e) for e in expected] for p in predicted]
    if not matrix or not expected:
        return 0
    if len(matrix) > len(expected):
        matrix = [list(column) for column in zip(*matrix)]
    return max_weight_assignment(matrix)


def score_tool_sample(output: str, *, expected_calls: list, tools: list) -> dict:
    """One tool-group sample. expected_calls == [] marks a No-tool sample."""
    for call in expected_calls:
        validate_call(call)
    extraction = extract_tool_calls(output)
    needs_tool = bool(expected_calls)
    called = extraction.has_marker
    format_valid = called and extraction.invalid_blocks == 0 and bool(extraction.calls)
    predicted = extraction.calls if format_valid else []
    names_match = Counter(call["name"] for call in predicted) == Counter(call["name"] for call in expected_calls)
    if not (needs_tool and format_valid):
        args_f1 = 0.0
    else:
        total = sum(len(call["arguments"]) for call in predicted + expected_calls)
        # Calls without arguments: correct names are all there is to get right.
        args_f1 = 2 * matched_arguments(predicted, expected_calls) / total if total else float(names_match)
    try:
        parse_tool_calls(output)
        strict_format = format_valid
    except ToolCallParseError:
        strict_format = False
    offered = {tool["name"] for tool in tools}
    return {
        "needs_tool": needs_tool,
        "called": called,
        "format_valid": format_valid,
        "strict_format": strict_format,
        "call_exact": needs_tool and format_valid
                      and Counter(map(canonical, predicted)) == Counter(map(canonical, expected_calls)),
        "tool_name_acc": needs_tool and format_valid and names_match,
        "args_f1": args_f1,
        "hallucinated_tool": any(call["name"] not in offered for call in extraction.calls),
        "missed_call": needs_tool and not called,
        "false_call": not needs_tool and called,
    }


def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _ended(stop_reason):
    """stop_reason as classified in src.playground; None marks a hand-written reference answer."""
    return stop_reason is None or stop_reason == "im_end"


def score_clarify(output: str, *, missing_param: str, synonyms=(), stop_reason: str | None = None) -> dict:
    """Missing-info rule layer: ends with <|im_end|>, no call, asks a question, names the missing argument."""
    called = extract_tool_calls(output).has_marker
    words = " ".join(_words(_strip_stop(output)))
    names = [missing_param, *synonyms]
    mentions = any(re.search(rf"\b{' '.join(_words(name))}\b", words) for name in names if _words(name))
    asks = "?" in output
    return {"called": called, "asks": asks, "mentions_param": mentions,
            "clarify_correct": _ended(stop_reason) and not called and asks and mentions}


# A number keeps its sign and decimals ("-31" is not "31", "31.5" is not "31"); a hyphen inside a
# word or date ("2020-05-15") is not a sign. Thousands separators are dropped first.
VALUE_TOKEN = re.compile(r"(?<![a-z0-9.])-?\d+(?:\.\d+)?|[a-z0-9]+")


def _drop_thousands(text):
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text.lower().replace("’", "'"))


def _normalize(text):
    return " ".join(VALUE_TOKEN.findall(_drop_thousands(text)))


# Clause ends: sentence marks, and . , : unless a digit follows ("31.5", "14:05" stay whole).
CLAUSE_END = re.compile(r"[!?;\n]|[.,:](?!\d)")
NEGATION = {"not", "no", "never", "cannot", "nor", "neither", "unable"}
# "not only 31", "no more than 31" state the value instead of denying it.
NOT_NEGATING_NEXT = {"only", "more", "less"}
NEGATION_WINDOW = 3


def _clause_tokens(text):
    """(token, clause index) for every value token; a value may span clauses ("Chicago, IL")."""
    return [(token, index) for index, clause in enumerate(CLAUSE_END.split(_drop_thousands(text)))
            for token in VALUE_TOKEN.findall(clause)]


def _negates(tokens, position):
    word = tokens[position]
    following = tokens[position + 1] if position + 1 < len(tokens) else ""
    if word == "t":  # "isn't", "couldn't" split into "isn" "t"; a lone "t" ("t test") is not one.
        return position > 0 and tokens[position - 1].endswith("n")
    return word in NEGATION and following not in NOT_NEGATING_NEXT


def _mentions(tokens, clauses, value):
    """'stated' when some occurrence is not denied by a negation just before it in the same clause,
    'negated' when every occurrence is, None when the value is absent."""
    target = VALUE_TOKEN.findall(_drop_thousands(str(value)))
    if not target:
        return None
    found = None
    for start in range(len(tokens) - len(target) + 1):
        if tokens[start:start + len(target)] != target:
            continue
        window = [i for i in range(max(0, start - NEGATION_WINDOW), start) if clauses[i] == clauses[start]]
        if not any(_negates(tokens, i) for i in window):
            return "stated"
        found = "negated"
    return found


def score_grounded(output: str, *, required: list, forbidden: list, stop_reason: str | None = None) -> dict:
    """Multi-turn rule layer: ends with <|im_end|>, every required value stated, no forbidden value stated.

    A value preceded in its clause by a negation ("not 31", "isn't available", "couldn't find Hotel A")
    is denied, not stated; "It is 31, not 30" still states 31. Negation after the value ("31 is wrong")
    is not seen by the rules."""
    called = extract_tool_calls(output).has_marker
    pairs = _clause_tokens(_strip_stop(output))
    tokens, clauses = [token for token, _ in pairs], [clause for _, clause in pairs]
    mentions = {str(value): _mentions(tokens, clauses, value) for value in [*required, *forbidden]}
    negated = [value for value in required if mentions[str(value)] == "negated"]
    required_all = all(mentions[str(value)] == "stated" for value in required)
    forbidden_none = not any(mentions[str(value)] == "stated" for value in forbidden)
    return {"called": called, "required_all": required_all, "forbidden_none": forbidden_none,
            "negated_required": negated,
            "grounded_rule": _ended(stop_reason) and not called and required_all and forbidden_none}


NO_FITTING_TOOL = re.compile("|".join([
    r"\bnone of the (?:\w+ ){0,2}(?:tools|functions)\b",
    r"\bno (?:\w+ )?(?:tool|function)s? (?:\w+ )?(?:can|could|that|to|for|available|fits?|matches?|is|are)\b",
    r"\b(?:tools?|functions?)\b[^.!?\n]{0,40}\b(?:can't|cannot|can not|don't|do not|doesn't|does not|aren't|are not|"
    r"isn't|is not|won't|will not)\b",
    r"\b(?:don't|do not|doesn't|does not) (?:have|include|offer|provide|support) (?:a |an |any |the )?"
    r"(?:\w+ ){0,2}(?:tool|function|access|abilit|capabilit)",
    r"\b(?:can't|cannot|can not|unable to|not able to) (?:access|look|retrieve|fetch|browse|check|get|find|search|"
    r"provide (?:real|live|current|up))",
    r"\b(?:no|without) (?:access to|real-time|live) ",
]))


def score_no_fitting(output: str) -> dict:
    """No-tool 3b side metric (PLAN 4.3): no call, and says that no provided tool fits the request."""
    called = extract_tool_calls(output).has_marker
    says = bool(NO_FITTING_TOOL.search(_strip_stop(output).lower().replace("’", "'")))
    return {"called": called, "says_no_tool": says, "no_fitting_said": not called and says}


ROLE_BREAK = re.compile(r"<\|im_start\|>\s*(user|system)\b")


def score_behavior(output: str, *, stop_reason: str) -> dict:
    """Format/behaviour metrics for every sample (stop reasons as classified in src.playground)."""
    words = _strip_stop(output).split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return {
        "im_end_stop": stop_reason == "im_end",
        "eot_stop": stop_reason == "endoftext",
        "truncated": stop_reason == "max_new_tokens",
        "role_break": bool(ROLE_BREAK.search(output)),
        "repetition_rate": 1 - len(set(grams)) / len(grams) if grams else 0.0,
    }


def _mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else None


def aggregate_tool_group(samples: list[dict]) -> dict:
    """Group metrics; need-tool metrics over need-tool samples, decision_bacc over both kinds."""
    need = [sample for sample in samples if sample["needs_tool"]]
    no_tool = [sample for sample in samples if not sample["needs_tool"]]
    metrics = {key: _mean(sample[key] for sample in need)
               for key in ("format_valid", "strict_format", "call_exact", "tool_name_acc", "args_f1", "missed_call")}
    metrics["hallucinated_tool"] = _mean(sample["hallucinated_tool"] for sample in samples)
    metrics["false_call_rate"] = _mean(sample["false_call"] for sample in no_tool)
    metrics["decision_bacc"] = (0.5 * (_mean(s["called"] for s in need) + _mean(not s["called"] for s in no_tool))
                                if need and no_tool else None)
    metrics["n_need_tool"], metrics["n_no_tool"] = len(need), len(no_tool)
    # 3b samples carry score_no_fitting's keys too; the side metric is over those samples only.
    no_fitting = [sample for sample in no_tool if "no_fitting_said" in sample]
    metrics["no_fitting_said"] = _mean(sample["no_fitting_said"] for sample in no_fitting)
    metrics["n_no_fitting"] = len(no_fitting)
    return metrics


def aggregate_grounded(samples: list[dict]) -> dict:
    """Multi-turn group: score_grounded results, each with the item's meta.derivable added.
    The side score keeps the samples whose answer cannot be worked out from the question (PLAN 3.3)."""
    tool_read = [sample for sample in samples if not sample["derivable"]]
    return {"grounded_correct": _mean(sample["grounded_rule"] for sample in samples),
            "grounded_correct_not_derivable": _mean(sample["grounded_rule"] for sample in tool_read),
            "n": len(samples), "n_not_derivable": len(tool_read)}
