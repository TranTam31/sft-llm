"""Strict tagged JSON parser and multiset scorer for the Phase 0 contract.

This is our initial structural scorer, not a BFCL compatibility implementation.
Arrays are ordered; object keys are unordered; bool is not a JSON number.
"""

from collections import Counter
from decimal import Decimal
import json
import math


class ToolCallParseError(ValueError):
    pass


def canonical(value):
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, (int, float, Decimal)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ToolCallParseError("Non-finite JSON number.")
        number = Decimal(str(value))
        if not number.is_finite():
            raise ToolCallParseError("Non-finite JSON number.")
        return ("number", number)
    if isinstance(value, list):
        return ("array", tuple(canonical(item) for item in value))
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return ("object", tuple(sorted((key, canonical(item)) for key, item in value.items())))
    raise ToolCallParseError("Value is not a JSON-compatible value.")


def validate_call(call):
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise ToolCallParseError("A call must contain exactly name and arguments.")
    if not isinstance(call["name"], str) or not call["name"].strip() or not isinstance(call["arguments"], dict):
        raise ToolCallParseError("name must be non-empty text; arguments must be an object.")
    canonical(call)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ToolCallParseError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ToolCallParseError(f"Invalid JSON constant: {value}")


def json_decoder():
    """Strict JSON: duplicate keys and NaN/Infinity are errors; decimals stay exact."""
    return json.JSONDecoder(object_pairs_hook=_unique_object, parse_float=Decimal, parse_constant=_reject_constant)


def format_tool_calls(calls: list[dict]) -> str:
    for call in calls:
        validate_call(call)
    return "\n".join("<tool_call>\n" + json.dumps(call, ensure_ascii=False, allow_nan=False) + "\n</tool_call>" for call in calls)


def format_tool_responses(responses: list[dict]) -> str:
    for response in responses:
        if not isinstance(response, dict) or set(response) != {"name", "content"}:
            raise ValueError("A tool response must contain name and content.")
        if not isinstance(response["name"], str) or not response["name"].strip():
            raise ValueError("A tool response must have a non-empty name.")
        canonical(response)
    return "\n".join("<tool_response>\n" + json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n</tool_response>" for response in responses)


def parse_tool_calls(text: str) -> list[dict]:
    body = text.strip()
    for stop in ("<|im_end|>", "<|endoftext|>"):
        if body.endswith(stop):
            body = body[:-len(stop)].rstrip()
            break
    if "<tool_call" not in body and "</tool_call" not in body:
        return []
    decoder = json_decoder()
    calls = []
    while body:
        if not body.startswith("<tool_call>"):
            raise ToolCallParseError("Unexpected text outside a tool_call block.")
        payload = body[len("<tool_call>"):].lstrip()
        try:
            call, end = decoder.raw_decode(payload)
        except (ValueError, json.JSONDecodeError) as error:
            raise ToolCallParseError(str(error)) from error
        validate_call(call)
        remainder = payload[end:].lstrip()
        if not remainder.startswith("</tool_call>"):
            raise ToolCallParseError("Missing closing tool_call tag.")
        calls.append(call)
        body = remainder[len("</tool_call>"):].strip()
    return calls


def score_call_exact(output: str, expected: list[dict]) -> bool:
    for call in expected:
        validate_call(call)
    try:
        predicted = parse_tool_calls(output)
    except ToolCallParseError:
        return False
    return Counter(canonical(call) for call in predicted) == Counter(canonical(call) for call in expected)
