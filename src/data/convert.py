"""Convert raw rows of each source into the shared example format.

Example: {"id", "source", "group", "tools", "turns", "meta"}. tools are JSON-Schema tool
definitions; turns are prepare_messages() input, with assistant calls rendered by
format_tool_calls and tool results by format_tool_responses. A row that cannot be
converted raises ConversionError with a short reason, which the build step counts.
"""

import json
import random
import re

from src.sft_data import prepare_messages
from src.tool_calls import ToolCallParseError, format_tool_calls, format_tool_responses, validate_call


class ConversionError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _finish(example, *, seed):
    """Shuffle the tool list deterministically (PLAN 2.3) and validate the conversation."""
    random.Random(f"{seed}:{example['id']}").shuffle(example["tools"])
    try:
        prepare_messages(example["turns"], example["tools"])
    except ValueError as error:
        reason = "control_token" if "delimiter" in str(error) else "invalid_turns"
        raise ConversionError(reason) from error
    return example


def _calls_text(calls):
    try:
        return format_tool_calls(calls)
    except (ToolCallParseError, ValueError) as error:
        raise ConversionError("bad_call") from error


# ---- xLAM: query + tools + answers, all JSON strings -------------------------------------

XLAM_TYPES = {"str": "string", "string": "string", "int": "integer", "integer": "integer",
              "float": "number", "number": "number", "bool": "boolean", "boolean": "boolean",
              "list": "array", "tuple": "array", "set": "array", "dict": "object", "object": "object"}


def xlam_parameter(spec: dict) -> tuple[dict, bool]:
    """xLAM writes Python-style types ("List[int]", "str, optional"); map to JSON Schema."""
    raw = str(spec.get("type", "")).strip()
    optional = bool(re.search(r",\s*optional\s*$", raw, re.IGNORECASE))
    base = re.sub(r",\s*optional\s*$", "", raw, flags=re.IGNORECASE).split("[")[0].strip().lower()
    schema = {}
    if base in XLAM_TYPES:
        schema["type"] = XLAM_TYPES[base]
    if spec.get("description"):
        schema["description"] = spec["description"]
    if "default" in spec and spec["default"] not in (None, ""):
        schema["default"] = spec["default"]
    required = bool(spec["required"]) if "required" in spec else not optional
    return schema, required


def xlam_tool(tool: dict) -> dict:
    properties, required = {}, []
    for name, spec in (tool.get("parameters") or {}).items():
        properties[name], is_required = xlam_parameter(spec if isinstance(spec, dict) else {})
        if is_required:
            required.append(name)
    return {"name": tool["name"], "description": tool.get("description", ""),
            "parameters": {"type": "object", "properties": properties, "required": required}}


def _loads(value):
    return json.loads(value) if isinstance(value, str) else value


def convert_xlam(row: dict, *, seed: int) -> dict:
    try:
        query, tools, answers = row["query"], _loads(row["tools"]), _loads(row["answers"])
    except (KeyError, json.JSONDecodeError) as error:
        raise ConversionError("unparsable") from error
    if not answers:
        raise ConversionError("no_answers")
    offered = {tool["name"] for tool in tools}
    if any(call.get("name") not in offered for call in answers):
        raise ConversionError("answer_tool_not_offered")
    group = "parallel" if len(answers) > 1 else ("single" if len(tools) == 1 else "select")
    example = {
        "id": f"xlam:{row.get('id')}", "source": "xlam", "group": group,
        "tools": [xlam_tool(tool) for tool in tools],
        "turns": [{"role": "user", "content": query},
                  {"role": "assistant", "content": _calls_text(answers)}],
        "meta": {"n_calls": len(answers), "n_tools": len(tools)},
    }
    return _finish(example, seed=seed)


# ---- Glaive: "SYSTEM: ... -\n{tool}\n{tool}" + "USER: ... ASSISTANT: ... FUNCTION RESPONSE: ..." --

GLAIVE_TOOLS_MARKER = "Use them if required -"
GLAIVE_TURN = re.compile(r"(?:^|\n+)(USER|ASSISTANT|FUNCTION RESPONSE): ")
GLAIVE_CALL = re.compile(r"""^\s*\{\s*"name"\s*:\s*"(?P<name>[^"]+)"\s*,\s*"arguments"\s*:\s*'(?P<arguments>.*)'\s*\}\s*$""",
                         re.DOTALL)


def glaive_tools(system: str) -> list:
    if GLAIVE_TOOLS_MARKER not in system:
        return []
    text, decoder, tools, position = system.split(GLAIVE_TOOLS_MARKER, 1)[1], json.JSONDecoder(), [], 0
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            return tools
        try:
            tool, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as error:
            raise ConversionError("bad_tool_schema") from error
        tool.setdefault("parameters", {"type": "object", "properties": {}})
        tools.append(tool)


def glaive_call(text: str) -> dict:
    """Glaive quotes the arguments JSON in single quotes, so the call itself is not JSON."""
    try:
        call = json.loads(text)
        if isinstance(call.get("arguments"), str):
            call["arguments"] = json.loads(call["arguments"])
    except (json.JSONDecodeError, AttributeError):
        match = GLAIVE_CALL.match(text)
        if not match:
            raise ConversionError("bad_call")
        try:
            call = {"name": match["name"], "arguments": json.loads(match["arguments"])}
        except json.JSONDecodeError as error:
            raise ConversionError("bad_call") from error
    try:
        validate_call(call)
    except ToolCallParseError as error:
        raise ConversionError("bad_call") from error
    return call


def convert_glaive(row: dict, index: int, *, seed: int) -> dict:
    tools = glaive_tools(row.get("system", ""))
    parts = GLAIVE_TURN.split(row.get("chat", ""))
    turns, last_call, prose_dropped = [], None, 0
    for speaker, raw in zip(parts[1::2], parts[2::2]):
        content = raw.replace("<|endoftext|>", "").strip()
        if speaker == "USER":
            turns.append({"role": "user", "content": content})
        elif speaker == "FUNCTION RESPONSE":
            if last_call is None:
                raise ConversionError("response_without_call")
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = content
            turns.append({"role": "tool", "content": format_tool_responses([{"name": last_call, "content": result}])})
            last_call = None
        elif "<functioncall>" in content:
            prose, call_text = content.split("<functioncall>", 1)
            prose_dropped += bool(prose.strip())
            call = glaive_call(call_text)
            if call["name"] not in {tool["name"] for tool in tools}:
                raise ConversionError("call_tool_not_offered")
            if turns and turns[-1]["role"] == "assistant" and not turns[-1]["content"].startswith("<tool_call>"):
                # Glaive often splits "Let me check." and the call into two assistant turns;
                # a call turn holds only calls (PLAN 2.3), so the lead-in is dropped.
                turns.pop()
                prose_dropped += 1
            turns.append({"role": "assistant", "content": _calls_text([call])})
            last_call = call["name"]
        else:
            turns.append({"role": "assistant", "content": content})
    while turns and turns[-1]["role"] != "assistant":
        turns.pop()
    if not turns:
        raise ConversionError("no_assistant_turn")
    assistant = [turn for turn in turns if turn["role"] == "assistant"]
    calls = [turn["content"].startswith("<tool_call>") for turn in assistant]
    if any(turn["role"] == "tool" for turn in turns):
        group = "multi_turn"
    elif any(calls):
        group = "call_no_response"
    else:
        group = "no_call" if tools else "chat_no_tools"
    # Missing-info candidate: the first reply asks a question and a later reply calls a tool.
    clarify = bool(calls) and not calls[0] and "?" in assistant[0]["content"] and any(calls[1:])
    example = {
        "id": f"glaive:{index}", "source": "glaive", "group": "clarify_candidate" if clarify else group,
        "tools": tools, "turns": turns,
        "meta": {"n_calls": sum(calls), "n_tools": len(tools), "prose_dropped": prose_dropped},
    }
    return _finish(example, seed=seed)


# ---- SmolTalk: messages [{role, content}] ------------------------------------------------

# Common English function words; prose in other Latin-script languages rarely uses them.
ENGLISH_FUNCTION_WORDS = frozenset("""
a about after all also an and any are as at be because been but by can could did do does for from had
has have he her him his how i if in into is it its just like may me more most my no not of on one or
other our out over she should so some such than that the their them then there these they this those
to up us was we were what when where which while who will with would you your
""".split())
CODE_OR_MATH = re.compile(r"```.*?```|`[^`]*`|\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)


def is_english(text: str, *, ascii_threshold=0.9, function_word_share=0.12, min_words=8) -> bool:
    """Mostly ASCII letters AND enough English function words in the prose (code and math removed).
    ASCII alone lets Spanish or French through; short texts are judged on ASCII only."""
    letters = [char for char in text if char.isalpha()]
    if letters and sum(char.isascii() for char in letters) / len(letters) < ascii_threshold:
        return False
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", CODE_OR_MATH.sub(" ", text).lower())
    if len(words) < min_words:
        return True
    return sum(word in ENGLISH_FUNCTION_WORDS for word in words) / len(words) >= function_word_share


def fold_system(messages: list) -> tuple[list, bool]:
    """Move a leading task-specific system prompt into the first user turn; PLAN 2.3 owns the system slot."""
    if len(messages) < 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        return messages, False
    instruction = messages[0]["content"].strip()
    if not instruction:
        return messages[1:], False
    first = {"role": "user", "content": f"{instruction}\n\n{messages[1]['content'].strip()}"}
    return [first, *messages[2:]], True


def convert_smoltalk(row: dict, *, subset: str, index: int, seed: int) -> dict:
    messages, folded = fold_system(row["messages"])
    if any(message["role"] not in ("user", "assistant") for message in messages):
        raise ConversionError("has_system_or_other_role")
    if not is_english(" ".join(message["content"] for message in messages)):
        raise ConversionError("non_english")
    turns = [{"role": message["role"], "content": message["content"].strip()} for message in messages]
    while turns and turns[-1]["role"] != "assistant":
        turns.pop()
    if not turns:
        raise ConversionError("no_assistant_turn")
    example = {"id": f"smoltalk:{subset}:{index}", "source": f"smoltalk:{subset}", "group": "chat",
               "tools": [], "turns": turns, "meta": {"n_turns": len(turns), "system_folded": folded}}
    return _finish(example, seed=seed)
