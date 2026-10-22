"""Handwritten demonstrations, separate from future train/dev/test datasets."""

from src.tool_calls import format_tool_calls, format_tool_responses


def inspection_examples():
    weather = {"name": "get_weather", "description": "Get the current weather in a city.",
               "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
    time = {"name": "get_time", "description": "Get the current local time in a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
    call = {"name": "get_weather", "arguments": {"city": "Hanoi"}}
    time_call = {"name": "get_time", "arguments": {"city": "Hanoi"}}
    return [
        {"id": "chat", "tools": [], "turns": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]},
        {"id": "tool_result", "tools": [weather], "turns": [
            {"role": "user", "content": "What is the current weather in Hanoi?"},
            {"role": "assistant", "content": format_tool_calls([call])},
            {"role": "tool", "content": format_tool_responses([{"name": "get_weather", "content": {"temp_c": 31, "condition": "sunny"}}])},
            {"role": "assistant", "content": "It is sunny and 31 C in Hanoi."},
        ]},
        {"id": "parallel", "tools": [weather, time], "turns": [
            {"role": "user", "content": "What are the weather and local time in Hanoi?"},
            {"role": "assistant", "content": format_tool_calls([call, time_call])},
            {"role": "tool", "content": format_tool_responses([
                {"name": "get_weather", "content": {"temp_c": 31}},
                {"name": "get_time", "content": {"time": "14:05"}},
            ])},
            {"role": "assistant", "content": "It is 31 C in Hanoi, and the local time is 14:05."},
        ]},
        {"id": "missing_info", "tools": [weather], "turns": [
            {"role": "user", "content": "What is the current weather?"},
            {"role": "assistant", "content": "Which city would you like the weather for?"},
        ]},
    ]
