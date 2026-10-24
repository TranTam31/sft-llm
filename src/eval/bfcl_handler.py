"""The BFCL handler for SmolLM2 checkpoints (see src.eval.bfcl). Kept at module level: `overrides` cannot find the
parent class of a class defined inside a function."""

from typing import Any

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override
import torch

from src.eval.bfcl import decode_calls, format_prompt


class SmolLM2Handler(QwenFCHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # vLLM serves bfloat16 only from compute capability 8.0; torch reports bf16 as supported on a T4 anyway.
        self.dtype = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16"

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        return decode_calls(result)

    @override
    def decode_execute(self, result, has_tool_call_tag):
        return convert_to_function_call(decode_calls(result))

    @override
    def _format_prompt(self, messages, function):
        return format_prompt(messages, function)

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        text = api_response.choices[0].text
        return {"model_responses": text,
                "model_responses_message_for_chat_history": {"role": "assistant", "content": text},
                "input_token": api_response.usage.prompt_tokens,
                "output_token": api_response.usage.completion_tokens}
