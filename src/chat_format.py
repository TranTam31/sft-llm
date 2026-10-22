"""The exploratory ChatML prompt; no special tokens are added."""

import json


TOOL_POLICY = (
    "Call a tool only when the request needs information or an action that one of "
    "the provided tools offers. If you can answer directly from general knowledge, "
    "answer directly. If a required argument is missing, ask the user for it instead "
    "of guessing. If no provided tool fits a request that needs external information, "
    "say so."
)


def system_content(tools: list | None = None) -> str:
    """Use the same fixed tool policy for exploration and SFT data."""
    tool_json = json.dumps(tools if tools is not None else [], ensure_ascii=False)
    reject_control_tokens(tool_json)
    return (
        f"You are a helpful assistant. {TOOL_POLICY}\n"
        f"You have access to the following tools:\n<tools>\n{tool_json}\n</tools>"
    )


def reject_control_tokens(text: str) -> None:
    if not isinstance(text, str):
        raise ValueError("Message content must be text.")
    if any(token in text for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
        raise ValueError("Chat input must not contain role or stop delimiters; use raw mode to inspect them.")


def render_chat(user_text: str, tools: list | None = None) -> str:
    """Render one user turn. Reject delimiter injection in this manual input."""
    reject_control_tokens(user_text)
    return (
        f"<|im_start|>system\n{system_content(tools)}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
