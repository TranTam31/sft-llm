"""Render and label SFT examples; only assistant content and its EOS learn."""

from src.chat_format import reject_control_tokens, system_content


# Keep role headers and separator newlines outside the assistant generation span.
# The assistant's im_end must be inside it so learning includes when to stop.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' }}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] + '<|im_end|>' }}{% endgeneration %}"
    "{% else %}{{ message['content'] + '<|im_end|>' }}{% endif %}"
    "{{ '\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


class SequenceTooLongError(ValueError):
    """The caller should count and reject this whole example, never slice it."""


def prepare_messages(turns: list[dict], tools: list | None = None) -> list[dict]:
    if not turns or turns[0].get("role") != "user" or turns[-1].get("role") != "assistant":
        raise ValueError("An SFT conversation must start with user and end with assistant.")
    allowed_next = {"user": {"assistant"}, "assistant": {"user", "tool"}, "tool": {"assistant"}}
    previous = None
    messages = [{"role": "system", "content": system_content(tools)}]
    for turn in turns:
        role, content = turn.get("role"), turn.get("content")
        if role not in allowed_next or (previous is not None and role not in allowed_next[previous]):
            raise ValueError(f"Invalid role transition: {previous} -> {role}")
        reject_control_tokens(content)
        messages.append({"role": role, "content": content})
        previous = role
    return messages


def render_messages(messages: list[dict]) -> str:
    return "".join(f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n" for item in messages)


def configure_tokenizer(tokenizer):
    vocab = tokenizer.get_vocab()
    if not all(token in vocab for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
        raise ValueError("This template requires the existing SmolLM2 control tokens.")
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.padding_side = "right"
    return tokenizer


def make_labels(input_ids: list[int], assistant_mask: list[int]) -> list[int]:
    if len(input_ids) != len(assistant_mask) or any(value not in (0, 1) for value in assistant_mask):
        raise ValueError("Assistant mask must be binary and aligned with input_ids.")
    if not any(assistant_mask):
        raise ValueError("An SFT example must contain supervised assistant tokens.")
    return [token_id if learn else -100 for token_id, learn in zip(input_ids, assistant_mask)]


def encode_sft(tokenizer, turns: list[dict], *, tools: list | None = None, max_seq_len=1024) -> dict:
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive.")
    configure_tokenizer(tokenizer)
    messages = prepare_messages(turns, tools)
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_assistant_tokens_mask=True, truncation=False,
    )
    ids = list(encoded["input_ids"])
    if len(ids) > max_seq_len:
        raise SequenceTooLongError(f"Rendered example has {len(ids)} tokens; limit is {max_seq_len}.")
    labels = make_labels(ids, list(encoded["assistant_masks"]))
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def pad_batch(features: list[dict], *, pad_token_id: int) -> dict:
    """Right-pad lists without masking genuine occurrences of pad_token_id."""
    if not features:
        raise ValueError("Cannot collate an empty batch.")
    for feature in features:
        sizes = [len(feature[key]) for key in ("input_ids", "attention_mask", "labels")]
        if not sizes[0] or len(set(sizes)) != 1:
            raise ValueError("input_ids, attention_mask and labels must align.")
        if any(value not in (0, 1) for value in feature["attention_mask"]):
            raise ValueError("attention_mask must be binary.")
        if any(mask == 0 and label != -100 for mask, label in zip(feature["attention_mask"], feature["labels"])):
            raise ValueError("A masked padding position cannot have a supervised label.")
    width = max(len(feature["input_ids"]) for feature in features)
    return {
        key: [feature[key] + [padding] * (width - len(feature[key])) for feature in features]
        for key, padding in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))
    }


def token_rows(tokenizer, feature: dict) -> list[dict]:
    return [
        {"position": index, "token_id": token_id,
         "token": repr(tokenizer.decode([token_id], skip_special_tokens=False)),
         "label": label, "learn": label != -100}
        for index, (token_id, label) in enumerate(zip(feature["input_ids"], feature["labels"]))
    ]
