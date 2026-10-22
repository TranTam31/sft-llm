"""Inspect a base model without training. Heavy imports happen only on load.

Run: python -m src.playground --interactive
This is an exploratory instrument, not the frozen E0 research evaluation.
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from src.chat_format import render_chat


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"


def validate_request(prompt: str, *, mode="raw", top_k=10, max_new_tokens=64):
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")
    if mode not in ("raw", "chat"):
        raise ValueError("Mode must be raw or chat.")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100.")
    if not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be between 1 and 256.")


def classify_stop(token_ids, *, im_end_id, eot_id, limit):
    """A stop token wins even when it is emitted at the length limit."""
    if token_ids and token_ids[-1] == im_end_id:
        return "im_end"
    if token_ids and token_ids[-1] == eot_id:
        return "endoftext"
    if len(token_ids) == limit:
        return "max_new_tokens"
    raise ValueError("Generation ended without a configured stop token or exhausted budget.")


def append_record(path, record):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


class Playground:
    """One model in memory; raw completion and one-turn ChatML share its weights."""

    def __init__(self, model, tokenizer, metadata):
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.device = next(model.parameters()).device
        vocab = tokenizer.get_vocab()
        for token in ("<|im_end|>", "<|endoftext|>"):
            if token not in vocab:
                raise ValueError(f"Required existing token not found: {token}")
        self.im_end_id = vocab["<|im_end|>"]
        self.eot_id = vocab["<|endoftext|>"]
        self.tokenizer.pad_token = "<|endoftext|>"
        self.model.eval()

    def _prepare(self, prompt, mode, tools=None):
        validate_request(prompt, mode=mode)
        rendered = prompt if mode == "raw" else render_chat(prompt, tools)
        encoded = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        # Reserve room for up to 256 generated tokens within the 1024-token budget.
        if encoded.input_ids.shape[1] > 768:
            raise ValueError("Rendered prompt exceeds 768 tokens. Shorten it; no truncation is applied.")
        return rendered, encoded.to(self.device)

    def next_tokens(self, prompt, *, mode="raw", top_k=10, tools=None):
        """Return probabilities over the full vocabulary, not renormalized top-k."""
        import torch

        validate_request(prompt, mode=mode, top_k=top_k)
        _, encoded = self._prepare(prompt, mode, tools)
        with torch.inference_mode():
            logits = self.model(**encoded, logits_to_keep=1).logits[0, -1].float()
            probabilities = logits.softmax(dim=-1)
            values, indices = probabilities.topk(min(top_k, len(probabilities)))
        return [
            {"token_id": index, "token": self.tokenizer.decode([index], skip_special_tokens=False),
             "probability": probability}
            for index, probability in zip(indices.cpu().tolist(), values.cpu().tolist())
        ]

    def generate(self, prompt, *, mode="raw", max_new_tokens=64, tools=None):
        import torch

        validate_request(prompt, mode=mode, max_new_tokens=max_new_tokens)
        rendered, encoded = self._prepare(prompt, mode, tools)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **encoded, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                eos_token_id=[self.im_end_id, self.eot_id], pad_token_id=self.eot_id,
                use_cache=True,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = perf_counter() - started
        new_ids = output[0, encoded.input_ids.shape[1]:].cpu().tolist()
        stop = classify_stop(new_ids, im_end_id=self.im_end_id, eot_id=self.eot_id, limit=max_new_tokens)
        return {
            **self.metadata, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "exploratory": True, "mode": mode, "prompt": prompt, "rendered_prompt": rendered,
            "prompt_tokens": encoded.input_ids.shape[1], "generated_tokens": len(new_ids),
            "generated_token_ids": new_ids,
            "completion": self.tokenizer.decode(new_ids, skip_special_tokens=False),
            "stop_reason": stop, "im_end_stop": stop == "im_end",
            "eot_stop": stop == "endoftext", "truncated": stop == "max_new_tokens",
            "generation_seconds": seconds, "tokens_per_second": len(new_ids) / seconds,
            "max_new_tokens": max_new_tokens, "decoding": "greedy",
        }


def load_playground(*, device="cpu", revision="main"):
    """Load original weights. Refuse a CPU load when RAM is clearly too tight."""
    import psutil
    import torch
    import transformers
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu, cuda or auto.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use CPU or select a GPU runtime in Colab.")
    available = psutil.virtual_memory().available
    if device == "cpu" and available < 3 * 1024 ** 3:
        raise RuntimeError(
            f"Only {available / 1024 ** 3:.2f} GiB RAM available. "
            "This CPU demo conservatively requires 3 GiB free; close apps or use Colab."
        )
    started = perf_counter()
    torch.manual_seed(0)
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    resolved_revision = model_info(DEFAULT_MODEL, revision=revision).sha
    if not resolved_revision:
        raise RuntimeError("Could not resolve the model revision to a Hub commit.")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, revision=resolved_revision)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL, revision=resolved_revision, dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).to(device)
    metadata = {
        "model_id": DEFAULT_MODEL, "model_revision": resolved_revision,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "python_version": platform.python_version(), "device": device, "dtype": str(dtype),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "load_seconds": perf_counter() - started, "seed": 0,
    }
    return Playground(model, tokenizer, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--mode", choices=["raw", "chat"], default="raw")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default="results/exploratory/outputs.jsonl")
    args = parser.parse_args()
    if not args.interactive and args.prompt is None:
        parser.error("Pass --prompt or --interactive.")
    validate_request(args.prompt or "interactive", mode=args.mode, top_k=args.top_k,
                     max_new_tokens=args.max_new_tokens)
    lab = load_playground(device=args.device, revision=args.revision)
    print(json.dumps(lab.metadata, indent=2))
    mode = args.mode
    if args.interactive:
        print("Enter a prompt. :raw / :chat switch modes; :quit exits. Each prompt is independent.")
    while True:
        try:
            prompt = input(f"\n[{mode}] > ") if args.interactive else args.prompt
            if args.interactive and prompt in (":quit", ":q"):
                break
            if args.interactive and prompt in (":raw", ":chat"):
                mode = prompt[1:]
                continue
            candidates = lab.next_tokens(prompt, mode=mode, top_k=args.top_k)
            for candidate in candidates:
                print(f"{candidate['token_id']:>6} {candidate['probability']:>9.3%} {candidate['token']!r}")
            record = lab.generate(prompt, mode=mode, max_new_tokens=args.max_new_tokens)
            record["next_token_candidates"] = candidates
            append_record(args.output, record)
            print("\nCompletion:", record["completion"])
            print(f"\n{record['stop_reason']} | {record['generated_tokens']} tokens | "
                  f"{record['tokens_per_second']:.2f} tokens/s")
        except (EOFError, KeyboardInterrupt):
            break
        except ValueError as error:
            if not args.interactive:
                raise
            print(error)
        if not args.interactive:
            break


if __name__ == "__main__":
    main()
