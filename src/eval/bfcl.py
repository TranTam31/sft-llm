"""BFCL from PLAN 3.2: the official bfcl-eval generator and AST checker on the simple / multiple / parallel /
irrelevance groups, run at the main checkpoints only.

BFCL has no SmolLM2 handler, so this module registers one that prompts with the project's own tool format
(src.chat_format, the same system prompt as SFT) and decodes <tool_call> blocks with src.tool_calls. The model is
served by vLLM from a pinned snapshot; scoring is BFCL's own.

Run (GPU): python -m src.eval.bfcl --model chat_reference [--output results/bfcl/<name>.json]
--model is a key of configs/models.yaml, a Hub id (then --revision pins it) or a local checkpoint directory.
"""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
from types import SimpleNamespace

from src.chat_format import system_content
from src.eval.benchmarks import resolve
from src.tool_calls import parse_tool_calls


CATEGORIES = ("simple_python", "multiple", "parallel", "irrelevance")
ROOT = Path("results/bfcl")


def format_prompt(messages: list[dict], functions: list[dict]) -> str:
    """One system turn with the SFT tool prompt (plus BFCL's own system text, if any), then the user turns."""
    system = system_content(functions)
    if messages and messages[0]["role"] == "system":
        system += "\n\n" + messages[0]["content"]
        messages = messages[1:]
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
    for message in messages:
        if message["role"] != "user":
            raise ValueError(f"Only single-turn categories are supported, got a {message['role']} turn.")
        prompt += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
    return prompt + "<|im_start|>assistant\n"


def _plain(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def decode_calls(output: str) -> list[dict]:
    """BFCL's AST form, [{name: arguments}]. Plain text gives []; a broken block raises, which BFCL counts as no
    valid call."""
    return [{call["name"]: _plain(call["arguments"])} for call in parse_tool_calls(output)]


def handler_class():
    # Imported here: bfcl_eval is only installed in the BFCL notebook.
    from src.eval.bfcl_handler import SmolLM2Handler
    return SmolLM2Handler


def register(name: str, model_id: str) -> None:
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

    MODEL_CONFIG_MAPPING[name] = ModelConfig(
        model_name=model_id, display_name=f"{name} (project prompt)", url=f"https://huggingface.co/{model_id}",
        org="local", license="apache-2.0", model_handler=handler_class(), is_fc_model=True, underscore_to_dot=False)


def summarise(score_dir: Path, name: str, categories=CATEGORIES) -> dict:
    """The header line BFCL writes at the top of each score file: accuracy and counts."""
    summary = {}
    for category in categories:
        path = next((score_dir / name).rglob(f"BFCL_v4_{category}_score.json"))
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        summary[category] = {"accuracy": header["accuracy"], "correct": header["correct_count"],
                             "n": header["total_count"]}
    return summary


def run(name: str, model_id: str, revision: str | None, *, categories=CATEGORIES, backend: str = "vllm") -> dict:
    # bfcl_eval fixes its result/score folders from this variable at import time.
    os.environ["BFCL_PROJECT_ROOT"] = str(ROOT.resolve())
    ROOT.mkdir(parents=True, exist_ok=True)
    if Path(model_id).is_dir():
        path = model_id
    else:
        from huggingface_hub import snapshot_download
        path = snapshot_download(model_id, revision=revision)
    register(name, model_id)

    from bfcl_eval._llm_response_generation import main as generate
    from bfcl_eval.eval_checker.eval_runner import main as evaluate
    from importlib.metadata import version

    generate(SimpleNamespace(
        model=[name], test_category=list(categories), temperature=0.001, include_input_log=False,
        exclude_state_log=False, num_gpus=1, num_threads=None, gpu_memory_utilization=0.9, backend=backend,
        skip_server_setup=False, local_model_path=path, result_dir=None, allow_overwrite=True, run_ids=False,
        enable_lora=False, max_lora_rank=None, lora_modules=None))
    evaluate([name], list(categories), None, None)
    return {"model": model_id, "revision": revision, "bfcl_eval": version("bfcl-eval"), "backend": backend,
            backend: version(backend), "categories": summarise(ROOT / "score", name, categories)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="chat_reference")
    parser.add_argument("--revision")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--output")
    args = parser.parse_args()
    model_id, revision = resolve(args.model, args.revision, args.config)
    name = Path(model_id).name if Path(model_id).is_dir() else model_id.split("/")[-1]
    result = run(name, model_id, revision, backend=args.backend)
    output = Path(args.output or ROOT / f"{name}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
