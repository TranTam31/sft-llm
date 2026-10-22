"""SFT on the Hugging Face Trainer; serialization and the loss mask come from src.sft_data.

Run: python -m src.train --config configs/smoke.yaml [--max-steps N] [--resume last|hub|PATH]
Checkpoints keep the full training state (optimizer, scheduler, GradScaler, RNG,
data position) so a resumed run continues the same loss curve (PLAN 5.3).
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from src.sft_data import SequenceTooLongError, configure_tokenizer, encode_sft, pad_batch


def load_config(path, **overrides) -> dict:
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def build_features(tokenizer, examples, *, max_seq_len):
    """Encode whole examples; too-long ones are dropped and counted, never truncated (PLAN 2.5)."""
    features, dropped = [], []
    for example in examples:
        try:
            features.append(encode_sft(tokenizer, example["turns"], tools=example["tools"], max_seq_len=max_seq_len))
        except SequenceTooLongError:
            dropped.append(example.get("id"))
    if not features:
        raise ValueError("Every example exceeded max_seq_len.")
    return features, {"kept": len(features), "dropped": len(dropped), "dropped_ids": dropped}


class SFTCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch

        batch = pad_batch(features, pad_token_id=self.pad_token_id)
        return {key: torch.tensor(value) for key, value in batch.items()}


def make_trainer(model, features, config, *, pad_token_id, stop_after=None):
    """Build the Trainer; split from run() so tests can use a tiny model and synthetic features.

    stop_after interrupts the run at that step (after its checkpoint) to exercise resume;
    the schedule still follows max_steps.
    """
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from transformers.trainer_callback import ExportableState

    class AssistantTokenCounter(TrainerCallback, ExportableState):
        """Supervised assistant tokens learned so far; saved in trainer_state.json, restored on resume."""

        def __init__(self, assistant_tokens_seen=0):
            self.assistant_tokens_seen = assistant_tokens_seen

        def state(self):
            return {"args": {"assistant_tokens_seen": self.assistant_tokens_seen}, "attributes": {}}

    class SFTTrainer(Trainer):
        def counter(self):
            # Resume replaces the callback with a restored instance; always look it up.
            return next(callback for callback in self.callback_handler.callbacks
                        if isinstance(callback, AssistantTokenCounter))

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if model.training:
                self.counter().assistant_tokens_seen += int((inputs["labels"] != -100).sum())
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

        def log(self, logs, *args, **kwargs):
            logs["assistant_tokens_seen"] = self.counter().assistant_tokens_seen
            super().log(logs, *args, **kwargs)

    class StopAfter(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            control.should_training_stop = state.global_step >= stop_after

    callbacks = [AssistantTokenCounter()] + ([StopAfter()] if stop_after else [])
    hub_model_id = config.get("hub_model_id")
    args = TrainingArguments(
        output_dir=config["output_dir"],
        run_name=config["run_name"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_steps=config["warmup_ratio"],  # A float below 1 is a ratio of max_steps.
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        # fp16 AMP keeps fp32 master weights; T4 has no bf16 (PLAN 2.6).
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=config["gradient_checkpointing"],
        logging_steps=config["logging_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_only_model=False,
        restore_callback_states_from_checkpoint=True,
        seed=config["seed"],
        data_seed=config["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=[],
        push_to_hub=bool(hub_model_id),
        hub_model_id=hub_model_id,
        # "checkpoint" pushes the model and a full last-checkpoint/ folder on every save.
        hub_strategy="checkpoint",
        hub_private_repo=True,
    )
    model.config.use_cache = False
    return SFTTrainer(model=model, args=args, train_dataset=features,
                      data_collator=SFTCollator(pad_token_id), callbacks=callbacks)


def resolve_resume(resume, config):
    """last: newest local checkpoint; hub: download last-checkpoint/ from hub_model_id."""
    if resume in (None, "last"):
        return resume == "last" or None
    if resume == "hub":
        from huggingface_hub import snapshot_download

        local = snapshot_download(config["hub_model_id"], allow_patterns=["last-checkpoint/*"],
                                  local_dir=Path(config["output_dir"]) / "hub")
        return str(Path(local) / "last-checkpoint")
    return resume


def run(config, *, resume=None, stop_after=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.playground import resolve_model
    from src.sft_examples import inspection_examples

    if config["data"] != "inspection":
        raise ValueError("Only the handwritten inspection data exists before Phase 1.")
    model_id, revision = resolve_model(config["model"])
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(model_id, revision=revision))
    features, data_report = build_features(tokenizer, inspection_examples(), max_seq_len=config["max_seq_len"])
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float32)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(
        {**config, "model_id": model_id, "model_revision": revision}, indent=2), encoding="utf-8")

    trainer = make_trainer(model, features, config, pad_token_id=tokenizer.pad_token_id, stop_after=stop_after)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    result = trainer.train(resume_from_checkpoint=resolve_resume(resume, config))
    metrics = {
        **result.metrics, **data_report, "wall_seconds": perf_counter() - started,
        "global_step": trainer.state.global_step,
        "assistant_tokens_seen": trainer.state.log_history[-1].get("assistant_tokens_seen"),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None,
        "log_history": trainer.state.log_history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if config.get("hub_model_id"):
        trainer.push_to_hub(commit_message=f"{config['run_name']}: step {trainer.state.global_step}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--hub-model-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", help="last, hub, or a checkpoint path.")
    parser.add_argument("--stop-after", type=int, help="Interrupt at this step to test resume.")
    args = parser.parse_args()
    config = load_config(args.config, max_steps=args.max_steps, hub_model_id=args.hub_model_id,
                         output_dir=args.output_dir)
    metrics = run(config, resume=args.resume, stop_after=args.stop_after)
    print(json.dumps({key: value for key, value in metrics.items() if key != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
