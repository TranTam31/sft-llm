"""SFT on the Hugging Face Trainer; serialization and the loss mask come from src.sft_data.

Run: python -m src.train --config configs/e2_pilot.yaml [--max-steps N] [--resume last|hub|PATH]
     [--hub-model-id USER/REPO] [--data-repo USER/REPO]
Checkpoints keep the full training state (optimizer, scheduler, GradScaler, RNG,
data position) so a resumed run continues the same loss curve (PLAN 5.3).

With `data: train_sets` the run reads D_chat/D_tool (data/train/, or --data-repo), lays out the example order of
E1 (`order: seq`) or E2 (`order: mix`) and saves weight-only snapshots for eval where cumulative assistant tokens
reach a log-scale grid (PLAN 3.7), with validation loss on chat, seen-tool and unseen-tool dev conversations at the
same points. `python -m src.eval.run_checkpoints` then scores every snapshot on dev.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from src.sft_data import SequenceTooLongError, configure_tokenizer, encode_sft, pad_batch


def load_config(path, **overrides) -> dict:
    """A run config may name a `base` config it inherits (PLAN 5.1: E1 and E2 share one); its own keys win."""
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "base" in config:
        config = {**load_config(config.pop("base")), **config}
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


def effective_batch(config) -> int:
    return config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]


def make_trainer(model, features, config, *, pad_token_id, stop_after=None, sequential=False, eval_dataset=None,
                 schedule=(), snapshot=None):
    """Build the Trainer; split from run() so tests can use a tiny model and synthetic features.

    stop_after interrupts the run at that step (after its checkpoint) to exercise resume;
    the schedule still follows max_steps. sequential reads `features` in the given order (the run laid it out).
    At every step in `schedule`, snapshot(model, state, counter) is called and eval_dataset (a dict) is evaluated.
    """
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from transformers.trainer_callback import ExportableState

    class AssistantTokenCounter(TrainerCallback, ExportableState):
        """Supervised assistant tokens and all non-padding tokens learned so far; saved in trainer_state.json,
        restored on resume."""

        def __init__(self, assistant_tokens_seen=0, tokens_seen=0):
            self.assistant_tokens_seen = assistant_tokens_seen
            self.tokens_seen = tokens_seen

        def state(self):
            return {"args": {"assistant_tokens_seen": self.assistant_tokens_seen, "tokens_seen": self.tokens_seen},
                    "attributes": {}}

    class Scheduled(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step in steps:
                if snapshot is not None:
                    snapshot(model, state, self.trainer.counter())
                control.should_evaluate = eval_dataset is not None

    class SFTTrainer(Trainer):
        def counter(self):
            # Resume replaces the callback with a restored instance; always look it up.
            return next(callback for callback in self.callback_handler.callbacks
                        if isinstance(callback, AssistantTokenCounter))

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if model.training:
                self.counter().assistant_tokens_seen += int((inputs["labels"] != -100).sum())
                self.counter().tokens_seen += int(inputs["attention_mask"].sum())
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

        def _get_train_sampler(self, train_dataset=None):
            if sequential:
                return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
            return super()._get_train_sampler(train_dataset)

        def log(self, logs, *args, **kwargs):
            logs["assistant_tokens_seen"] = self.counter().assistant_tokens_seen
            logs["tokens_seen"] = self.counter().tokens_seen
            super().log(logs, *args, **kwargs)

    class StopAfter(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            control.should_training_stop = state.global_step >= stop_after

    steps = set(schedule)
    callbacks = [AssistantTokenCounter()] + ([StopAfter()] if stop_after else [])
    hub_model_id = config.get("hub_model_id")
    args = TrainingArguments(
        output_dir=config["output_dir"],
        run_name=config["run_name"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
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
        save_total_limit=config.get("save_total_limit"),
        save_only_model=False,
        restore_callback_states_from_checkpoint=True,
        eval_strategy="no",  # validation runs at the scheduled steps only (Scheduled callback)
        prediction_loss_only=True,
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
    trainer = SFTTrainer(model=model, args=args, train_dataset=features, eval_dataset=eval_dataset,
                         data_collator=SFTCollator(pad_token_id), callbacks=callbacks)
    trainer.add_callback(Scheduled(trainer))
    return trainer


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


def snapshot_name(step: int) -> str:
    return f"step-{step:05d}"


def snapshot_saver(output_dir: Path, tokenizer, config, schedule: list[dict]):
    """Weight-only fp32 snapshots for eval and ΔW (PLAN 5.3; fp16 would round away a small ΔW), each with meta.json,
    uploaded to the run's Hub repo under snapshots/ when hub_model_id is set."""
    targets = {entry["step"]: entry for entry in schedule}

    def save(model, state, counter):
        folder = output_dir / "snapshots" / snapshot_name(state.global_step)
        model.save_pretrained(folder, safe_serialization=True)
        tokenizer.save_pretrained(folder)
        meta = {"run": config["run_name"], "seed": config["seed"], "order": config.get("order"),
                "step": state.global_step, "epoch": state.epoch, "tokens_seen": counter.tokens_seen,
                "assistant_tokens_seen": counter.assistant_tokens_seen,
                "target_tokens": targets.get(state.global_step, {}).get("target_tokens")}
        (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        if config.get("hub_model_id"):
            from huggingface_hub import create_repo, upload_folder

            create_repo(config["hub_model_id"], private=True, exist_ok=True)
            upload_folder(repo_id=config["hub_model_id"], folder_path=str(folder),
                          path_in_repo=f"snapshots/{folder.name}",
                          commit_message=f"{config['run_name']}: snapshot {folder.name}")
    return save


def fetch_train_sets(repo_id: str, data_dir: Path):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, repo_type="dataset", local_dir=str(data_dir.parent),
                      allow_patterns=[f"{data_dir.name}/d_chat.jsonl", f"{data_dir.name}/d_tool.jsonl"])


def prepare_train_sets(tokenizer, config, *, data_repo=None) -> dict:
    """Features in run order, the checkpoint schedule and the validation sets (src.train_data)."""
    from src.eval.run_eval import load_items
    from src.train_data import (build_order, checkpoint_schedule, load_jsonl, step_tokens, take_fraction,
                                val_examples)

    data_dir = Path(config.get("data_dir", "data/train"))
    if data_repo and not (data_dir / "d_tool.jsonl").exists():
        fetch_train_sets(data_repo, data_dir)
    examples = {name: take_fraction(load_jsonl(data_dir / f"d_{name}.jsonl"), config.get("fraction", 1.0))
                for name in ("chat", "tool")}
    encoded, reports = {}, {}
    for name, items in examples.items():
        features, report = build_features(tokenizer, items, max_seq_len=config["max_seq_len"])
        if report["dropped"]:  # D_chat/D_tool are length-checked when built; a drop means the template changed.
            raise ValueError(f"{report['dropped']} examples of d_{name} exceed max_seq_len: {report['dropped_ids'][:5]}")
        encoded[name], reports[name] = features, {key: report[key] for key in ("kept", "dropped")}
    batch = effective_batch(config)
    sequence = build_order(len(encoded["chat"]), len(encoded["tool"]), order=config["order"],
                           epochs=config["epochs"], batch=batch, seed=config["seed"])
    assistant = {(name, index): sum(label != -100 for label in feature["labels"])
                 for name, features in encoded.items() for index, feature in enumerate(features)}
    per_step = step_tokens(sequence, assistant, batch=batch)
    if config.get("max_steps"):
        per_step = per_step[:config["max_steps"]]
    schedule = checkpoint_schedule(per_step, **config["checkpoint_tokens"])
    val = {}
    if config.get("val_split"):
        for name, items in val_examples(load_items(config["val_split"], probe=False)).items():
            val[name], _ = build_features(tokenizer, items, max_seq_len=config["max_seq_len"])
    return {"features": [encoded[name][index] for name, index in sequence],
            "order": [examples[name][index]["id"] for name, index in sequence],
            "max_steps": len(per_step), "schedule": schedule, "val": val or None,
            "report": {"examples": reports, "positions": len(sequence), "effective_batch": batch,
                       "assistant_tokens": sum(per_step), "schedule": schedule,
                       "val": {name: len(features) for name, features in val.items()}}}


def run(config, *, resume=None, stop_after=None, data_repo=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.playground import resolve_model
    from src.sft_examples import inspection_examples

    model_id, revision = resolve_model(config["model"])
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(model_id, revision=revision))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    extra = {}
    if config["data"] == "inspection":
        features, data_report = build_features(tokenizer, inspection_examples(), max_seq_len=config["max_seq_len"])
    elif config["data"] == "train_sets":
        prepared = prepare_train_sets(tokenizer, config, data_repo=data_repo)
        features, data_report = prepared["features"], prepared["report"]
        config = {**config, "max_steps": prepared["max_steps"]}
        (output_dir / "order.json").write_text(json.dumps(prepared["order"]), encoding="utf-8")
        extra = {"sequential": True, "eval_dataset": prepared["val"],
                 "schedule": [entry["step"] for entry in prepared["schedule"]],
                 "snapshot": snapshot_saver(output_dir, tokenizer, config, prepared["schedule"])}
    else:
        raise ValueError(f"Unknown data {config['data']!r}; use inspection or train_sets.")
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float32)
    (output_dir / "config.json").write_text(json.dumps(
        {**config, "model_id": model_id, "model_revision": revision}, indent=2), encoding="utf-8")

    trainer = make_trainer(model, features, config, pad_token_id=tokenizer.pad_token_id, stop_after=stop_after,
                           **extra)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    result = trainer.train(resume_from_checkpoint=resolve_resume(resume, config))
    metrics = {
        **result.metrics, "data": data_report, "wall_seconds": perf_counter() - started,
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
    parser.add_argument("--data-repo", help="Dataset repo holding train/d_chat.jsonl and train/d_tool.jsonl.")
    parser.add_argument("--resume", help="last, hub, or a checkpoint path.")
    parser.add_argument("--stop-after", type=int, help="Interrupt at this step to test resume.")
    args = parser.parse_args()
    config = load_config(args.config, max_steps=args.max_steps, hub_model_id=args.hub_model_id,
                         output_dir=args.output_dir)
    metrics = run(config, resume=args.resume, stop_after=args.stop_after, data_repo=args.data_repo)
    print(json.dumps({key: value for key, value in metrics.items() if key != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
