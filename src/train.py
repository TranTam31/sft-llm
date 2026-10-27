"""SFT on the Hugging Face Trainer; serialization and the loss mask come from src.sft_data.

Run: python -m src.train --config configs/e2_mix.yaml [--seed S] [--max-steps N] [--resume last|hub|PATH]
     [--hub-model-id USER/REPO] [--data-repo USER/REPO]
Checkpoints keep the full training state (optimizer, scheduler, GradScaler, RNG,
data position) so a resumed run continues the same loss curve (PLAN 5.3). With a hub_model_id each one is pushed
to the run repo (CheckpointPusher: two slots taking turns, resume.json, progress.json) and the repo history squashed,
so superseded checkpoints stop using Hub storage; a run that finishes deletes the slots. `hub_private: false` makes a new run repo public (weights only: eval outputs, which quote
the gated eval sets, never go there).

With `data: train_sets` the run reads D_chat/D_tool (data/train/, or --data-repo), lays out the example order of
E1 (`order: seq`) or E2 (`order: mix`) and saves weight-only snapshots for eval where cumulative assistant tokens
reach a log-scale grid (PLAN 3.7), with validation loss on chat, seen-tool and unseen-tool dev conversations at the
same points. `python -m src.eval.run_checkpoints` then scores every snapshot on dev.

A config with `lora` trains a LoRA adapter on the frozen base (E3, PLAN 5.2); its snapshots are the merged weights,
so they are scored and compared like a full fine-tune's.
"""

import argparse
import copy
import json
from pathlib import Path
import threading
from time import perf_counter

from src.sft_data import SequenceTooLongError, configure_tokenizer, encode_sft, pad_batch


SEEDED_KEYS = ("run_name", "output_dir", "hub_model_id")  # may contain {seed}: one config serves every seed


def load_config(path, **overrides) -> dict:
    """A run config may name a `base` config it inherits (PLAN 5.1: E1 and E2 share one); its own keys win."""
    config = read_config(path)
    config.update({key: value for key, value in overrides.items() if value is not None})
    for key in SEEDED_KEYS:
        if isinstance(config.get(key), str):
            config[key] = config[key].format(seed=config["seed"])
    return config


def read_config(path) -> dict:
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "base" in config:
        config = {**read_config(config.pop("base")), **config}
    return config


def apply_lora(model, lora: dict):
    """Wrap the model in a LoRA adapter (PLAN 5.2): only the adapter trains; embeddings and lm_head stay frozen."""
    from peft import LoraConfig, get_peft_model

    # Gradient checkpointing needs a grad-carrying input when the embedding is frozen.
    model.enable_input_require_grads()
    return get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                            target_modules=list(lora["target_modules"]), bias="none",
                                            task_type="CAUSAL_LM"))


def trainable_parameters(model) -> dict:
    return {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in model.parameters())}


def snapshot_model(model):
    """The weights a snapshot saves: a LoRA model is merged into a copy, so eval and ΔW read a plain model; the
    training model keeps its adapter unmerged."""
    from peft import PeftModel

    if isinstance(model, PeftModel):
        return copy.deepcopy(model).merge_and_unload()
    return model


SLOTS = ("a", "b")  # the run repo keeps the newest checkpoint and the one before it


class CheckpointPusher:
    """Pushes full checkpoints to the run repo in a background thread. Two slots, checkpoint-a/ and checkpoint-b/,
    take turns, so the checkpoint before the newest one is still there if the newest cannot be loaded. The same
    commit writes resume.json (the newest slot and its step) and progress.json (where training is), so neither
    points at a checkpoint that did not finish uploading. The history is then squashed: snapshots stay (they are in
    the tree), superseded checkpoints stop using storage. One commit at a time: call wait() before anything else
    commits to the repo or the local checkpoint folder is rotated away. A failed push is reported and kept in
    `errors`; training goes on and the next checkpoint tries again."""

    def __init__(self, repo_id: str, *, private: bool = True, api=None):
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        self.repo_id, self.private, self.api = repo_id, private, api
        self.thread, self.errors, self.created, self.latest = None, [], False, None

    def ensure_repo(self):
        if not self.created:
            self.api.create_repo(self.repo_id, private=self.private, exist_ok=True)
            if self.api.file_exists(self.repo_id, "resume.json"):  # a resumed run continues the slot rotation
                self.latest = json.loads(Path(self.api.hf_hub_download(self.repo_id, "resume.json")).read_text())["slot"]
            self.created = True

    def push(self, folder, step: int, progress: dict | None = None):
        self.wait()
        self.thread = threading.Thread(target=self._push, args=(Path(folder), step, progress or {}))
        self.thread.start()

    def _push(self, folder: Path, step: int, progress: dict):
        from datetime import datetime, timezone

        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        try:
            self.ensure_repo()
            slot = SLOTS[1] if self.latest == SLOTS[0] else SLOTS[0]
            target = f"checkpoint-{slot}"
            files = {f"{target}/{path.relative_to(folder).as_posix()}": path
                     for path in sorted(folder.rglob("*")) if path.is_file()}
            stale = [path for path in self.api.list_repo_files(self.repo_id)
                     if path.startswith(f"{target}/") and path not in files]
            progress = {**progress, "step": step, "slot": slot,
                        "pushed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            operations = ([CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in files.items()]
                          + [CommitOperationDelete(path_in_repo=path) for path in stale]
                          + [CommitOperationAdd(path_in_repo="resume.json",
                                                path_or_fileobj=json.dumps({"slot": slot, "step": step}).encode()),
                             CommitOperationAdd(path_in_repo="progress.json",
                                                path_or_fileobj=json.dumps(progress, indent=1).encode())])
            self.api.create_commit(self.repo_id, operations=operations, commit_message=f"checkpoint step {step} -> {target}")
            self.latest = slot
            self.api.super_squash_history(repo_id=self.repo_id, commit_message=f"squash after checkpoint step {step}")
        except Exception as error:  # noqa: BLE001 - a lost push must not stop training
            self.errors.append(f"step {step}: {error!r}")
            print(f"[train] checkpoint push failed at step {step}: {error!r}", flush=True)

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def discard(self):
        """A finished run needs no resume state: delete both slots and resume.json, then squash them away.
        progress.json stays as the record of the last checkpoint."""
        from huggingface_hub import CommitOperationDelete

        self.wait()
        doomed = [path for path in self.api.list_repo_files(self.repo_id)
                  if path == "resume.json" or path.startswith(tuple(f"checkpoint-{slot}/" for slot in SLOTS))]
        if doomed:
            self.api.create_commit(self.repo_id, operations=[CommitOperationDelete(path_in_repo=path) for path in doomed],
                                   commit_message="run finished: drop resume state")
        self.api.super_squash_history(repo_id=self.repo_id, commit_message="squash after the run finished")


def checkpoint_problems(folder) -> list[str]:
    """Why a downloaded checkpoint cannot be resumed from, if it cannot: unreadable trainer state, missing or
    unreadable weights, or a torch file (optimizer, scheduler, RNG, scaler) whose zip archive fails its CRC check."""
    import zipfile

    from safetensors import safe_open

    folder, problems = Path(folder), []
    try:
        json.loads((folder / "trainer_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        problems.append(f"trainer_state.json: {error!r}")
    weights = sorted(folder.glob("*.safetensors"))
    if not weights:
        problems.append("no *.safetensors weights")
    for path in weights:
        try:
            with safe_open(str(path), "pt") as stream:
                for key in stream.keys():
                    stream.get_tensor(key)
        except Exception as error:  # noqa: BLE001 - any read failure means the file is unusable
            problems.append(f"{path.name}: {error!r}")
    for path in sorted(folder.glob("*.pt")) + sorted(folder.glob("*.pth")):
        try:
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
            if bad:
                problems.append(f"{path.name}: corrupt member {bad}")
        except (OSError, zipfile.BadZipFile) as error:
            problems.append(f"{path.name}: {error!r}")
    return problems


def fetch_resume_checkpoint(repo_id: str, local_dir, *, api=None) -> str:
    """Download the newest checkpoint slot named by resume.json; if it fails checkpoint_problems, the other slot.
    A repo from before the slots (last-checkpoint/ only) is read as one slot."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    local_dir = Path(local_dir)
    if api.file_exists(repo_id, "resume.json"):
        newest = json.loads(Path(api.hf_hub_download(repo_id, "resume.json", local_dir=str(local_dir))).read_text())["slot"]
        names = [f"checkpoint-{newest}"] + [f"checkpoint-{slot}" for slot in SLOTS if slot != newest]
    else:
        names = ["last-checkpoint"]
    present = {path.split("/")[0] for path in api.list_repo_files(repo_id)}
    for name in names:
        if name not in present:
            continue
        api.snapshot_download(repo_id, allow_patterns=[f"{name}/*"], local_dir=str(local_dir))
        problems = checkpoint_problems(local_dir / name)
        if not problems:
            print(f"[train] resuming from {repo_id}/{name}", flush=True)
            return str(local_dir / name)
        print(f"[train] {name} cannot be resumed from: {problems}", flush=True)
    raise RuntimeError(f"No usable checkpoint in {repo_id} ({names}).")


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
                 schedule=(), snapshot=None, pusher=None):
    """Build the Trainer; split from run() so tests can use a tiny model and synthetic features.

    stop_after interrupts the run at that step (after its checkpoint) to exercise resume;
    the schedule still follows max_steps. sequential reads `features` in the given order (the run laid it out).
    At every step in `schedule`, snapshot(model, state, counter) is called and eval_dataset (a dict) is evaluated.
    A CheckpointPusher, if given, receives every saved checkpoint.
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

    class StepTimer(TrainerCallback):
        """Seconds inside training steps only (forward, backward, optimizer): snapshots, validation and checkpoint
        pushes happen between steps, so they are left out (PLAN 6: train time per 1k samples)."""

        def __init__(self):
            self.seconds, self.steps, self.started = 0.0, 0, None

        def on_step_begin(self, args, state, control, **kwargs):
            self.started = perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if self.started is not None:
                self.seconds += perf_counter() - self.started
                self.steps += 1
            self.started = None

    class PushCheckpoints(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            # The save that follows rotates the previous checkpoint away; its upload must be finished first.
            if control.should_save:
                pusher.wait()

        def on_save(self, args, state, control, **kwargs):
            counter = trainer.counter()
            progress = {"max_steps": state.max_steps, "assistant_tokens_seen": counter.assistant_tokens_seen,
                        "tokens_seen": counter.tokens_seen, "step_seconds_this_process": round(timer.seconds, 1),
                        "last_snapshot_step": max((step for step in steps if step <= state.global_step), default=None)}
            pusher.push(Path(args.output_dir) / f"checkpoint-{state.global_step}", state.global_step, progress)

        def on_train_end(self, args, state, control, **kwargs):
            pusher.wait()

    class StopAfter(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            control.should_training_stop = state.global_step >= stop_after

    steps = set(schedule)
    timer = StepTimer()
    callbacks = ([AssistantTokenCounter(), timer] + ([StopAfter()] if stop_after else [])
                 + ([PushCheckpoints()] if pusher is not None else []))
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
        push_to_hub=False,  # CheckpointPusher and snapshot_saver do the Hub uploads
    )
    model.config.use_cache = False
    trainer = SFTTrainer(model=model, args=args, train_dataset=features, eval_dataset=eval_dataset,
                         data_collator=SFTCollator(pad_token_id), callbacks=callbacks)
    trainer.add_callback(Scheduled(trainer))
    trainer.step_timer = timer
    return trainer


def resolve_resume(resume, config):
    """last: newest local checkpoint; hub: the newest usable checkpoint slot of hub_model_id (fetch_resume_checkpoint),
    downloaded next to output_dir (inside it, the download would be uploaded with the run's files)."""
    if resume in (None, "last"):
        return resume == "last" or None
    if resume == "hub":
        output_dir = Path(config["output_dir"])
        return fetch_resume_checkpoint(config["hub_model_id"], output_dir.parent / f"{output_dir.name}-resume")
    return resume


def snapshot_name(step: int) -> str:
    return f"step-{step:05d}"


def snapshot_saver(output_dir: Path, tokenizer, config, schedule: list[dict], pusher=None):
    """Weight-only fp32 snapshots for eval and ΔW (PLAN 5.3; fp16 would round away a small ΔW), each with meta.json,
    uploaded to the run's Hub repo under snapshots/ when hub_model_id is set."""
    targets = {entry["step"]: entry for entry in schedule}

    def save(model, state, counter):
        folder = output_dir / "snapshots" / snapshot_name(state.global_step)
        weights = snapshot_model(model)
        weights.save_pretrained(folder, safe_serialization=True)
        del weights
        tokenizer.save_pretrained(folder)
        meta = {"run": config["run_name"], "seed": config["seed"], "order": config.get("order"),
                "step": state.global_step, "epoch": state.epoch, "tokens_seen": counter.tokens_seen,
                "assistant_tokens_seen": counter.assistant_tokens_seen,
                "target_tokens": targets.get(state.global_step, {}).get("target_tokens"),
                "boundary": targets.get(state.global_step, {}).get("boundary", False),
                "stage_target_tokens": targets.get(state.global_step, {}).get("stage_target_tokens")}
        (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        if pusher is not None:
            pusher.wait()  # a checkpoint push and its squash finish before the next commit
            pusher.ensure_repo()
            pusher.api.upload_folder(repo_id=pusher.repo_id, folder_path=str(folder),
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
    from src.train_data import (add_stage_snapshots, build_order, checkpoint_schedule, load_jsonl, step_tokens,
                                take_fraction, val_examples)

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
    if config.get("stage_snapshots"):
        schedule = add_stage_snapshots(schedule, per_step, sequence, batch=batch, **config["checkpoint_tokens"])
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
    pusher = (CheckpointPusher(config["hub_model_id"], private=config.get("hub_private", True))
              if config.get("hub_model_id") else None)
    extra = {"pusher": pusher}
    if config["data"] == "inspection":
        features, data_report = build_features(tokenizer, inspection_examples(), max_seq_len=config["max_seq_len"])
    elif config["data"] == "train_sets":
        prepared = prepare_train_sets(tokenizer, config, data_repo=data_repo)
        features, data_report = prepared["features"], prepared["report"]
        config = {**config, "max_steps": prepared["max_steps"]}
        (output_dir / "order.json").write_text(json.dumps(prepared["order"]), encoding="utf-8")
        extra.update(sequential=True, eval_dataset=prepared["val"],
                     schedule=[entry["step"] for entry in prepared["schedule"]],
                     snapshot=snapshot_saver(output_dir, tokenizer, config, prepared["schedule"], pusher))
    else:
        raise ValueError(f"Unknown data {config['data']!r}; use inspection or train_sets.")
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float32)
    if config.get("lora"):
        model = apply_lora(model, config["lora"])
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
        "global_step": trainer.state.global_step, "parameters": trainable_parameters(model),
        "completed": trainer.state.global_step >= config["max_steps"],
        "checkpoint_push_errors": pusher.errors if pusher is not None else [],
        # This process only: after a resume, the steps run before the interruption are not counted.
        "step_seconds": trainer.step_timer.seconds, "timed_steps": trainer.step_timer.steps,
        "assistant_tokens_seen": trainer.state.log_history[-1].get("assistant_tokens_seen"),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None,
        "log_history": trainer.state.log_history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if pusher is not None:
        pusher.wait()
        pusher.ensure_repo()
        pusher.api.upload_folder(repo_id=pusher.repo_id, folder_path=str(output_dir),
                                 allow_patterns=["metrics*.json", "config.json", "order.json"],
                                 commit_message=f"{config['run_name']}: train metrics at step {trainer.state.global_step}")
        if metrics["completed"]:
            pusher.discard()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, help="Overrides the config's seed; fills {seed} in run_name, output_dir "
                        "and hub_model_id.")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--hub-model-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--data-repo", help="Dataset repo holding train/d_chat.jsonl and train/d_tool.jsonl.")
    parser.add_argument("--resume", help="last, hub, or a checkpoint path.")
    parser.add_argument("--stop-after", type=int, help="Interrupt at this step to test resume.")
    args = parser.parse_args()
    config = load_config(args.config, seed=args.seed, max_steps=args.max_steps, hub_model_id=args.hub_model_id,
                         output_dir=args.output_dir)
    metrics = run(config, resume=args.resume, stop_after=args.stop_after, data_repo=args.data_repo)
    print(json.dumps({key: value for key, value in metrics.items() if key != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
