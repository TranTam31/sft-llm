"""Training-loop contracts on a tiny random Llama; no pretrained weights are downloaded."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "accelerate"))
HAS_PEFT = HAS_RUNTIME and importlib.util.find_spec("peft") is not None
LORA = {"r": 4, "alpha": 8, "dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def tiny_model():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))


def synthetic_features():
    # Uneven lengths force padding; the first half of each row is unsupervised context.
    features = []
    for index, length in enumerate((6, 9, 7, 12, 8, 10)):
        ids = [(index * 7 + position) % 31 + 1 for position in range(length)]
        labels = [-100] * (length // 2) + ids[length // 2:]
        features.append({"input_ids": ids, "attention_mask": [1] * length, "labels": labels})
    return features


@unittest.skipUnless(HAS_RUNTIME, "Install torch, transformers and accelerate (requirements.txt).")
class TrainLoopTests(unittest.TestCase):
    def config(self, output_dir, **overrides):
        return {"run_name": "test", "output_dir": output_dir, "max_steps": 4,
                "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
                "learning_rate": 1e-2, "lr_scheduler_type": "cosine", "warmup_ratio": 0.0,
                "weight_decay": 0.0, "max_grad_norm": 1.0, "gradient_checkpointing": False,
                "save_steps": 2, "logging_steps": 1, "seed": 0, "hub_model_id": None, **overrides}

    def train(self, output_dir, *, stop_after=None, resume=None):
        from src.train import make_trainer

        trainer = make_trainer(tiny_model(), synthetic_features(), self.config(output_dir),
                               pad_token_id=0, stop_after=stop_after)
        trainer.train(resume_from_checkpoint=resume)
        return {entry["step"]: entry for entry in trainer.state.log_history if "loss" in entry}

    def test_resumed_run_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as straight, tempfile.TemporaryDirectory() as split:
            reference = self.train(straight)
            self.train(split, stop_after=2)
            resumed = self.train(split, resume=f"{split}/checkpoint-2")
        # The resumed history restores steps 1-2 from the checkpoint and adds 3-4.
        self.assertEqual(sorted(reference), [1, 2, 3, 4])
        self.assertEqual(sorted(resumed), [1, 2, 3, 4])
        for step in (1, 2, 3, 4):
            with self.subTest(step=step):
                self.assertAlmostEqual(resumed[step]["loss"], reference[step]["loss"], places=5)
                self.assertEqual(resumed[step]["assistant_tokens_seen"], reference[step]["assistant_tokens_seen"])

    def test_step_timer_counts_the_steps_of_this_process(self):
        from src.train import make_trainer

        with tempfile.TemporaryDirectory() as folder:
            trainer = make_trainer(tiny_model(), synthetic_features(), self.config(folder), pad_token_id=0,
                                   stop_after=3)
            trainer.train()
        self.assertEqual(trainer.step_timer.steps, 3)
        self.assertGreater(trainer.step_timer.seconds, 0)

    def test_assistant_tokens_seen_counts_supervised_labels(self):
        features = synthetic_features()
        per_epoch = sum(label != -100 for feature in features for label in feature["labels"])
        with tempfile.TemporaryDirectory() as folder:
            # 3 steps x batch 2 = exactly one pass over the 6 examples.
            logs = self.train_steps(folder, max_steps=3)
        self.assertEqual(logs[3]["assistant_tokens_seen"], per_epoch)
        self.assertLess(logs[1]["assistant_tokens_seen"], logs[2]["assistant_tokens_seen"])

    def test_sequential_runs_read_the_given_order_and_act_at_scheduled_steps(self):
        from src.train import make_trainer

        features = synthetic_features()
        seen, snapshots = [], []
        trainer = None

        def snapshot(model, state, counter):
            snapshots.append((state.global_step, counter.assistant_tokens_seen))

        with tempfile.TemporaryDirectory() as folder:
            trainer = make_trainer(tiny_model(), features, self.config(folder, max_steps=3, save_steps=100),
                                   pad_token_id=0, sequential=True, schedule=[1, 3], snapshot=snapshot,
                                   eval_dataset={"chat": features[:2], "tool": features[2:4]})
            original = trainer.compute_loss

            def recording(model, inputs, **kwargs):
                if model.training:
                    seen.append(inputs["input_ids"][:, 0].tolist())
                return original(model, inputs, **kwargs)

            trainer.compute_loss = recording
            trainer.train()
        firsts = [feature["input_ids"][0] for feature in features]
        self.assertEqual(seen, [firsts[0:2], firsts[2:4], firsts[4:6]])
        per_feature = [sum(label != -100 for label in feature["labels"]) for feature in features]
        self.assertEqual(snapshots, [(1, sum(per_feature[:2])), (3, sum(per_feature))])
        # Each validation set logs its own entry, with its loss under its name.
        for name in ("chat", "tool"):
            steps = [entry["step"] for entry in trainer.state.log_history if f"eval_{name}_loss" in entry]
            self.assertEqual(steps, [1, 3])

    def train_steps(self, output_dir, *, max_steps):
        from src.train import make_trainer

        trainer = make_trainer(tiny_model(), synthetic_features(),
                               self.config(output_dir, max_steps=max_steps, save_steps=100), pad_token_id=0)
        trainer.train()
        return {entry["step"]: entry for entry in trainer.state.log_history if "loss" in entry}


class ConfigTests(unittest.TestCase):
    def test_seed_fills_the_run_names_of_a_main_run_config(self):
        from src.train import load_config

        config = load_config("configs/e2_mix.yaml", seed=2, hub_model_id="user/smollm2-360m-e2-mix-s{seed}")
        self.assertEqual((config["seed"], config["run_name"], config["output_dir"], config["hub_model_id"]),
                         (2, "e2_mix_s2", "results/runs/e2_mix_s2", "user/smollm2-360m-e2-mix-s2"))
        self.assertEqual(load_config("configs/e2_mix.yaml")["run_name"], "e2_mix_s0")

    def test_main_runs_share_the_base_settings_and_differ_where_plan_5_1_says(self):
        from src.train import load_config

        e1, e2, r64, r8 = (load_config(f"configs/{name}.yaml")
                           for name in ("e1_seq", "e2_mix", "e3_lora_r64", "e3_lora_r8"))
        shared = ("model", "data", "epochs", "max_seq_len", "per_device_train_batch_size",
                  "gradient_accumulation_steps", "lr_scheduler_type", "warmup_ratio", "checkpoint_tokens", "val_split")
        for config in (e1, r64, r8):
            self.assertEqual({key: config[key] for key in shared}, {key: e2[key] for key in shared})
        self.assertEqual((e1["order"], e2["order"], r64["order"]), ("seq", "mix", "mix"))
        self.assertEqual((e1["learning_rate"], e2["learning_rate"]), (5e-5, 5e-5))
        self.assertTrue(e1["stage_snapshots"])
        self.assertNotIn("stage_snapshots", e2)
        self.assertNotIn("fraction", e2)
        self.assertNotIn("lora", e2)
        for config, rank in ((r64, 64), (r8, 8)):
            self.assertEqual(config["learning_rate"], 2e-4)
            self.assertEqual((config["lora"]["r"], config["lora"]["alpha"], config["lora"]["dropout"]),
                             (rank, 2 * rank, 0.05))
            self.assertEqual(len(config["lora"]["target_modules"]), 7)


class NoTokenizer:
    def save_pretrained(self, folder):
        pass


@unittest.skipUnless(HAS_PEFT, "Install torch, transformers, accelerate and peft (requirements.txt).")
class LoraTests(unittest.TestCase):
    config = TrainLoopTests.config

    def lora_trainer(self, output_dir, overrides=None, **kwargs):
        from src.train import apply_lora, make_trainer

        return make_trainer(apply_lora(tiny_model(), LORA), synthetic_features(),
                            self.config(output_dir, gradient_checkpointing=True, **(overrides or {})),
                            pad_token_id=0, **kwargs)

    def test_only_the_adapter_trains(self):
        import torch

        with tempfile.TemporaryDirectory() as folder:
            trainer = self.lora_trainer(folder, {"save_steps": 100})
            before = {name: value.clone() for name, value in trainer.model.named_parameters()}
            trainer.train()
        changed = {name for name, value in trainer.model.named_parameters() if not torch.equal(value, before[name])}
        self.assertTrue(any("lora_B" in name for name in changed))
        self.assertTrue(all("lora_" in name for name in changed), changed)

    def test_snapshot_saves_merged_weights_and_leaves_the_adapter_unmerged(self):
        import torch
        from transformers import LlamaForCausalLM

        from src.train import snapshot_saver

        inputs = torch.tensor([[1, 2, 3, 4, 5]])
        with tempfile.TemporaryDirectory() as folder:
            save = snapshot_saver(Path(folder), NoTokenizer(), self.config(folder, seed=0, order="mix"),
                                  [{"step": 2, "target_tokens": 10}])
            trainer = self.lora_trainer(folder, {"max_steps": 2, "save_steps": 100}, schedule=[2], snapshot=save)
            trainer.train()
            trainer.model.eval()
            loaded = LlamaForCausalLM.from_pretrained(f"{folder}/snapshots/step-00002").eval()
            with torch.no_grad():
                expected, merged = trainer.model(input_ids=inputs).logits, loaded(input_ids=inputs).logits
            meta = json.loads(Path(f"{folder}/snapshots/step-00002/meta.json").read_text(encoding="utf-8"))
        self.assertTrue(torch.allclose(merged, expected, atol=1e-5))
        self.assertFalse(any("lora" in name for name in loaded.state_dict()))
        self.assertTrue(any("lora_A" in name for name, _ in trainer.model.named_parameters()))
        self.assertEqual((meta["step"], meta["boundary"], meta["stage_target_tokens"]), (2, False, None))

    def test_resumed_lora_run_matches_uninterrupted_run(self):
        def losses(trainer, resume=None):
            trainer.train(resume_from_checkpoint=resume)
            return {entry["step"]: entry["loss"] for entry in trainer.state.log_history if "loss" in entry}

        with tempfile.TemporaryDirectory() as straight, tempfile.TemporaryDirectory() as split:
            reference = losses(self.lora_trainer(straight))
            losses(self.lora_trainer(split, stop_after=2))
            resumed = losses(self.lora_trainer(split), resume=f"{split}/checkpoint-2")
        self.assertEqual(sorted(resumed), [1, 2, 3, 4])
        for step in (1, 2, 3, 4):
            with self.subTest(step=step):
                self.assertAlmostEqual(resumed[step], reference[step], places=5)


if __name__ == "__main__":
    unittest.main()
