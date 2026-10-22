"""Training-loop contracts on a tiny random Llama; no pretrained weights are downloaded."""

import importlib.util
import tempfile
import unittest


HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "accelerate"))


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

    def test_assistant_tokens_seen_counts_supervised_labels(self):
        features = synthetic_features()
        per_epoch = sum(label != -100 for feature in features for label in feature["labels"])
        with tempfile.TemporaryDirectory() as folder:
            # 3 steps x batch 2 = exactly one pass over the 6 examples.
            logs = self.train_steps(folder, max_steps=3)
        self.assertEqual(logs[3]["assistant_tokens_seen"], per_epoch)
        self.assertLess(logs[1]["assistant_tokens_seen"], logs[2]["assistant_tokens_seen"])

    def train_steps(self, output_dir, *, max_steps):
        from src.train import make_trainer

        trainer = make_trainer(tiny_model(), synthetic_features(),
                               self.config(output_dir, max_steps=max_steps, save_steps=100), pad_token_id=0)
        trainer.train()
        return {entry["step"]: entry for entry in trainer.state.log_history if "loss" in entry}


if __name__ == "__main__":
    unittest.main()
