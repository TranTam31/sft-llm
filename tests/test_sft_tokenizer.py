"""Boundary checks against the real, cached SmolLM2 tokenizer; no weights needed."""

import importlib.util
import unittest


HAS_TOKENIZER = importlib.util.find_spec("transformers") is not None
REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"


@unittest.skipUnless(HAS_TOKENIZER, "Run in Colab after Phase 0 has cached the tokenizer.")
class SFTTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M", revision=REVISION, local_files_only=True)
        except OSError as error:
            raise unittest.SkipTest("SmolLM2 tokenizer is not cached; run the base-loading cell first.") from error

    def test_template_matches_rendering_without_adding_tokens(self):
        from src.sft_data import configure_tokenizer, prepare_messages, render_messages
        original_vocab = self.tokenizer.get_vocab().copy()
        configure_tokenizer(self.tokenizer)
        messages = prepare_messages([{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}])
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        self.assertEqual(text, render_messages(messages))
        self.assertEqual(original_vocab, self.tokenizer.get_vocab())
        self.assertEqual(self.tokenizer.eos_token, "<|im_end|>")
        self.assertEqual(self.tokenizer.pad_token, "<|endoftext|>")

    def test_only_assistant_content_and_its_end_tokens_are_learned(self):
        from src.sft_data import encode_sft
        from src.sft_examples import inspection_examples
        example = inspection_examples()[1]
        feature = encode_sft(self.tokenizer, example["turns"], tools=example["tools"])
        learned_ids = [label for label in feature["labels"] if label != -100]
        learned = self.tokenizer.decode(learned_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        expected = "".join(turn["content"] + "<|im_end|>" for turn in example["turns"] if turn["role"] == "assistant")
        self.assertEqual(learned, expected)
        self.assertEqual(learned_ids.count(self.tokenizer.eos_token_id), 2)
        self.assertEqual(feature["input_ids"].count(self.tokenizer.eos_token_id), 5)
        self.assertNotIn("<|im_start|>", learned)
        self.assertTrue(all(mask == 1 for mask in feature["attention_mask"]))

    def test_every_manual_example_has_assistant_labels(self):
        from src.sft_data import encode_sft
        from src.sft_examples import inspection_examples
        for example in inspection_examples():
            with self.subTest(example=example["id"]):
                feature = encode_sft(self.tokenizer, example["turns"], tools=example["tools"])
                self.assertGreater(sum(label != -100 for label in feature["labels"]), 0)
                self.assertEqual(len(feature["input_ids"]), len(feature["labels"]))

    def test_too_long_example_is_rejected_not_truncated(self):
        from src.sft_data import SequenceTooLongError, encode_sft
        turns = [{"role": "user", "content": "word " * 2000}, {"role": "assistant", "content": "Hi"}]
        with self.assertRaises(SequenceTooLongError):
            encode_sft(self.tokenizer, turns, max_seq_len=1024)


if __name__ == "__main__":
    unittest.main()
