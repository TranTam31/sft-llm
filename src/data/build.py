"""Download the PLAN 4.1 sources, convert them, apply the length rule and write the report.

Run: python -m src.data.build --config configs/data.yaml [--sources xlam glaive smoltalk]
     [--limit N] [--push-to USER/REPO]
Writes data/processed/<source>.jsonl (Git-ignored) and data/REPORT.md (tracked).
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
import statistics

from src.data.convert import ConversionError, convert_glaive, convert_smoltalk, convert_xlam
from src.sft_data import SequenceTooLongError


LENGTH_ALERT = 0.15  # PLAN 2.5: revisit max_seq_len if a source loses more than this to length


@dataclass
class SourceStats:
    rows: int = 0
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    groups: Counter = field(default_factory=Counter)
    lengths: list = field(default_factory=list)
    prose_dropped: int = 0
    system_folded: int = 0

    def too_long_rate(self):
        converted = self.kept + self.dropped["too_long"]
        return self.dropped["too_long"] / converted if converted else 0.0


def length_measurer(tokenizer, max_seq_len):
    """Render with the training template; the count is what PLAN 2.5 limits."""
    from src.sft_data import encode_sft

    def measure(example):
        feature = encode_sft(tokenizer, example["turns"], tools=example["tools"], max_seq_len=max_seq_len)
        example["meta"]["n_tokens"] = len(feature["input_ids"])
        example["meta"]["n_assistant_tokens"] = sum(label != -100 for label in feature["labels"])
    return measure


def process(rows, convert, measure, stats: SourceStats, *, cap=None):
    """rows yields (index, row); stops once cap examples are kept."""
    kept = []
    for index, row in rows:
        if cap is not None and len(kept) >= cap:
            break
        stats.rows += 1
        try:
            example = convert(row, index)
            measure(example)
        except ConversionError as error:
            stats.dropped[error.reason] += 1
            continue
        except SequenceTooLongError:
            stats.dropped["too_long"] += 1
            continue
        kept.append(example)
        stats.kept += 1
        stats.groups[example["group"]] += 1
        stats.lengths.append(example["meta"]["n_tokens"])
        stats.prose_dropped += example["meta"].get("prose_dropped", 0)
        stats.system_folded += example["meta"].get("system_folded", False)
    return kept


def _download(repo, filename, revision):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, revision=revision, repo_type="dataset")


def _limited(rows, limit):
    for position, item in enumerate(rows):
        if limit is not None and position >= limit:
            return
        yield item


def build_xlam(config, measure, limit):
    source = config["xlam"]
    rows = json.loads(Path(_download(source["repo"], source["file"], source["revision"])).read_text(encoding="utf-8"))
    stats = SourceStats()
    examples = process(_limited(enumerate(rows), limit), lambda row, _: convert_xlam(row, seed=config["seed"]),
                       measure, stats)
    return {"xlam": (examples, stats)}


def build_glaive(config, measure, limit):
    source = config["glaive"]
    rows = json.loads(Path(_download(source["repo"], source["file"], source["revision"])).read_text(encoding="utf-8"))
    stats = SourceStats()
    examples = process(_limited(enumerate(rows), limit),
                       lambda row, index: convert_glaive(row, index, seed=config["seed"]), measure, stats)
    return {"glaive": (examples, stats)}


def build_smoltalk(config, measure, limit):
    import pandas
    from huggingface_hub import HfApi

    source = config["smoltalk"]
    files = HfApi().list_repo_files(source["repo"], repo_type="dataset", revision=source["revision"])
    results = {}
    for subset, cap in source["subsets"].items():
        train_file = sorted(path for path in files if path.startswith(f"data/{subset}/train-"))[0]
        frame = pandas.read_parquet(_download(source["repo"], train_file, source["revision"]), columns=["messages"])
        order = list(range(len(frame)))
        random.Random(f"{config['seed']}:{subset}").shuffle(order)
        rows = ((index, {"messages": [dict(message) for message in frame["messages"].iloc[index]]}) for index in order)
        stats = SourceStats()
        examples = process(_limited(rows, limit), lambda row, index, subset=subset: convert_smoltalk(
            row, subset=subset, index=index, seed=config["seed"]), measure, stats, cap=cap)
        results[f"smoltalk-{subset}"] = (examples, stats)
    return results


BUILDERS = {"xlam": build_xlam, "glaive": build_glaive, "smoltalk": build_smoltalk}


def write_outputs(results, config, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, (examples, _) in results.items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for example in examples:
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def render_report(results, hashes, config, *, limit=None):
    lines = ["# Báo cáo dữ liệu", "",
             "Sinh tự động bởi `python -m src.data.build`. Đừng sửa tay; chạy lại để cập nhật.", "",
             f"Seed {config['seed']}, `max_seq_len` {config['max_seq_len']} (PLAN 2.5). "
             + (f"**Chạy thử với `--limit {limit}`, chưa phải dữ liệu đầy đủ.**" if limit else ""), "",
             "## Số mẫu theo nguồn", "",
             "| Nguồn | Dòng đọc | Giữ | Quá dài | Tỉ lệ quá dài | Lỗi chuyển đổi | Nhóm | Độ dài trung vị / p95 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, (_, stats) in results.items():
        errors = {reason: count for reason, count in stats.dropped.items() if reason != "too_long"}
        lengths = sorted(stats.lengths)
        spread = f"{statistics.median(lengths):.0f} / {lengths[int(0.95 * (len(lengths) - 1))]}" if lengths else "–"
        rate = stats.too_long_rate()
        lines.append(f"| {name} | {stats.rows} | {stats.kept} | {stats.dropped['too_long']} | "
                     f"{rate:.1%}{' ⚠' if rate > LENGTH_ALERT else ''} | "
                     f"{', '.join(f'{k}: {v}' for k, v in sorted(errors.items())) or '–'} | "
                     f"{', '.join(f'{k}: {v}' for k, v in sorted(stats.groups.items()))} | {spread} |")
    alerts = [name for name, (_, stats) in results.items() if stats.too_long_rate() > LENGTH_ALERT]
    lines += ["", f"⚠ đánh dấu nguồn mất hơn {LENGTH_ALERT:.0%} số mẫu vì độ dài: xem lại giới hạn trước khi train (PLAN 2.5)."
              + (f" Hiện có: {', '.join(alerts)}." if alerts else " Hiện không có nguồn nào."), ""]
    prose = {name: stats.prose_dropped for name, (_, stats) in results.items() if stats.prose_dropped}
    if prose:
        lines += [f"Câu dẫn trước lời gọi bị bỏ (Glaive): {prose}.", ""]
    folded = {name: stats.system_folded for name, (_, stats) in results.items() if stats.system_folded}
    if folded:
        lines += [f"System prompt riêng được gộp vào lượt user đầu (SmolTalk): {folded}.", ""]
    if config.get("notes"):
        lines += ["## Ghi chú", ""] + [f"- {note}" for note in config["notes"]] + [""]
    excluded = config["smoltalk"].get("excluded", {})
    if excluded:
        lines += ["## Subset SmolTalk không dùng", ""] + [f"- `{name}`: {why}" for name, why in excluded.items()] + [""]
    lines += ["## Revision và hash", "",
              *[f"- `{config[key]['repo']}` @ `{config[key]['revision']}`" for key in ("xlam", "glaive", "smoltalk")],
              "", "| File | SHA-256 |", "|---|---|",
              *[f"| `{name}.jsonl` | `{digest}` |" for name, digest in hashes.items()], ""]
    return "\n".join(lines)


def push(output_dir, repo_id, names):
    """Upload only this run's files: a partial build must not re-send stale files of other sources."""
    from huggingface_hub import create_repo, upload_folder

    create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    upload_folder(folder_path=str(output_dir), repo_id=repo_id, repo_type="dataset",
                  allow_patterns=[f"{name}.jsonl" for name in names],
                  commit_message="Processed SFT data (src.data.build)")


def main():
    import yaml
    from transformers import AutoTokenizer

    from src.playground import resolve_model
    from src.sft_data import configure_tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--sources", nargs="+", choices=sorted(BUILDERS), default=sorted(BUILDERS))
    parser.add_argument("--limit", type=int, help="Read at most N rows per source (dry run).")
    parser.add_argument("--report", default="data/REPORT.md")
    parser.add_argument("--push-to", help="Private HF dataset repo for data/processed.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_id, revision = resolve_model(config["tokenizer_model"])
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(model_id, revision=revision))
    measure = length_measurer(tokenizer, config["max_seq_len"])
    results = {}
    for name in args.sources:
        print(f"[{name}] building…", flush=True)
        built = BUILDERS[name](config, measure, args.limit)
        for key, (_, stats) in built.items():
            print(f"[{key}] kept {stats.kept} of {stats.rows}; dropped {dict(stats.dropped)}", flush=True)
        results.update(built)
    output_dir = Path(config["output_dir"])
    hashes = write_outputs(results, config, output_dir)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(render_report(results, hashes, config, limit=args.limit), encoding="utf-8")
    if args.push_to:
        push(output_dir, args.push_to, hashes)


if __name__ == "__main__":
    main()
