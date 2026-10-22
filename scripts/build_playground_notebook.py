"""Bundle the checked-in Python sources into a self-contained Colab notebook.

Run again after editing src/ or tests/. Notebook cells call src/, rather than
maintaining a second copy of inference logic. No credentials or local results
are included in the bundle.
"""

import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Same pins as requirements.txt; Colab supplies torch.
INFERENCE_PINS = [("transformers", "5.20.2"), ("huggingface-hub", "1.32.0"), ("psutil", "7.2.2"),
                  ("safetensors", "0.8.0"), ("pyyaml", "6.0.3")]
TRAINING_PINS = [("accelerate", "1.15.0"), ("datasets", "5.1.0"), ("peft", "0.21.2"), ("trl", "1.16.0")]


def cell(kind, text):
    result = {"cell_type": kind, "id": hashlib.sha256((kind + text).encode()).hexdigest()[:12],
              "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def source_bundle():
    paths = (sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "configs").glob("*.yaml"))
             + sorted((ROOT / "tests").glob("test_*.py")))
    payload = json.dumps({str(path.relative_to(ROOT)).replace("\\", "/"): path.read_text(encoding="utf-8")
                          for path in paths}, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(payload).decode("ascii"), hashlib.sha256(payload).hexdigest()


def bundle_code(encoded, digest):
    return f'''import base64, hashlib, json, os, sys
from pathlib import Path

payload = base64.b64decode("{encoded}")
assert hashlib.sha256(payload).hexdigest() == "{digest}"
root = Path("/content/sft-base-playground")
root.mkdir(parents=True, exist_ok=True)
for relative, source in json.loads(payload).items():
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
os.chdir(root)
sys.path.insert(0, str(root))
# Các object đã tạo (như lab) vẫn giữ nguyên; lần import tiếp dùng source mới.
for module_name in list(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]
print("Source bundle SHA-256:", "{digest}")
print("Working directory:", root)
'''


def install_code(pins):
    """pip can leave Colab's preinstalled versions in place without failing; check after installing."""
    wanted = ", ".join(f'"{name}": "{version}"' for name, version in pins)
    return f'''%pip install -q {" ".join(f"{name}=={version}" for name, version in pins)}
from importlib.metadata import version
wanted = {{{wanted}}}
installed = {{name: version(name) for name in wanted}}
print(installed)
assert installed == wanted, "Phiên bản chưa khớp requirements.txt; chạy lại cell này."
'''


TESTS_CODE = '''import subprocess
checks = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], capture_output=True, text=True)
print(checks.stdout)
print(checks.stderr)
checks.check_returncode()
'''


def write_notebook(name, cells, digest):
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}, "colab": {"name": name}},
                "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "notebooks" / name
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Built {target} ({len(cells)} cells); source SHA-256 {digest}")


def build():
    encoded, digest = source_bundle()
    cells = [
        cell("markdown", """# Phase 0 — Quan sát base model trước SFT

Chạy từng cell từ trên xuống. Notebook này **không train**, không chạy test set nghiên cứu và không cần token Hugging Face.

Mục tiêu: xem tokenization, xác suất token tiếp theo, raw completion, phản ứng với ChatML và cách chuẩn bị nhãn SFT.
**Token không nhất thiết là một từ**: có thể là một phần từ, dấu cách hoặc dấu câu.

CPU runtime đủ cho lần thử ngắn; GPU là tùy chọn. Mã inference nằm trong `src/`, được đóng gói từ repo khi tạo notebook này.
Kết quả chỉ nằm trong runtime Colab cho đến khi tải xuống. Notebook chưa lưu trên Drive thì chọn **Copy to Drive** nếu muốn giữ lại các thay đổi.
"""),
        cell("markdown", "## 1. Cài thư viện\nCell này dùng torch có sẵn của Colab, chỉ cài các thư viện inference còn lại."),
        cell("code", install_code(INFERENCE_PINS)),
        cell("markdown", "## 2. Nạp mã nguồn của dự án\nBundle bên dưới chỉ gồm `src/*.py`, `configs/*.yaml` và các test offline; không có dữ liệu cá nhân hay thông tin đăng nhập. Sửa logic tại repo rồi chạy lại `python scripts/build_playground_notebook.py` để cập nhật notebook."),
        cell("code", bundle_code(encoded, digest)),
        cell("markdown", "## 3. Kiểm tra mã trước khi tải model\nBao gồm test offline và integration test trên một Llama ngẫu nhiên cực nhỏ. Model nhỏ này chỉ kiểm tra code, không đo năng lực SmolLM2."),
        cell("code", TESTS_CODE),
        cell("markdown", "## 4. Tải base model một lần\nDùng `auto`: có CUDA thì chạy GPU, còn lại CPU. Không cập nhật trọng số. Cell in revision chính xác để lần sau có thể tải lại đúng bản."),
        cell("code", '''import importlib, psutil, torch
import src.playground
importlib.reload(src.playground)  # Nạp lại nếu đã chạy lại cell source bundle.
from src.playground import load_playground, append_record
print("Available RAM (GiB):", round(psutil.virtual_memory().available / 1024**3, 2))
print("CUDA:", torch.cuda.is_available())
lab = load_playground(device="auto")
print(json.dumps(lab.metadata, indent=2))
'''),
        cell("markdown", "## 5. Sửa prompt ở đây\n`raw`: dự đoán phần văn bản nối tiếp. `chat`: bọc prompt bằng ChatML và tool policy trong PLAN. Base chưa SFT có thể không hành xử như chatbot. Mỗi lần thử độc lập, không tự giữ lịch sử hội thoại."),
        cell("code", '''PROMPT = "The capital of France is"
MODE = "raw"  # "raw" hoặc "chat"
MAX_NEW_TOKENS = 32  # Đặt 1 nếu chỉ muốn model sinh đúng một token.
TOOLS = []  # Có thể điền schema khi thử MODE = "chat".

from src.chat_format import render_chat
print("Model receives:\\n", PROMPT if MODE == "raw" else render_chat(PROMPT, TOOLS))
'''),
        cell("markdown", "## 6. Xem input token và top 10 token tiếp theo\nXác suất được tính trên **toàn bộ vocabulary**. Tổng top 10 thường nhỏ hơn 100%; đây không phải xác suất model trả lời đúng."),
        cell("code", '''rendered = PROMPT if MODE == "raw" else render_chat(PROMPT, TOOLS)
ids = lab.tokenizer.encode(rendered, add_special_tokens=False)
print("Input token count:", len(ids))
for token_id in ids[:30]:
    print(token_id, repr(lab.tokenizer.decode([token_id], skip_special_tokens=False)))
print("..." if len(ids) > 30 else "")
candidates = lab.next_tokens(PROMPT, mode=MODE, top_k=10, tools=TOOLS)
print("\\nNext-token candidates:")
for row in candidates:
    print(f"{row['token_id']:>6}  {row['probability']:>9.3%}  {row['token']!r}")
print("Total top-10 probability:", sum(row["probability"] for row in candidates))
'''),
        cell("markdown", "## 7. Sinh tiếp và ghi phép đo\nGreedy chọn token có xác suất cao nhất ở mỗi bước. Giữ special token trong output để quan sát cách dừng. Thời gian sinh không gồm thời gian tải model; tốc độ có gồm bước xử lý prompt nên phụ thuộc độ dài prompt và môi trường."),
        cell("code", '''record = lab.generate(PROMPT, mode=MODE, max_new_tokens=MAX_NEW_TOKENS, tools=TOOLS)
record["next_token_candidates"] = lab.next_tokens(PROMPT, mode=MODE, top_k=10, tools=TOOLS)
append_record("results/exploratory/outputs.jsonl", record)
print("Completion:\\n", record["completion"])
print("\\nMeasurements:")
for key in ("prompt_tokens", "generated_tokens", "stop_reason", "generation_seconds", "tokens_per_second"):
    print(f"{key}: {record[key]}")
'''),
        cell("markdown", """## 8. Thử bằng tay

Sửa cell 5, rồi chạy lại 5–7:

1. `The capital of France is` ở raw mode: quan sát nối tiếp kiến thức quen thuộc.
2. `1, 2, 3, 4,` ở raw mode: quan sát một chuỗi số.
3. `Explain why the sky is blue.` ở chat mode: quan sát base trước khi học hội thoại.
4. Đặt `MAX_NEW_TOKENS = 1`: model chỉ phát ra một token. Top token được hiển thị ở cell 6.
5. Thử cùng prompt nhiều lần: greedy thường lặp lại kết quả trong cùng môi trường.

Các prompt này là ví dụ khám phá, không phải P/dev/test đã đóng băng. Khi muốn quan sát nhiều token liên tiếp, tăng ngân sách sinh; model sẽ tính lại phân phối tiếp theo sau mỗi token.
"""),
        cell("markdown", """## 9. Dữ liệu hội thoại để SFT

Chưa train model. Chọn một ví dụ viết tay: chat, gọi tool rồi trả lời, parallel hoặc hỏi lại khi thiếu thông tin. Kết quả tool được giả lập để xem format; không gọi API thời tiết hay thời gian thật.

Token `input_ids` chứa toàn bộ hội thoại. `labels` chỉ giữ token assistant cần học; các vị trí khác là `-100`. Causal language model sẽ tự shift labels khi tính loss, nên dữ liệu không shift trước.
"""),
        cell("code", '''from copy import deepcopy
from src.sft_data import configure_tokenizer, prepare_messages, render_messages, encode_sft, pad_batch, token_rows
from src.sft_examples import inspection_examples

# Chỉ copy tokenizer, không copy model và không đổi tokenizer của playground.
sft_tokenizer = configure_tokenizer(deepcopy(lab.tokenizer))
examples = inspection_examples()
for index, item in enumerate(examples):
    print(index, item["id"])
EXAMPLE_INDEX = 1  # 0=chat, 1=tool_result, 2=parallel, 3=missing_info
example = examples[EXAMPLE_INDEX]
feature = encode_sft(sft_tokenizer, example["turns"], tools=example["tools"])
print("\\nSelected:", example["id"])
print(render_messages(prepare_messages(example["turns"], example["tools"])))
print("Total tokens:", len(feature["input_ids"]))
print("Assistant tokens to learn:", sum(label != -100 for label in feature["labels"]))
'''),
        cell("markdown", """## 10. Xem loss mask quanh lượt assistant

`LEARN`: token có tham gia loss. `CONTEXT`: token vẫn có trong input, label là `-100`. Mặc định hiển thị vùng bắt đầu của câu trả lời; đổi `START` / `STOP` để xem vùng khác. Bảng cuối liệt kê mọi `<|im_end|>` để kiểm tra chỉ token kết thúc lượt assistant được học.
"""),
        cell("code", '''rows = token_rows(sft_tokenizer, feature)
first_learn = next(row["position"] for row in rows if row["learn"])
START = max(0, first_learn - 8)
STOP = min(len(rows), first_learn + 50)
print("position  label  kind     token")
for row in rows[START:STOP]:
    kind = "LEARN" if row["learn"] else "CONTEXT"
    print(f"{row['position']:>8} {row['label']:>6} {kind:<7} {row['token']}")
print("\\nAll im_end positions:")
for row in rows:
    if row["token_id"] == sft_tokenizer.eos_token_id:
        print(row["position"], "LEARN" if row["learn"] else "CONTEXT")
learned_ids = [label for label in feature["labels"] if label != -100]
print("\\nSupervised content only:\\n", sft_tokenizer.decode(learned_ids, skip_special_tokens=False))
'''),
        cell("markdown", """## 11. Ghép batch và padding

Các mẫu có độ dài khác nhau được pad bên phải bằng `<|endoftext|>` có sẵn. Vị trí pad có attention mask 0 và label `-100`. Việc mask dựa vào vị trí padding, không xóa mọi token có cùng ID. Mẫu vượt 1024 token bị từ chối nguyên mẫu, không cắt bớt.
"""),
        cell("code", '''features = [encode_sft(sft_tokenizer, item["turns"], tools=item["tools"]) for item in examples]
batch = pad_batch(features, pad_token_id=sft_tokenizer.pad_token_id)
for index, item in enumerate(examples):
    print(item["id"], "length=", len(features[index]["input_ids"]),
          "batch_width=", len(batch["input_ids"][index]),
          "supervised=", sum(label != -100 for label in batch["labels"][index]))
    print("  attention tail:", batch["attention_mask"][index][-12:])
    print("  labels tail:   ", batch["labels"][index][-12:])
assert all(label == -100 for masks, labels in zip(batch["attention_mask"], batch["labels"])
           for mask, label in zip(masks, labels) if mask == 0)
print("Padding check: OK")
'''),
        cell("markdown", """## 12. Parser và scorer lời gọi tool

Scorer cấu trúc ban đầu: tên tool, toàn bộ tham số và số lượng lời gọi phải khớp. Parallel được so như multiset: bỏ qua thứ tự nhưng giữ số lần xuất hiện. JSON hỏng bị chấm sai. Đây là hợp đồng chấm của bản triển khai này; chưa phải toàn bộ evaluator BFCL hoặc bộ metric nghiên cứu đã khóa.
"""),
        cell("code", '''from src.tool_calls import format_tool_calls, parse_tool_calls, score_call_exact
expected_calls = parse_tool_calls(examples[2]["turns"][1]["content"])
cases = {
    "correct": format_tool_calls(expected_calls),
    "parallel reversed": format_tool_calls(list(reversed(expected_calls))),
    "missing one call": format_tool_calls(expected_calls[:1]),
    "extra duplicate": format_tool_calls(expected_calls + expected_calls[:1]),
    "invalid JSON": "<tool_call>{bad}</tool_call>",
}
print("Expected calls:", expected_calls)
for label, output in cases.items():
    print(label, "=>", score_call_exact(output, expected_calls))
'''),
        cell("markdown", """## 13. So sánh base và các model Instruct

Cùng một bộ prompt khám phá cho base, `SmolLM2-360M-Instruct` và `SmolLM2-1.7B-Instruct` (revision cố định trong `configs/models.yaml`). Mọi model dùng cùng cấu hình dừng `<|im_end|>` / `<|endoftext|>` và cùng ChatML của dự án; SmolLM2-Instruct cũng được train trên ChatML. Mỗi model được nạp, chạy rồi giải phóng trước model tiếp theo.

Đây chưa phải E0: prompt ít và không đóng băng. Đặt `MODELS` ngắn hơn nếu runtime CPU chậm; bản 1.7B trên CPU tốn khoảng 7 GB RAM.
"""),
        cell("code", '''import gc
from src.playground import COMPARISON_PROMPTS, load_playground, run_prompts

MODELS = ["base", "chat_reference", "tool_reference"]
COMPARE_MAX_NEW_TOKENS = 64
if "lab" in globals():
    del lab  # Giải phóng base đã nạp ở mục 4 trước khi nạp model khác.
gc.collect()
comparison = []
for name in MODELS:
    model_lab = load_playground(name, device="auto")
    print(f"{name}: {model_lab.metadata['model_id']} @ {model_lab.metadata['model_revision'][:10]} "
          f"({model_lab.metadata['device']}, load {model_lab.metadata['load_seconds']:.1f}s)")
    for record in run_prompts(model_lab, COMPARISON_PROMPTS, max_new_tokens=COMPARE_MAX_NEW_TOKENS):
        append_record("results/exploratory/outputs.jsonl", record)
        comparison.append(record)
    del model_lab
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

for prompt in COMPARISON_PROMPTS:
    print(f"\\n=== {prompt['id']} [{prompt['mode']}]: {prompt['prompt']}")
    for record in (r for r in comparison if r["prompt_id"] == prompt["id"]):
        print(f"--- {record['model_name']} | {record['stop_reason']} | {record['generated_tokens']} tokens")
        print(record["completion"])
'''),
        cell("markdown", "## 14. Tải log về máy\nRuntime Colab có thể bị xóa khi ngắt phiên. Chạy cell này sau khi thử để giữ lại prompt, output, token ID, revision và các phép đo. Nếu browser không tải file, sao chép các dòng JSON giữa BEGIN_BASE_LOG và END_BASE_LOG vào file `.jsonl`; hai dòng đánh dấu không thuộc dữ liệu."),
        cell("code", '''print("BEGIN_BASE_LOG\\n" + Path("results/exploratory/outputs.jsonl").read_text(encoding="utf-8") + "END_BASE_LOG")
from google.colab import files
files.download("results/exploratory/outputs.jsonl")
'''),
    ]
    write_notebook("00_base_playground.ipynb", cells, digest)


def build_train_smoke():
    encoded, digest = source_bundle()
    cells = [
        cell("markdown", """# Phase 0 — Chạy thử train SFT

Kiểm tra hạ tầng train trước Phase 1: train vài step trên 4 ví dụ viết tay, lưu checkpoint đầy đủ, đẩy lên HF Hub, rồi resume từ Hub. Đây **không** phải thí nghiệm; loss ở đây không mang ý nghĩa năng lực.

Nên dùng runtime **T4 GPU** (Runtime → Change runtime type) để đo thời gian và bộ nhớ như lúc train thật. Logic train nằm trong `src/train.py`; notebook chỉ gọi nó.
"""),
        cell("markdown", "## 1. Cài thư viện\nDùng torch có sẵn của Colab; cài đúng các phiên bản trong `requirements.txt`."),
        cell("code", install_code(INFERENCE_PINS + TRAINING_PINS)),
        cell("markdown", "## 2. Nạp mã nguồn của dự án\nSửa logic tại repo rồi chạy lại `python scripts/build_playground_notebook.py` để cập nhật notebook."),
        cell("code", bundle_code(encoded, digest)),
        cell("markdown", "## 3. Kiểm tra mã\nGồm test resume trên Llama ngẫu nhiên cực nhỏ: chạy liền 4 step phải cho cùng loss với 2 step, lưu, resume, 2 step."),
        cell("code", TESTS_CODE),
        cell("markdown", """## 4. Đăng nhập HF Hub

Cần secret `HF_TOKEN` (token có quyền ghi) trong 🔑 **Secrets** của Colab, đã bật **Notebook access**. Token không được in ra hay ghi vào repo. Điền `HUB_MODEL_ID` dạng `<username>/<tên repo>`; repo được tạo ở chế độ private ở lần push đầu.
"""),
        cell("code", '''from google.colab import userdata
from huggingface_hub import whoami

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
HUB_MODEL_ID = ""  # ví dụ "your-username/smollm2-360m-sft-smoke"
print("Logged in as:", whoami()["name"])
assert HUB_MODEL_ID, "Điền HUB_MODEL_ID trước khi chạy tiếp."
'''),
        cell("markdown", """## 5. Chạy liền 5 step (mốc so sánh)

Không đẩy lên Hub. Đường loss này dùng để so với lần chạy bị ngắt rồi resume ở mục 6–7.
"""),
        cell("code", '''import json, shutil, subprocess

shutil.rmtree("results/runs", ignore_errors=True)  # Mỗi lần chạy thử bắt đầu sạch.

def train(*extra):
    command = [sys.executable, "-m", "src.train", "--config", "configs/smoke.yaml", *extra]
    completed = subprocess.run(command, capture_output=True, text=True)
    print(completed.stdout[-3000:])
    if completed.returncode:
        print(completed.stderr[-5000:])
    completed.check_returncode()

def losses(output_dir):
    history = json.loads(Path(output_dir, "metrics.json").read_text())["log_history"]
    return {row["step"]: (round(row["loss"], 5), row["assistant_tokens_seen"]) for row in history if "loss" in row}

train("--output-dir", "results/runs/smoke-straight")
straight = losses("results/runs/smoke-straight")
print(straight)
'''),
        cell("markdown", "## 6. Chạy lại, dừng sau step 2 và đẩy checkpoint lên Hub\nCheckpoint ở step 2 được đẩy lên `last-checkpoint/` của repo Hub."),
        cell("code", '''train("--output-dir", "results/runs/smoke-split", "--hub-model-id", HUB_MODEL_ID, "--stop-after", "2")
print(losses("results/runs/smoke-split"))
'''),
        cell("markdown", "## 7. Resume từ Hub và so sánh\nTải `last-checkpoint/` từ Hub, train tiếp tới step 5. Loss ở step 3–5 phải gần bằng lần chạy liền (fp16 trên GPU có thể lệch ở chữ số cuối)."),
        cell("code", '''train("--output-dir", "results/runs/smoke-split", "--hub-model-id", HUB_MODEL_ID, "--resume", "hub")
resumed = losses("results/runs/smoke-split")
for step in sorted(straight):
    print(step, "straight", straight[step], "resumed", resumed.get(step))
metrics = json.loads(Path("results/runs/smoke-split/metrics.json").read_text())
print({key: metrics[key] for key in ("global_step", "assistant_tokens_seen", "train_runtime", "peak_gpu_memory_gib")})
'''),
        cell("markdown", "## 8. Ghi lại môi trường\nIn danh sách phiên bản thư viện để lưu vào `requirements.lock` trong repo."),
        cell("code", '''print(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
'''),
    ]
    write_notebook("01_train_smoke.ipynb", cells, digest)


if __name__ == "__main__":
    build()
    build_train_smoke()
