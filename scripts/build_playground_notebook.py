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
    paths = (sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "configs").glob("*.yaml"))
             + sorted((ROOT / "tests").glob("test_*.py"))
             + [ROOT / "eval_sets" / "MANIFEST.md", ROOT / "eval_sets" / "p.jsonl"])
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
        cell("markdown", "## 2. Nạp mã nguồn của dự án\nBundle bên dưới chỉ gồm `src/*.py`, `configs/*.yaml`, các test offline, cùng `eval_sets/MANIFEST.md` và bộ P; không có dữ liệu cá nhân hay thông tin đăng nhập. Sửa logic tại repo rồi chạy lại `python scripts/build_playground_notebook.py` để cập nhật notebook."),
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


HF_LOGIN_MD = """## 4. Đăng nhập HF Hub

Cần secret `HF_TOKEN` trong 🔑 **Secrets** của Colab, đã bật **Notebook access**. Chạy cell này **bằng nút ▶ trên giao diện Colab**: secret không đọc được khi cell được chạy qua MCP.
"""


def build_data():
    encoded, digest = source_bundle()
    cells = [
        cell("markdown", """# Phase 1 — Dữ liệu

Tải xLAM-60k, Glaive và các subset SmolTalk theo revision trong `configs/data.yaml`, chuyển về định dạng chung, loại mẫu quá `max_seq_len` (không cắt), rồi ghi `data/processed/*.jsonl` và `data/REPORT.md`. Sau đó tách seen/unseen theo nhóm chức năng và khử trùng lặp (`data/split/`, `data/SPLIT.md`). Dữ liệu được đẩy lên một dataset **private** trên HF Hub.

xLAM bị gated: tài khoản HF phải đồng ý điều khoản trên trang dataset trước. CPU runtime là đủ.
"""),
        cell("markdown", "## 1. Cài thư viện"),
        cell("code", install_code(INFERENCE_PINS + [("datasets", "5.1.0")])),
        cell("markdown", "## 2. Nạp mã nguồn của dự án"),
        cell("code", bundle_code(encoded, digest)),
        cell("markdown", "## 3. Kiểm tra mã"),
        cell("code", TESTS_CODE),
        cell("markdown", HF_LOGIN_MD + "\nĐiền `DATASET_REPO` dạng `<username>/<tên repo>`."),
        cell("code", '''from google.colab import userdata
from huggingface_hub import whoami

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
DATASET_REPO = ""  # ví dụ "your-username/sft-capability-data"
print("Logged in as:", whoami()["name"])
assert DATASET_REPO, "Điền DATASET_REPO trước khi chạy tiếp."
'''),
        cell("markdown", "## 5. Tải, chuyển đổi, lọc độ dài và đẩy lên Hub\nKhoảng 10–15 phút; log tiến độ hiện ngay trong lúc chạy."),
        cell("code", '!python -m src.data.build --config configs/data.yaml --push-to {DATASET_REPO}\n'),
        cell("markdown", "## 6. Báo cáo\nSao chép nội dung này vào `data/REPORT.md` trong repo."),
        cell("code", 'print(Path("data/REPORT.md").read_text(encoding="utf-8"))\n'),
        cell("markdown", """## 7. Phân cụm tool, tách seen/unseen, khử trùng lặp

Embed mọi tool, phân thành nhóm chức năng, chọn khoảng 15% số cụm làm unseen, giữ lại các pool eval rồi bỏ khỏi train mọi mẫu trùng hoặc gần trùng với eval (PLAN 4.2, 4.5). Nếu runtime đã mất `data/processed`, dữ liệu được tải lại từ `DATASET_REPO`. Kết quả đẩy lên thư mục `split/` của dataset."""),
        cell("code", '!python -m src.data.split --config configs/data.yaml --from-hub {DATASET_REPO} --push-to {DATASET_REPO}\n'),
        cell("markdown", "## 8. Báo cáo tách\nSao chép nội dung này vào `data/SPLIT.md` trong repo."),
        cell("code", 'print(Path("data/SPLIT.md").read_text(encoding="utf-8"))\n'),
        cell("markdown", """## 9. Dựng các bộ eval chấm bằng luật

T-seen, T-unseen (bỏ mẫu có tool sát tool seen), T-rename mức tên vô nghĩa và C, chia dev/test; kèm nháp nhãn Missing-info và Multi-turn (PLAN 3.2, 4.2, 4.4). Nếu runtime đã mất `data/split`, dữ liệu được tải lại từ `DATASET_REPO`. Kết quả đẩy lên thư mục `eval_sets/` của dataset."""),
        cell("code", '!python -m src.eval.build_sets --config configs/data.yaml --from-hub {DATASET_REPO} --push-to {DATASET_REPO}\n'),
        cell("markdown", "## 10. Báo cáo bộ eval\nSao chép nội dung này vào `eval_sets/REPORT.md` trong repo."),
        cell("code", 'print(Path("eval_sets/REPORT.md").read_text(encoding="utf-8"))\n'),
    ]
    write_notebook("02_data.ipynb", cells, digest)


EVAL_PINS = [("lm_eval", "0.4.13")]


def build_eval():
    encoded, digest = source_bundle()
    cells = [
        cell("markdown", """# Phase 1 — Benchmark chung
ARC-Easy và HellaSwag qua `lm-evaluation-harness` (PLAN 3.3): 500 câu đầu mỗi bộ, 0-shot, chấm theo log-likelihood. Không cần token HF. GPU nhanh hơn; trên CPU đặt `DEVICE = "cpu"`.
"""),
        cell("markdown", "## 1. Cài thư viện"),
        cell("code", install_code(INFERENCE_PINS + [("datasets", "5.1.0")] + EVAL_PINS)),
        cell("markdown", "## 2. Nạp mã nguồn của dự án"),
        cell("code", bundle_code(encoded, digest)),
        cell("markdown", "## 3. Kiểm tra mã"),
        cell("code", TESTS_CODE),
        cell("markdown", "## 4. Chạy ARC-Easy và HellaSwag\n`MODEL` là khóa trong `configs/models.yaml` hoặc id trên Hub. Kết quả ghi vào `results/benchmarks/`."),
        cell("code", '''import torch
MODEL = "chat_reference"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
!python -m src.eval.benchmarks --model {MODEL} --device {DEVICE} --batch-size 16
'''),
    ]
    write_notebook("03_eval.ipynb", cells, digest)


BFCL_PINS = [("vllm", "0.32.0"), ("bfcl-eval", "2026.3.23")]
# bfcl-eval's own requirements, minus the exact numpy / networkx / filelock / mpmath / faiss pins that conflict with
# vLLM's torch; bfcl-eval itself goes in with --no-deps.
BFCL_DEPS = ("requests tqdm numpy pandas huggingface_hub pydantic>=2.8.2 python-dotenv>=1.0.1 tree_sitter==0.21.3 "
             "tree-sitter-java==0.21.0 tree-sitter-javascript==0.21.4 openai>=1.86.0 mistralai==1.7.0 anthropic>=0.75.0 "
             "cohere==5.18.0 typer>=0.12.5 tabulate>=0.9.0 datamodel-code-generator==0.25.7 google-genai>=1.52.0 "
             "qwen-agent mpmath tenacity>=8.5.0 writer-sdk>=2.1.0 overrides boto3 beautifulsoup4 html2text "
             "rank_bm25==0.2.2 google-search-results sentence-transformers>=2.7.0 faiss-cpu networkx filelock")


def bfcl_install_code():
    wanted = ", ".join(f'"{name}": "{version}"' for name, version in BFCL_PINS)
    deps = " ".join(f"'{dep}'" for dep in BFCL_DEPS.split())
    return f'''%pip install -q vllm=={BFCL_PINS[0][1]} {deps}
%pip install -q --no-deps bfcl-eval=={BFCL_PINS[1][1]}
from importlib.metadata import version
wanted = {{{wanted}}}
installed = {{name: version(name) for name in wanted}}
print(installed, "torch", version("torch"), "transformers", version("transformers"))
assert installed == wanted, "Phiên bản chưa khớp; chạy lại cell này."
'''


def build_bfcl():
    encoded, digest = source_bundle()
    cells = [
        cell("markdown", """# Phase 1 — BFCL
Nhóm simple / multiple / parallel / irrelevance của BFCL (PLAN 3.2), chấm bằng AST checker chính thức của `bfcl-eval`. BFCL không có handler cho SmolLM2, nên `src/eval/bfcl.py` đăng ký một handler dùng đúng prompt tool của SFT. Model chạy qua vLLM: cần runtime GPU. Không cần token HF.
"""),
        cell("markdown", "## 1. Cài thư viện\nvLLM kéo theo torch và transformers của nó, nên notebook này không ghim transformers như các notebook khác. Cài xong phải khởi động lại session (Runtime → Restart session) vì torch đổi bản, rồi chạy tiếp từ mục 2. Colab báo xung đột với vài gói có sẵn (google-colab, cudf…); notebook này không dùng chúng."),
        cell("code", bfcl_install_code()),
        cell("markdown", "## 2. Nạp mã nguồn của dự án"),
        cell("code", bundle_code(encoded, digest)),
        cell("markdown", "## 3. Kiểm tra mã\nChỉ chạy test của phần prompt, parser và BFCL; các test huấn luyện cần đúng bản transformers đã ghim."),
        cell("code", '''import subprocess
for pattern in ("test_bfcl.py", "test_scorers.py"):
    checks = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"], capture_output=True, text=True)
    print(checks.stdout)
    print(checks.stderr)
    checks.check_returncode()
'''),
        cell("markdown", "## 4. Chạy BFCL\n`MODEL` là khóa trong `configs/models.yaml`, id trên Hub hoặc thư mục checkpoint. Kết quả ghi vào `results/bfcl/`."),
        cell("code", '''MODEL = "chat_reference"
!python -m src.eval.bfcl --model {MODEL}
'''),
    ]
    write_notebook("04_bfcl.ipynb", cells, digest)


def build_checkpoint_eval():
    encoded, digest = source_bundle()
    cells = [
        cell("markdown", """# Eval checkpoint trên bộ đã đóng băng
Sinh output greedy trên dev (hoặc test, chỉ sau khi khóa giao thức) và bộ P, chấm bằng luật, ghi `results/<run>/<checkpoint>/` (PLAN 3.6); thêm perplexity và KL so với base trên wikitext-2 (PLAN 3.3). Mặc định chạy E0: base zero-shot, base few-shot và 360M-Instruct bằng template riêng của nó. Nên dùng runtime GPU.
"""),
        cell("markdown", "## 1. Cài thư viện"),
        cell("code", install_code(INFERENCE_PINS)),
        cell("markdown", "## 2. Nạp mã nguồn của dự án"),
        cell("code", bundle_code(encoded, digest)),
        cell("markdown", "## 3. Kiểm tra mã"),
        cell("code", TESTS_CODE),
        cell("markdown", HF_LOGIN_MD + "\nĐiền `DATASET_REPO` dạng `<username>/<tên repo>`: dataset private chứa `eval_sets/`."),
        cell("code", '''from google.colab import userdata
from huggingface_hub import whoami

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
DATASET_REPO = ""  # ví dụ "your-username/sft-capability-data"
print("Logged in as:", whoami()["name"])
assert DATASET_REPO, "Điền DATASET_REPO trước khi chạy tiếp."
'''),
        cell("markdown", "## 5. Tải dev/test và kiểm hash\nDừng ở đây nếu `src.eval.manifest` báo file thiếu hoặc khác."),
        cell("code", '''from src.eval.run_eval import fetch
fetch(DATASET_REPO)
!python -m src.eval.manifest
'''),
        cell("markdown", "## 6. Sinh output và chấm\n`MODELS` gồm (tên checkpoint, model, tùy chọn). Model là khóa trong `configs/models.yaml`, id trên Hub hoặc thư mục checkpoint. Mỗi dòng in thời gian sinh để đo ngân sách eval (PLAN 3.6)."),
        cell("code", '''import torch
RUN, SPLIT = "e0", "dev"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = [("base-zero-shot", "base", ""), ("base-few-shot", "base", "--few-shot"),
          ("instruct-360m", "chat_reference", "--template native")]
for name, model, options in MODELS:
    !python -m src.eval.run_eval --model {model} --run {RUN} --checkpoint {name} --split {SPLIT} --device {DEVICE} {options}
'''),
        cell("markdown", "## 7. Perplexity và KL trên wikitext-2\nKL tính so với base; với chính base thì bỏ qua KL."),
        cell("code", '''for model in ("base", "chat_reference"):
    !python -m src.eval.knowledge --model {model} --device {DEVICE} --output results/{RUN}/knowledge-{model}.json
'''),
    ]
    write_notebook("05_eval.ipynb", cells, digest)


def build_pilot():
    encoded, digest = source_bundle()
    cells = [
        cell("markdown", """# Pilot E2 — train và eval theo checkpoint
Train E2 (D_chat và D_tool trộn ngẫu nhiên) trên 10% mỗi tập, 1 seed (`configs/e2_pilot.yaml`), rồi chấm mọi snapshot trên **dev** (PLAN 6, Giai đoạn 2). Snapshot chỉ gồm trọng số, lưu ở nơi số token assistant cộng dồn chạm 25k, 50k, 100k… và ở step cuối; val loss chat / tool seen / tool unseen tính ở cùng các mốc đó. Checkpoint đầy đủ để resume nằm ở `last-checkpoint/` trên Hub. Dùng runtime **T4 GPU**.
"""),
        cell("markdown", "## 1. Cài thư viện"),
        cell("code", install_code(INFERENCE_PINS + TRAINING_PINS)),
        cell("markdown", "## 2. Nạp mã nguồn của dự án"),
        cell("code", bundle_code(encoded, digest)),
        cell("markdown", "## 3. Kiểm tra mã"),
        cell("code", TESTS_CODE),
        cell("markdown", HF_LOGIN_MD + "\nĐiền `DATASET_REPO` (dataset private chứa `eval_sets/` và `train/`) và `HUB_MODEL_ID` (repo model private của run, tạo ở lần push đầu)."),
        cell("code", '''from google.colab import userdata
from huggingface_hub import whoami

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
DATASET_REPO = ""  # ví dụ "your-username/sft-capability-data"
HUB_MODEL_ID = ""  # ví dụ "your-username/smollm2-360m-e2-pilot"
print("Logged in as:", whoami()["name"])
assert DATASET_REPO and HUB_MODEL_ID, "Điền DATASET_REPO và HUB_MODEL_ID trước khi chạy tiếp."
'''),
        cell("markdown", "## 5. Tải dev/test, D_chat/D_tool và kiểm hash\nDừng nếu `src.eval.manifest` báo file thiếu hoặc khác; hash của D_chat/D_tool phải khớp `data/TRAIN.md`."),
        cell("code", '''import hashlib
from src.eval.run_eval import fetch
from src.train import fetch_train_sets
fetch(DATASET_REPO)
fetch_train_sets(DATASET_REPO, Path("data/train"))
!python -m src.eval.manifest
for name in ("d_chat", "d_tool"):
    print(name, hashlib.sha256(Path(f"data/train/{name}.jsonl").read_bytes()).hexdigest())
'''),
        cell("markdown", """## 6. Train
Bị ngắt giữa chừng thì chạy lại cell này với `RESUME = "hub"`: train tiếp từ `last-checkpoint/` trên Hub, snapshot đã đẩy lên vẫn còn. In ra thời gian train mỗi 1000 mẫu và bộ nhớ GPU đỉnh (PLAN 6).
"""),
        cell("code", '''import json
RUN, CONFIG, RESUME = "e2_pilot", "configs/e2_pilot.yaml", ""
resume = f"--resume {RESUME}" if RESUME else ""
!python -m src.train --config {CONFIG} --hub-model-id {HUB_MODEL_ID} --data-repo {DATASET_REPO} {resume}
metrics = json.loads(Path(f"results/runs/{RUN}/metrics.json").read_text())
positions = metrics["data"]["positions"]
print({"global_step": metrics["global_step"], "assistant_tokens_seen": metrics["assistant_tokens_seen"],
       "train_seconds": round(metrics["train_runtime"]), "seconds_per_1k_samples": round(1000 * metrics["train_runtime"] / positions, 1),
       "peak_gpu_memory_gib": round(metrics["peak_gpu_memory_gib"], 2)})
'''),
        cell("markdown", "## 7. Chấm mọi snapshot trên dev\nSnapshot đã chấm được bỏ qua, nên bị ngắt thì chạy lại cell này."),
        cell("code", '''!python -m src.eval.run_checkpoints --run {RUN} --hub-model-id {HUB_MODEL_ID} --split dev
'''),
        cell("markdown", "## 8. Bảng tóm tắt theo checkpoint\nMetric dev chính và val loss theo số token assistant đã học."),
        cell("code", '''history = json.loads(Path(f"results/runs/{RUN}/metrics.json").read_text())["log_history"]
val = {}
for row in history:
    for key, value in row.items():
        if key.startswith("eval_") and key.endswith("_loss"):
            val.setdefault(row["step"], {})[key[5:-5]] = round(value, 3)
for folder in sorted(Path(f"results/{RUN}").glob("step-*")):
    summary = json.loads((folder / "metrics.json").read_text())
    seen, row = summary["metrics"]["t_seen"], {}
    for name, result in summary["metrics"].items():
        row[f"{name}.stop"] = round(result["behaviour"]["stop_rate"], 2)
        if "tool" in result:
            row[f"{name}.call_exact"] = round(result["tool"]["call_exact"], 2)
            row[f"{name}.decision_bacc"] = round(result["tool"]["decision_bacc"], 2)
    for group, metric in (("missing_info", "clarify_correct"), ("multi_turn", "grounded_correct")):
        row[f"t_seen.{metric}"] = round(seen["groups"].get(group, {}).get(metric) or 0, 2)
    step = summary["snapshot"]["step"]
    print(folder.name, summary["snapshot"]["assistant_tokens_seen"], val.get(step), row)
'''),
        cell("markdown", "## 9. Perplexity và KL của snapshot cuối trên wikitext-2"),
        cell("code", '''final = sorted(Path(f"results/runs/{RUN}/snapshots").glob("step-*"))[-1]
!python -m src.eval.knowledge --model {final} --device cuda --output results/{RUN}/knowledge-{final.name}.json
'''),
        cell("markdown", "## 10. Đẩy kết quả lên Hub\nOutput, metric và log của run lên `results/` của repo model, để tải về máy bằng `huggingface_hub` thay vì in qua notebook."),
        cell("code", '''from huggingface_hub import upload_folder
upload_folder(repo_id=HUB_MODEL_ID, folder_path=f"results/{RUN}", path_in_repo=f"results/{RUN}",
              commit_message=f"{RUN}: dev results")
upload_folder(repo_id=HUB_MODEL_ID, folder_path=f"results/runs/{RUN}", path_in_repo=f"results/runs/{RUN}",
              allow_patterns=["metrics.json", "config.json", "order.json"], commit_message=f"{RUN}: train metrics")
'''),
    ]
    write_notebook("06_pilot.ipynb", cells, digest)


if __name__ == "__main__":
    build()
    build_train_smoke()
    build_data()
    build_eval()
    build_bfcl()
    build_checkpoint_eval()
    build_pilot()
