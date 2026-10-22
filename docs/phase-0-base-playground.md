# Phase 0 — Quan sát base model trước SFT

Hướng dẫn chạy base model, xem xác suất token tiếp theo và so sánh raw completion với ChatML. Phase này chuẩn bị công cụ quan sát trước khi xây dữ liệu và train. Xem [tổng quan dự án](../README.md), [kế hoạch](../PLAN.md) và [tài liệu tham khảo](../REFERENCES.md).

## Mục đích

- Nhập một prompt và xem top token tiếp theo cùng xác suất trên toàn bộ vocabulary.
- Sinh tiếp văn bản bằng greedy decoding; mỗi token mới lại tạo ngữ cảnh cho bước tiếp theo.
- So sánh raw completion với cùng base được bọc trong ChatML.
- Ghi prompt, output, token ID, revision model, phiên bản thư viện, độ dài, kiểu dừng và tốc độ.

Token có thể là một phần từ, dấu cách hoặc dấu câu. Xác suất token không phải độ tin cậy rằng câu trả lời là đúng. Base chưa SFT có thể viết tiếp văn bản thay vì trả lời như chatbot.

## Chạy trên Colab

Notebook trên Colab là bản chụp mã nguồn lúc upload; khi code local thay đổi cần tạo lại và tải lại notebook.

Mở [notebooks/00_base_playground.ipynb](../notebooks/00_base_playground.ipynb) qua **File → Upload notebook** trong Colab. Chạy lần lượt các cell; sửa prompt ở cell được đánh dấu rồi chạy lại phần xem token và generate. CPU runtime đủ cho lần thử ngắn; GPU là tùy chọn.

Notebook chứa bản đóng gói `src/` và các test từ repo. Logic inference được duy trì tại `src/`; sau khi sửa code, tạo lại notebook bằng:

```powershell
python scripts/build_playground_notebook.py
```

Chưa cần tài khoản/token Hugging Face cho model công khai này. Kết quả trong runtime có thể mất khi ngắt phiên: tải file JSONL bằng cell cuối. Nếu browser không tải được file, cell cũng in các dòng JSON để sao chép (bỏ hai dòng đánh dấu). Chọn **Copy to Drive** nếu notebook đang ở scratchpad và cần lưu lại thay đổi.

## Chạy local trên Windows

Các bước dưới đây dùng PowerShell và Python. Thông tin các môi trường đã chạy thực tế nằm trong [EXPERIMENTS.md](../EXPERIMENTS.md).

Chỉ trọng số khoảng 360 triệu tham số ở fp32 đã cần xấp xỉ 1,35 GiB; thư viện, buffer và quá trình sinh còn cần thêm RAM. Demo dùng ngưỡng thận trọng **3 GiB RAM khả dụng** trước khi nạp model lên CPU; đây là ngưỡng bảo vệ, không phải kết quả benchmark bộ nhớ. Bước training đầy đủ cần nhiều bộ nhớ hơn và sẽ ưu tiên Colab.

Chạy các lệnh sau từ thư mục gốc của repo để tạo môi trường riêng và cài bản torch CPU:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Chạy một lần:

```powershell
.\.venv\Scripts\python -m src.playground --prompt "The capital of France is" --max-new-tokens 32
```

Hoặc nhập prompt liên tục, giữ model trong RAM:

```powershell
.\.venv\Scripts\python -m src.playground --interactive
```

Trong chế độ tương tác, `:raw` và `:chat` đổi cách bọc prompt, `:quit` thoát. Mỗi prompt độc lập, không có lịch sử hội thoại tự động. Dùng `--max-new-tokens 1` để chỉ sinh một token; dùng `--revision COMMIT_HASH` để chạy lại đúng bản model đã ghi trong log.

## Các phép đo

Log mặc định: `results/exploratory/outputs.jsonl`.

- `prompt_tokens`, `generated_tokens`: độ dài input và phần mới sinh.
- `stop_reason`: `im_end`, `endoftext` hoặc hết `max_new_tokens`. Hai token dừng được cấu hình thống nhất theo PLAN.
- `generation_seconds`, `tokens_per_second`: thời gian và tốc độ sinh, gồm bước xử lý prompt, không gồm nạp model hoặc tính bảng top token trước đó.
- `next_token_candidates`: token ID, biểu diễn text và xác suất tại vị trí ngay sau prompt.

CPU dùng fp32, CUDA dùng fp16 cho suy luận; khi so kết quả cần đọc cả `device` và `dtype`. Model dùng trọng số gốc, không quantize, không thêm token. Prompt render quá 768 token bị từ chối để dành tối đa 256 token sinh trong ngân sách 1024.

Đây là log khám phá từng lần chạy. Chưa có tỷ lệ tổng hợp, `call_exact`, bộ P/dev/test đóng băng, training, loss mask hay checkpoint/resume. Không dùng các lần thử này làm kết luận nghiên cứu E0. Phase 0 là giai đoạn triển khai; E0 là mã thí nghiệm baseline trong PLAN.

## Kiểm tra code

Các test thuần Python chạy được mà không tải model:

```powershell
python -m unittest discover -s tests -v
```

Nếu đã cài torch/transformers, cùng lệnh đó chạy thêm integration test trên Llama ngẫu nhiên cực nhỏ. Các test này kiểm tra pipeline, không đánh giá năng lực của SmolLM2. Notebook chạy chúng trước khi nạp base model.

## Tài liệu API

- [Model config chính thức](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json).
- [Transformers: cấu hình generate](https://huggingface.co/docs/transformers/main_classes/text_generation).
- [Transformers: nạp model](https://huggingface.co/docs/transformers/main_classes/model).

## Tiếp tục Phase 0

Các mục 9–12 của notebook minh họa dữ liệu SFT, loss mask, padding và scorer cấu trúc. Xem [hướng dẫn format dữ liệu và loss mask](phase-0-sft-format.md) để hiểu token nào sẽ được học trước khi bắt đầu train. Mục tải log nằm ở cuối notebook, sau các phần kiểm tra dữ liệu.
