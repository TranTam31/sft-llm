# Nhật ký triển khai và thí nghiệm

## 2026-10-22 — Phase 0: khám phá base, chưa train

**Loại hoạt động:** khám phá và kiểm tra hạ tầng. Không phải E0 test, không sử dụng P/dev/test nghiên cứu và không khóa thêm giả thuyết.

**Mục tiêu:** quan sát phân phối token tiếp theo, raw completion, phản ứng của base với ChatML và các kiểu kết thúc generation.

**Mã:** `src/playground.py`, `src/chat_format.py`; notebook được tạo từ cùng mã qua `scripts/build_playground_notebook.py`.

**Notebook:** `notebooks/00_base_playground.ipynb`, tải lên Colab của người dùng.

**Thiết lập:** base `HuggingFaceTB/SmolLM2-360M`; greedy; tối đa 256 token sinh; dừng ở `<|im_end|>` hoặc `<|endoftext|>` và ghi riêng nguyên nhân. Không quantize, không thêm token. Prompt khám phá độc lập, không giữ lịch sử.

**Môi trường local:** Windows, i5-1240P, RAM 8 GB, Intel Iris Xe, Python 3.12.10. Khi kiểm tra chỉ khoảng 0,8 GiB RAM khả dụng; chưa cài thư viện model vào Python global và chưa nạp model local. Local chạy các test thuần Python; integration test có thư viện model được thực hiện trên Colab.

**Môi trường Colab:** runtime CPU có khoảng 12,7 GB RAM. Thao tác bằng browser trong Codex. Colab MCP đã được phát hiện nhưng kết nối chưa thành công; lần này dùng giao diện notebook.

**Điều chỉnh hạ tầng:** bộ phiên bản Transformers 4.57.1 / safetensors 0.6.2 gây xung đột với các gói có sẵn của Colab. Chuyển sang Transformers 5.20.2, huggingface-hub 1.32.0, safetensors 0.8.0 và psutil 7.2.2 sau khi kiểm tra metadata phiên bản trên PyPI. Notebook và requirements dùng cùng bộ phiên bản này.

**Log mỗi lần sinh:** `results/exploratory/outputs.jsonl`, gồm revision model, phiên bản thư viện, device/dtype, prompt, output, token ID, ứng viên token tiếp theo, thời gian và kiểu dừng. Kết quả trong runtime cần được tải về để giữ lại.

### Kết quả kiểm tra đã chạy

- Local: 7 test thuần Python qua; 3 integration test bỏ qua vì chưa cài runtime model.
- Colab: cả 10 test qua, gồm 3 integration test dùng Llama ngẫu nhiên cực nhỏ.
- Base được nạp và generate thật trên CPU fp32: 361.821.120 tham số, model revision `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`, torch `2.11.0+cpu`, Transformers `5.20.2`, Python `3.13.16`.
- Thời gian nạp trong lần chạy này khoảng 18,9 giây, có tải trọng số qua mạng; không dùng con số này làm benchmark tốc độ nạp chuẩn.
- Raw prompt `The capital of France is`: top token ` Paris`, xác suất khoảng 20,3%; sinh 32 token trong khoảng 6,50 giây (4,93 token/giây), kết thúc vì hết ngân sách sinh.
- Chat prompt `What is the capital of France?`: bọc bằng ChatML, input 110 token; output lặp lại câu hỏi rồi sinh chuỗi đánh số. Sinh 32 token trong khoảng 7,56 giây (4,23 token/giây), kết thúc vì hết ngân sách sinh.
- Cả hai output cùng thông tin đo đã được lưu về `results/exploratory/outputs.jsonl` trong workspace. Browser không phát sự kiện download như dự kiến; dùng phần log hiển thị trong notebook để bảo toàn file local.
- Source bundle đã chạy: SHA-256 `1899c59af5dd92040ed942f2cb6bd51302d4af07776500ef7d5d22a37162b446`.

Đây là hai ví dụ khám phá với prompt khác nhau, không phải phép so sánh kiểm soát raw/chat và không đủ để kết luận năng lực tổng quát. Tốc độ phụ thuộc prompt, ngân sách sinh, dtype và runtime. Trong bước tích hợp đã sửa cách lấy model revision: dùng `huggingface_hub.model_info(...).sha` thay cho thuộc tính nội bộ `_commit_hash` không còn có trên config ở phiên bản Transformers này.

### Kết nối Colab MCP trong browser Codex

Kết nối thành công bằng một client MCP cục bộ mới, dùng SDK và server Colab MCP đã cài. Client chặn riêng bước tự mở browser trong tiến trình con, lưu link phiên và mở link đó trên notebook trong Codex. Không thay browser mặc định của Windows hoặc sửa package MCP đã cài.

Server trả `result: true`; đã lấy danh sách công cụ và gọi `get_cells` đọc đúng cell đầu của notebook Phase 0. Client phiên này nằm trong `results/exploratory/colab_bridge_client.py` (không thuộc mã inference, không đóng gói vào notebook). Link kết nối chỉ được lưu trong thư mục `results/` đã bị Git ignore. Đây là phiên client phụ; không đồng nghĩa với việc kết nối cũ của tool tích hợp đã được sửa.

## 2026-10-22 — Phase 0: format dữ liệu SFT và loss mask

**Mục tiêu:** xác nhận dữ liệu được serialize đúng, assistant EOS có label, các lượt system/user/tool và padding không được tính loss. Bước này không cập nhật trọng số, không dùng eval nghiên cứu và chưa tạo tập train/dev/test.

**Triển khai:** thêm `src/sft_data.py`, `src/tool_calls.py` và bốn ví dụ viết tay trong `src/sft_examples.py`. Chat template dùng vùng generation bao quanh nội dung assistant và `<|im_end|>`. Role header và newline phân cách nằm ngoài vùng được học. Các token đã có được tái sử dụng; tokenizer dành cho SFT được copy độc lập từ playground.

**Parser/scorer:** JSON có đúng name/arguments, chặn key trùng và số không hữu hạn; đọc JSON trước khi tìm tag đóng để xử lý được chuỗi chứa text giống tag. Chấm multiset giữ số lần xuất hiện, phân biệt boolean/string/number, giữ thứ tự array. Hợp đồng chi tiết nằm trong `docs/phase-0-sft-format.md`; đây là scorer cấu trúc ban đầu, chưa phải toàn bộ evaluator BFCL.

**Kiểm chứng:**

- Local: 20 test qua, 7 test cần thư viện/tokenizer được bỏ qua.
- Colab: cả 27 test qua, không skip. Bốn test mới dùng tokenizer SmolLM2 thật đã cache ở revision `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`; ba test trước đó dùng Llama ngẫu nhiên nhỏ.
- Ví dụ tool result: input 260 token, 47 label assistant. Năm vị trí EOS là 141, 156, 194, 239, 258; chỉ 194 và 258 được học.
- Bốn ví dụ có độ dài 119, 260, 383 và 169 token; số label được học tương ứng 8, 47, 90 và 10. Batch có width 383; padding có attention mask 0 và label -100.
- Scorer cho đúng và parallel đảo thứ tự kết quả True; thiếu lời gọi, thêm lời gọi trùng và JSON hỏng kết quả False.
- Tám cell mới tại các mục 9–12 được thêm, đọc lại đối chiếu với notebook local và chạy thành công bằng MCP.

**Source bundle:** SHA-256 `a599348ace53714d0ca2ff4d79df0f7fc2b7633c057dfc0566e384d934c3ac1e`. Notebook hiện có 26 cell, mục xuất log chuyển xuống mục 13. Client MCP phụ dùng request file để chuyển source bundle lớn, không đưa các file kết nối hoặc client vào bundle chia sẻ.

**Còn lại trong Phase 0:** baseline Instruct, config train và vòng lặp train ngắn có lưu/resume checkpoint, đếm token và kiểm tra upload. Các mục này chưa được đánh dấu hoàn thành trong PLAN.

**Mốc mã nguồn:** commit `5768945`. Trước khi commit đã bỏ dòng trống dư ở cuối một số file và tạo lại notebook; source bundle local có SHA-256 `34ce17d256a9febae84b33f535c9d5ec6ba16a34d956fc7688407f5b0ee1a7a4`. Hash của bản đã chạy trên Colab ở trên vẫn được giữ để phân biệt snapshot. Các thay đổi này chỉ chuẩn hóa whitespace; kiểm tra local và đối chiếu source nhúng được chạy lại trước commit.

## 2026-10-22 — Phase 0: base và các model Instruct, cùng cấu hình dừng

**Loại hoạt động:** khám phá, chưa phải E0. Sáu prompt tự viết (`COMPARISON_PROMPTS` trong `src/playground.py`), không thuộc P/dev/test.

**Thiết lập:** Colab CPU (khoảng 11,4 GiB RAM trống), fp32, greedy, tối đa 64 token, dừng ở `<|im_end|>` hoặc `<|endoftext|>`. Mọi model dùng cùng ChatML và tool policy của dự án; SmolLM2-Instruct cũng dùng ChatML nên prompt khớp định dạng nó đã học. Revision ghim trong `configs/models.yaml`: base `f8027fd`, 360M-Instruct `a10cc15`, 1.7B-Instruct `31b70e2`. Notebook mục 13, chạy qua Colab MCP; source bundle SHA-256 `2ee3933a5435486d57c956b35591f09dbce66519eb94e5cf559d87048b903ba1`. Trên Colab 26 test qua, 1 test tokenizer được bỏ qua vì chưa cache tokenizer.

**Quan sát:**

- **Cách dừng:** base không dừng ở prompt nào (6/6 hết 64 token). 360M-Instruct dừng bằng `<|im_end|>` ở 4/6 prompt; 1.7B-Instruct dừng ở cả 6. Không model nào dừng bằng `<|endoftext|>`. Cấu hình dừng ở mục 2.4 hoạt động như dự kiến.
- **Base với ChatML:** lặp lại câu hỏi rồi sinh chuỗi số `1.2.2.2…`, hoặc tự viết thêm `assistant` và lặp lại câu hỏi. Đây là dạng role break mà `role_break_rate` cần bắt; cần kiểm tra trên token ID xem base có sinh đúng `<|im_start|>` không.
- **360M-Instruct:** trả lời chat đúng; prompt giải thích bị cắt ở 64 token. Với prompt cần tool, nó **bịa** dữ liệu thời tiết dạng JSON thay vì gọi `<tool_call>`, khớp với model card (bản 360M không hỗ trợ function calling).
- **1.7B-Instruct với format của dự án:** không sinh `<tool_call>`; từ chối vì "không có dữ liệu thời gian thực", và từ chối cả `What is 2 + 2?` khi có tool trong prompt. Như PLAN 2.1 đã ghi, mốc tool của bản này phải đo bằng prompt format riêng của nó.
- **Raw `The capital of France is`:** cả ba đều bắt đầu bằng ` Paris`; base và 360M-Instruct lặp tới hết ngân sách, 1.7B-Instruct dừng sau 3 token.

**Ý nghĩa cho các bước sau:** base cần SFT để biết dừng và giữ vai trò; 360M-Instruct là mốc chat hợp lý nhưng không phải mốc tool. Trước E0 cần một prompt riêng cho 1.7B theo format function calling của nó.

## 2026-10-22 — Phase 0: chạy thử train SFT, checkpoint lên Hub và resume

**Loại hoạt động:** kiểm tra hạ tầng train, không phải thí nghiệm. Dữ liệu là 4 ví dụ viết tay trong `src/sft_examples.py`; loss ở đây không nói gì về năng lực.

**Thiết lập:** `src/train.py` với `configs/smoke.yaml`: base `f8027fd`, 5 step, batch 2, lr 2e-5 cosine, fp16 AMP, gradient checkpointing, lưu mỗi 2 step với đầy đủ trạng thái (optimizer, scheduler, GradScaler, RNG, vị trí dữ liệu). Hub dùng `hub_strategy="checkpoint"`, repo private. Colab T4, notebook `01_train_smoke.ipynb` chạy qua MCP; phiên bản thư viện ghi trong `requirements.lock`.

**Kiểm chứng ở local (Llama ngẫu nhiên nhỏ, CPU):** chạy liền 4 step và chạy 2 step → lưu → resume → 2 step cho loss và `assistant_tokens_seen` trùng nhau. Test bắt được lỗi khi cố tình tắt khôi phục vị trí dữ liệu hoặc tắt khôi phục bộ đếm.

**Kết quả trên Colab:**

| Step | Chạy liền (loss, token) | Dừng ở step 2, đẩy Hub, resume từ Hub |
|---|---|---|
| 1 | 1.75653, 55 | 1.75653, 55 |
| 2 | 1.11975, 155 | 1.11975, 155 |
| 3 | 1.48754, 212 | 1.48728, 212 |
| 4 | 0.75629, 310 | 0.7558, 310 |
| 5 | 1.1093, 365 | 1.11, 365 |

- `assistant_tokens_seen` khớp ở mọi step, nên thứ tự dữ liệu được khôi phục đúng. Loss sau resume lệch ở chữ số thứ tư. Hai lần chạy liền cũng lệch nhau cùng cỡ (step 4: 0.75554 và 0.75629), còn hai lần resume cho kết quả giống hệt; độ lệch đến từ fp16 trên GPU không tất định, không phải từ resume.
- Step 1 có `grad_norm = inf`: GradScaler fp16 bỏ qua bước đầu khi scale khởi tạo quá lớn, và lr ở step 1 bằng 0 do warmup 3% làm tròn thành 1 step. Cả hai là hành vi bình thường, nhưng với lần chạy thật cần nhớ step đầu không cập nhật trọng số.
- Bộ nhớ GPU đỉnh 5,6 GiB (trọng số fp32 + Adam + gradient của 360M). Thời gian train khoảng 114 giây cho 5 step, chủ yếu do lưu hai checkpoint đầy đủ; tính cả upload lên Hub thì một lần chạy 2 step mất khoảng 2 phút.
- Lần chạy đầu, cell cài thư viện không cài đúng phiên bản mà không báo lỗi (Colab giữ `transformers` 5.20.1, thiếu `trl`). Notebook nay kiểm tra phiên bản sau khi cài; kết quả ở bảng trên là lần chạy lại với đúng phiên bản.

**Ghi chú vận hành:** secret `HF_TOKEN` chỉ đọc được khi cell chạy từ giao diện Colab, không đọc được qua MCP; người dùng chạy cell đăng nhập một lần, các cell sau dùng biến môi trường đó.
