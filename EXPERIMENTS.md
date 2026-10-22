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
