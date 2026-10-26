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

## 2026-10-24 — Pilot: E0 trên dev, đo thời gian eval

**Loại hoạt động:** pilot (PLAN §6, Giai đoạn 2), chỉ dùng dev. Kiểm bộ chạy eval và scorer trên output thật; chưa phải kết luận.

**Thiết lập:** `src/eval/run_eval.py` trên dev (T-seen, T-unseen, hai mức T-rename, C; 644 mẫu) cùng bộ P (60 mẫu, chỉ sinh, không chấm). Greedy, `max_new_tokens=256`, batch 32, left padding, fp16, Colab T4, notebook `05_eval`, source bundle `7bc7db04…`; `src.eval.manifest` khớp 11 file trước khi chạy. Base `f8027fd` chạy zero-shot và few-shot (3 ví dụ cố định trong `FEW_SHOT`) với template dự án; 360M-Instruct `a10cc15` dùng chat template riêng của nó (cùng chuỗi ChatML khi có system prompt). Output và metric tải về `results/e0/` (git-ignore).

**Kết quả (dev, không có CI; clarify và grounded đã chấm lại theo luật "phải dừng đúng" bên dưới):**

| | stop_rate T-seen / T-unseen / C | call_exact | decision_bacc T-seen / T-unseen | clarify_correct T-seen / T-unseen | grounded_correct T-seen / T-unseen |
|---|---|---|---|---|---|
| base zero-shot | 0 / 0 / 0 | 0 | 0.50 / 0.50 | 0 / 0 (trước: 0.18 / 0.33) | 0 / 0 (trước: 0.00 / 0.15) |
| base few-shot | 0 / 0 / 0 | 0 | 0.47 / 0.48 | 0 / 0 (trước: 0.67 / 0.48) | 0 / 0 (trước: 0.12 / 0.15) |
| 360M-Instruct | 0.65 / 0.78 / 0.82 | 0 | 0.50 / 0.50 | 0.03 / 0.03 (trước: 0.06 / 0.03) | 0.24 / 0.12 (không đổi) |

- Không model nào gọi tool ở nhóm cần tool (`format_valid` = 0). Chuỗi `<tool_call>` chỉ xuất hiện ở Multi-turn, nơi model chép lại lời gọi ở lượt trước thay vì đọc kết quả tool.
- Perplexity wikitext-2 test (301 653 token, cửa sổ 1024): base 13.01, 360M-Instruct 14.50; KL(Instruct ‖ base) = 0.148 nat/token.
- **Thời gian sinh** (P + dev, 704 prompt): base zero-shot 311 giây, 360M-Instruct 314 giây, base few-shot 641 giây. Mỗi batch chạy tới khi mẫu dài nhất xong, nên khi base không bao giờ dừng thì mọi batch đều đủ 256 token. Checkpoint đã học cách dừng sẽ nhanh hơn hoặc bằng mức ~5 phút, trong ngân sách 10 phút (PLAN 3.6).

**Phát hiện về scorer (cần quyết định trước khi khóa giao thức):**

- Base không sinh token điều khiển thật. Ở chỗ cần `<|im_end|>` / `<|im_start|>` (id 2 / 1) nó sinh token khác như `tolist` (28928) hay `ipython`, rồi tự viết tiếp các lượt `user` / `system` giả. Vì vậy `role_break_rate` = 0 dù role break xảy ra liên tục: luật chỉ bắt token thật.
- `clarify_correct` và `grounded_correct` chấm trên toàn bộ output, nên được cộng cả phần chữ của lượt giả. Ví dụ base zero-shot được tính đúng chỉ vì lặp lại câu hỏi của user (có dấu `?` và tên tham số). Một phần điểm few-shot là thật: model chép mẫu "Sure. What is the …?" của ví dụ thứ ba, nhưng phần đuôi vẫn là lượt giả.

**Quyết định (người dùng chọn, ghi ở PLAN 3.3):** `clarify_correct` và `grounded_correct` chỉ tính đúng khi output dừng bằng `<|im_end|>`. Ba lần chạy được chấm lại từ `outputs.jsonl` bằng `python -m src.eval.run_eval --rescore results/e0/<checkpoint>`, không sinh lại. Base về 0 ở cả hai metric; 360M-Instruct mất một mẫu clarify ở T-seen (bị cắt ở 256 token). `role_break_rate` vẫn không bắt được lượt giả của base; chỉ số dừng (`stop_rate`, `truncated_rate`) là nơi phản ánh hành vi này.

## 2026-10-24 — Pilot: E2 10%, eval dev theo checkpoint

**Loại hoạt động:** pilot (PLAN §6, Giai đoạn 2), chỉ dùng dev. Mục đích là kiểm vòng train → snapshot → eval chạy đúng; chưa đọc kết luận.

**Thiết lập:** commit `1dcfecb`, source bundle `bfa69889…`, notebook `06_pilot`, Colab T4. `configs/e2_pilot.yaml` kế thừa `train_base.yaml`:
- 10% đầu của D_chat và D_tool (1000 + 1000 mẫu, hash như `data/TRAIN.md`), E2 trộn, 2 epoch, seed 0;
- 125 step, batch hiệu dụng 32 (8 × 4), lr 2e-5, cosine, warmup 3%, fp16 AMP, gradient checkpointing;
- 6 snapshot fp32 ở các mốc 25k / 50k / 100k / 200k / 400k token assistant và ở step cuối.

Repo run: `trantam31/smollm2-360m-e2-pilot` (private), có snapshot, `last-checkpoint/`, `order.json`, `metrics.json` và `results/`. Kết quả tải về `results/e2_pilot/` (git-ignore).

**Tài nguyên:**
- Bộ nhớ GPU đỉnh 11.07 GiB.
- Cell train mất 2048 giây. Con số này gồm cả lưu và đẩy 6 snapshot (mỗi cái 1.45 GB), 2 checkpoint đầy đủ và 6 lượt val loss, nên chưa phải tốc độ train thuần.
- Eval dev cộng P (704 prompt) mất 280–321 giây mỗi snapshot.

**Val loss** (chat / tool seen / tool unseen):

| step | chat | tool seen | tool unseen |
|---|---|---|---|
| 5 | 1.282 | 0.883 | 0.846 |
| 9 | 1.186 | 0.608 | 0.572 |
| 19 | 1.122 | 0.403 | 0.381 |
| 37 | 1.057 | 0.338 | 0.319 |
| 76 | 1.028 | 0.309 | 0.291 |
| 125 | 1.023 | 0.304 | 0.286 |

**Metric dev** (điểm ước lượng, chưa có CI; mỗi ô là T-seen / T-unseen; base zero-shot ở E0 bằng 0 với mọi metric trừ `decision_bacc` = 0.5):

| step | token assistant | stop_rate | format_valid | call_exact | call_exact Single | decision_bacc | false_call_rate | clarify_correct | grounded_correct | call_exact rename (vô nghĩa / đồng nghĩa) | stop C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 29k | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.50 / 0.50 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| 9 | 51k | 0 / 0 | 0.06 / 0.07 | 0.02 / 0.05 | 0.03 / 0.09 | 0.64 / 0.63 | 0.06 / 0.06 | 0 / 0 | 0 / 0 | 0 / 0.01 | 0 |
| 19 | 106k | 0.51 / 0.60 | 0.44 / 0.55 | 0.10 / 0.23 | 0.24 / 0.48 | 0.67 / 0.64 | 0.67 / 0.70 | 0 / 0 | 0.03 / 0 | 0.05 / 0.06 | 0.24 |
| 37 | 203k | 0.93 / 0.96 | 0.96 / 0.95 | 0.40 / 0.52 | 0.48 / 0.76 | 0.80 / 0.76 | 0.39 / 0.45 | 0 / 0 | 0.91 / 0.79 | 0.25 / 0.30 | 0.76 |
| 76 | 406k | 0.95 / 0.94 | 0.97 / 0.95 | 0.43 / 0.58 | 0.48 / 0.76 | 0.80 / 0.86 | 0.39 / 0.27 | 0.09 / 0.15 | 0.91 / 0.79 | 0.32 / 0.38 | 0.76 |
| 125 | 672k | 0.95 / 0.96 | 0.98 / 0.95 | 0.47 / 0.61 | 0.48 / 0.76 | 0.82 / 0.86 | 0.36 / 0.27 | 0.12 / 0.15 | 0.88 / 0.82 | 0.34 / 0.40 | 0.72 |

Snapshot cuối: perplexity wikitext-2 là 13.19 (base 13.01, 360M-Instruct 14.50); KL tới base là 0.011 nat/token (360M-Instruct là 0.148).

**Ghi chú cho các bước còn lại của pilot (chưa phải kết luận):**
- **Vòng chạy:** chạy trọn từ đầu đến cuối. Thứ tự mẫu, mốc snapshot và số token khớp với lúc chạy thử dữ liệu ở máy.
- **Định dạng và cách dừng:** cùng xuất hiện quanh 100k–200k token. Lúc đó `false_call_rate` vọt lên 0.67–0.70 ở step 19 rồi giảm, gần với dạng mà H2 mô tả. Nhưng đây là một seed và chưa có CI.
- **Missing-info yếu nhất:** `clarify_correct` ≤ 0.15. Pilot chỉ có khoảng 70 mẫu Missing-info (D_tool chỉ có 694 câu khác nhau). Cần xem output trước khi đọc tiêu chí đi/dừng (b).
- **T-unseen cao hơn T-seen ở `call_exact`** (Single: 0.76 so với 0.48). Pool unseen nhiều bài tính toán với lời gọi đơn giản, nên chưa thể hiểu là "tổng quát hóa tốt". Cần xem output và phần so khớp theo độ khó (PLAN 4.2 ý 5).
- **Bộ C bị cắt nhiều:** 24–28% câu trả lời chạm `max_new_tokens` = 256, vì câu trả lời chat của SmolTalk dài. `stop_rate` của C vì vậy bị chặn trên bởi cấu hình eval chứ không phải hành vi (PLAN 2.4). Cần bàn trước khi khóa giao thức: tăng `max_new_tokens` cho C, hoặc báo `stop_rate` C chỉ trên phần không bị cắt.
- **Chưa làm:** bootstrap CI (cần cho tiêu chí đi/dừng), chấm tay 50 output, bộ output dựng có chủ đích, kiểm resume ở quy mô thật, đo thời gian train thuần, và sweep lr.

## 2026-10-26 — Pilot: bootstrap CI và tiêu chí đi/dừng

**Loại hoạt động:** pilot (PLAN §6, Giai đoạn 2), chỉ dùng dev. Tính offline từ `outputs.jsonl` của E0 và của 6 snapshot pilot E2 10%; không generate lại.

**Cách tính:** `python -m src.eval.stats --go-no-go e2_pilot`, mất khoảng 7 giây.
- Bootstrap 95% theo phân vị, 2000 lần. Lấy mẫu lại theo cụm, cách gom cụm như PLAN §3.5 (2026-10-26). Hiệu số so với base zero-shot là theo cặp.
- Kết quả ra `results/all.csv`, `diff_base.csv`, `emergence.csv` (git-ignore).
- Trước khi sửa scorer (xem mục chấm tay bên dưới), điểm đã lưu khớp scorer: chấm lại không đổi mẫu nào. Giá trị điểm trong `all.csv` trùng `metrics.json` ở cả 2466 dòng. Bảng dưới là số sau khi sửa và chấm lại.
- Cụm ở dev (33 mẫu mỗi nhóm) phần lớn chỉ 1–4 mẫu. Riêng Multi-turn gom mạnh hơn: T-seen 19 cụm, T-unseen 13 cụm, cụm lớn nhất 10 mẫu.

**Snapshot cuối (step 125), giá trị [CI 95%] và cận dưới CI của hiệu số so với base zero-shot.** Số đã tính lại sau khi sửa scorer (`grounded_correct` T-unseen trước khi sửa là 0.82):

| Nhóm | Metric | T-seen | T-unseen |
|---|---|---|---|
| tất cả | `stop_rate` | 0.95 [0.92, 0.98], +0.92 | 0.96 [0.93, 0.98], +0.93 |
| 4 nhóm tool | `format_valid` | 0.98 [0.95, 1.00], +0.95 | 0.95 [0.90, 0.98], +0.90 |
| 4 nhóm tool | `call_exact` | 0.46 [0.37, 0.56], +0.37 | 0.61 [0.50, 0.70], +0.50 |
| Single | `call_exact` | 0.48 [0.33, 0.64], +0.33 | 0.76 [0.61, 0.88], +0.61 |
| 4 nhóm tool | `decision_bacc` | 0.82 [0.73, 0.90], +0.23 | 0.86 [0.78, 0.93], +0.28 |
| No-tool | `false_call_rate` | 0.36 [0.21, 0.55] | 0.27 [0.12, 0.42] |
| Missing-info | `clarify_correct` | 0.12 [0.03, 0.25], +0.03 | 0.15 [0.03, 0.31], +0.03 |
| Multi-turn | `grounded_correct` | 0.88 [0.75, 0.97], +0.75 | 0.91 [0.74, 1.00], +0.74 |
| Multi-turn | `grounded_correct_not_derivable` | 0.64 [0.33, 0.90], +0.33 | 0.75 [0.51, 1.00], +0.51 |

Bộ C, `stop_rate`: 0.72 [0.58, 0.84]; `truncated_rate`: 0.28 [0.16, 0.42].

**Tiêu chí đi/dừng (PLAN §6), theo scorer:**
- **(a) đạt.** Single T-seen: `format_valid` từ 0 lên 1.00 (hiệu số +1.00, CI [+1.00, +1.00]); `call_exact` từ 0 lên 0.48 (+0.48, CI [+0.33, +0.64]).
- **(b) đạt, rõ nhất ở hai metric.** `call_exact` Single T-unseen có cận dưới CI +0.61; `grounded_correct` có cận dưới +0.75 (seen) và +0.74 (unseen). `clarify_correct` chỉ vừa qua 0, cận dưới +0.03.
- **Kết luận theo luật đã chốt trước pilot: đạt (a) và (b), giữ SmolLM2-360M.** Chấm tay bên dưới xác nhận hai metric chính của (b). Luật `grounded_correct` trùng chấm tay ở 11/12 mẫu sau khi sửa. `call_exact` T-unseen cao hơn T-seen là do mẫu T-unseen dễ hơn, không phải do scorer.

**t_reach / t_improve** (base zero-shot là step 0; một seed, chỉ để kiểm cách tính):
- `format_valid`, `stop_rate`: đạt θ = 0.5 ở step 19–37 (khoảng 100k–200k token assistant). t_improve ở step 19.
- `call_exact` T-seen: không đạt θ = 0.5. Chỉ đạt θ = 0.3, ở step 37.
- `tool_name_acc` ≥ 0.7 và `decision_bacc` T-unseen ≥ 0.85: ghi "chưa xác nhận", vì chỉ thỏa từ step 76, sau đó còn đúng một checkpoint.
- `clarify_correct`: không đạt ở ngưỡng nào, kể cả t_improve.
- `false_call_rate` ở step 19 là 0.67 [0.52, 0.82] (seen) và 0.70 [0.52, 0.85] (unseen): đỉnh ở step 19 không phải nhiễu của tập prompt. Vẫn mới một seed.

### Chấm tay 50 output (snapshot step 125)

Rút ngẫu nhiên, seed 0, phân tầng theo nhóm: Missing-info 12, Multi-turn 12, Single 12 (mỗi loại 6 mẫu T-seen và 6 mẫu T-unseen), No-tool 7, C 7. Mỗi mẫu được chấm tay rồi so với metric chính của nhóm.
- **Trùng scorer:** 48/50 trước khi sửa, 49/50 sau khi sửa.
- **Missing-info, 12/12 trùng.** Cả 12 mẫu: model bịa giá trị cho tham số thiếu rồi gọi tool. Xem thêm cả 10/72 mẫu Missing-info mà model không gọi tool: 9 mẫu hỏi lại đúng tham số và được scorer chấm đúng, 1 mẫu bị cắt. Luật không chặt quá. `clarify_correct` thấp là do model thật.
  - Lệch nguồn: mẫu xLAM viết lại thì model gọi tool ở 36/36 mẫu, đúng 0. Mẫu Glaive thì đúng 9/30. Train chỉ có Missing-info từ Glaive (`data/TRAIN.md`), nên model chưa học được cách hỏi lại khi câu hỏi theo kiểu xLAM thiếu một tham số.
- **Multi-turn, 10/12 trùng.**
  - Sai âm (đã sửa): luật so số như chuỗi, nên "$57.50" không khớp 57.5. `score_grounded` nay bỏ số 0 thừa ở phần thập phân trước khi so; `_normalize` của `build_sets` giữ nguyên nên bộ eval không đổi. Câu trả lời mẫu vẫn đúng 266/266. Mỗi snapshot từ step 37 trở đi có 2–3 mẫu T-unseen chuyển sang đúng; chấm lại bằng `run_eval.rescore`.
  - Sai dương (chưa sửa): "The discounted price of the product is $30." Ở đây $30 là số tiền được giảm. Luật chỉ kiểm giá trị có xuất hiện, không kiểm vai trò của nó. Trong mẫu này có 1/10 mẫu luật chấm đúng mà chấm tay chấm sai, nên `grounded_correct` có thể cao hơn thực tế đôi chút.
- **Single, 12/12 trùng** theo định nghĩa khớp hoàn toàn. Có 1 mẫu ở ranh giới: model thêm `source: "en"`, một tham số tùy chọn mà câu hỏi không nêu. Theo định nghĩa thì sai.
  - Vì sao T-unseen > T-seen: lời gọi đáp án của Single T-seen có trung bình 0.82 tham số tùy chọn, phải suy từ câu (như "Amazon.in" → `country: IN`, "latest" → `sort_by`). T-unseen chỉ có 0.18, phần lớn là bài tính đủ tham số ngay trong câu. Chỉ xét lời gọi toàn tham số bắt buộc: T-seen 0.63 (19 mẫu), T-unseen 0.74 (27 mẫu). Cần so khớp theo độ khó (PLAN 4.2 ý 5) trước khi đọc RQ3.
- **No-tool, 7/7 trùng** về quyết định gọi/không gọi. Hai mẫu không gọi tool nhưng câu trả lời kém (lặp vòng tới khi bị cắt; nói "không có tool" rồi hỏi lại câu không liên quan), mà `false_call_rate` không đo chất lượng câu trả lời.
- **C, 7/7 trùng về `stop_rate`.** Có 1/50 mẫu C sinh `<tool_call>` hỏng trong khi không có tool nào, và không metric nào bắt được trường hợp này.
- **Bộ C bị cắt** (14/50 ở step 125): khoảng một nửa là lặp vòng (repetition ≥ 0.3), tức lỗi dừng thật; phần còn lại là câu trả lời dài hợp lệ. Ước theo số từ, khoảng 22% câu trả lời mẫu của dev (11/50) và 16% của test (24/150) dài hơn 256 token, nên `stop_rate` C bị chặn trên khoảng 0.8. Ở 512 token chỉ còn 1/50 và 2/150. **Đã quyết (2026-10-26): bộ C dùng `max_new_tokens` = 512, các bộ khác giữ 256** (PLAN §3.6). Số liệu C của E0 và của pilot ở trên chạy với 256, nên không so được với các lần chạy sau. Trước khi khóa giao thức cần chạy lại riêng bộ C cho E0, và cho pilot nếu cần.

### Bộ output dựng có chủ đích kiểm độ nhạy scorer

`python -m src.eval.sensitivity` dựng output từ nhãn của **mọi** mẫu dev và test. Mỗi mẫu có một output đúng và các output mắc từng lỗi đã biết. Mỗi output được chấm bằng `run_eval.score`, rồi so với kết quả kỳ vọng.

**Các lỗi dựng cho từng nhóm:**
- Single/Select/Parallel: sai tên tool, thiếu tham số, thừa tham số, sai kiểu (5 → "5", "Hanoi" → ["Hanoi"]), sai giá trị, JSON hỏng, không gọi tool, gọi lặp, bỏ một lời gọi (Parallel). Thêm một biến thể đúng có chữ bao quanh lời gọi: `call_exact` vẫn đúng, `strict_format` sai.
- No-tool: gọi tool khi không cần.
- Missing-info: hỏi sai tham số, không hỏi, gọi tool, bị cắt.
- Multi-turn: thiếu giá trị, phủ định giá trị đúng ("It is not 31"), gọi lại tool, bị cắt. Câu trả lời mẫu và câu nêu đủ giá trị phải được chấm đúng.

**Kết quả:** mọi cặp (mẫu, biến thể) đều chấm đúng kỳ vọng, 0 lệch. Trong đó 532 mẫu cho mỗi nhóm gọi tool, 266 cho mỗi nhóm còn lại; Single/Select thiếu khoảng 8–9 mẫu ở ba lỗi về tham số vì lời gọi không có tham số nào.

**Kiểm chính phép thử:** làm hỏng scorer có chủ đích thì phép thử bắt được. Tắt luật phủ định: 266/266 mẫu `negated_value` bị phát hiện. Cho `canonical` bỏ qua giá trị: 524/524 mẫu `wrong_value` bị phát hiện.

**Giới hạn:**
- Không mẫu Multi-turn nào có nhãn `forbidden`, nên luật giá trị cấm chỉ được kiểm bằng unit test, chưa chạy trên nhãn thật.
- Ở 122/266 mẫu Missing-info, mọi tham số khác của tool đều trùng từ đồng nghĩa, nên lỗi "hỏi sai tham số" phải dùng một tham số giả.
- Bộ này chỉ kiểm lỗi đã biết. Lỗi về vai trò của giá trị ("giá sau giảm là $30", khi $30 là số tiền được giảm) không bị bắt; xem mục chấm tay.
