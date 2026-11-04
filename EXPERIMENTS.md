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

### Chạy lại bộ C với 512 token (2026-10-26)

**Cách chạy:** notebook `06_pilot_lr` (Colab T4), mục 11: `run_eval --sets c` cho E0, `run_checkpoints --sets c` cho 6 snapshot pilot. Chỉ C được sinh lại; các bộ khác giữ output cũ và được chấm lại theo luật hiện tại. Mọi metric ngoài C trùng bản ở máy. Kết quả nằm ở `results/{e0,e2_pilot}/` trên Hub `trantam31/smollm2-360m-e2-pilot`.

**`stop_rate` C, 256 → 512 token** (`truncated_rate` trong ngoặc):

| Model | 256 | 512 |
|---|---|---|
| base zero-shot / few-shot | 0.00 (0.96) | 0.00 (0.96) |
| 360M-Instruct | 0.82 (0.18) | 0.96 (0.04) |
| pilot step 19 | 0.24 (0.76) | 0.26 (0.74) |
| pilot step 37 | 0.76 (0.24) | 0.80 (0.20) |
| pilot step 76 | 0.76 (0.24) | 0.92 (0.08) |
| pilot step 125 | 0.72 (0.28) | 0.82 (0.18) |

- 360M-Instruct gần như hết bị cắt, nên trần ở 256 đúng là do cấu hình.
- Pilot ở step 125 vẫn bị cắt 18% mẫu C. Phần này phần lớn là lặp vòng (xem mục chấm tay), tức lỗi dừng thật của model.

## 2026-10-26 — Pilot: sweep lr, kiểm resume, thời gian train

**Loại hoạt động:** pilot (PLAN §5.2, §6 Giai đoạn 2), chỉ dùng dev. Tiêu chí chọn lr được ghi vào PLAN §5.2 trước khi chạy.

**Thiết lập:** như pilot E2 10% (`configs/e2_pilot.yaml`), chỉ đổi lr: `configs/e2_pilot_lr1e-5.yaml`, `configs/e2_pilot_lr5e-5.yaml`. Notebook `06_pilot_lr`, Colab T4, bundle `fd73d015…`. Repo run: `trantam31/smollm2-360m-e2-pilot-lr1e-5`, `…-lr5e-5`. Bộ C đã dùng 512 token cho cả ba lr, vì pilot 2e-5 đã được chạy lại phần C.

**Kết quả ở step 125** (metric dev là điểm ước lượng, mỗi ô T-seen / T-unseen):

| lr | val loss chat | val loss tool seen | val loss tool unseen | perplexity | KL tới base | `call_exact` | `decision_bacc` | `clarify_correct` | `grounded_correct` | `stop_rate` C |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-5 | 1.054 | 0.337 | 0.318 | 13.13 | 0.007 | 0.40 / 0.55 | 0.77 / 0.83 | 0.00 / 0.00 | 0.82 / 0.82 | 0.80 |
| 2e-5 | 1.023 | 0.304 | 0.286 | 13.19 | 0.011 | 0.46 / 0.61 | 0.82 / 0.86 | 0.12 / 0.15 | 0.88 / 0.91 | 0.82 |
| 5e-5 | 1.005 | 0.274 | 0.258 | 13.41 | 0.026 | 0.55 / 0.71 | 0.88 / 0.88 | 0.39 / 0.27 | 0.94 / 1.00 | 0.88 |
| 1e-4 | 1.017 | 0.268 | 0.254 | **13.93** | 0.058 | 0.58 / 0.73 | 0.89 / 0.92 | 0.39 / 0.30 | 0.91 / 1.00 | 0.88 |

Base: perplexity 13.01, nên ràng buộc là ≤ 13.66.

**Chọn theo tiêu chí đã ghi trước: lr 5e-5.** Lưới được mở rộng thêm 1e-4 sau khi thấy 5e-5 nằm ở mép (PLAN §5.2). Run 1e-4 chạy trên notebook `06_pilot_lr`, bundle `f876fa0c…`, repo `trantam31/smollm2-360m-e2-pilot-lr1e-4`.
- 1e-4 bị loại: perplexity 13.93 vượt ràng buộc 13.66. Dù không bị loại thì nó cũng chỉ hơn 5e-5 0.006 val loss tool seen, dưới ngưỡng hòa 0.01, nên vẫn chọn lr nhỏ hơn.
- val loss tool seen thấp nhất, hơn 2e-5 0.030 (> 0.01);
- perplexity 13.41 ≤ 13.66;
- val loss chat 1.005, thấp hơn của 2e-5.

Metric dev đi cùng chiều, nhất là `clarify_correct` (0 → 0.12 → 0.39 trên T-seen), nhưng không dùng để chọn.

**Lưu ý:**
- Điểm tối ưu nay đã nằm trong lưới: từ 5e-5 lên 1e-4, val loss tool gần như đứng yên (0.274 → 0.268), trong khi độ lệch khỏi base tăng mạnh (KL 0.026 → 0.058, perplexity 13.41 → 13.93). Metric dev của 1e-4 ngang 5e-5, trong khoảng nhiễu của 33 mẫu mỗi nhóm.
- Perplexity và KL cũng tăng theo lr (KL 0.007 → 0.026). Run chính dài gấp khoảng 10 lần pilot, nên phải theo dõi phần quên.
- Chưa sửa `configs/train_base.yaml`: chờ chốt cùng lúc khóa giao thức.

**Kiểm resume ở quy mô thật (PLAN 5.3):** run lr 5e-5 dừng ở step 50 (`--stop-after 50`), rồi chạy tiếp từ `last-checkpoint/` trên Hub (`--resume hub`).
- Thứ tự mẫu trùng pilot.
- Số token assistant cộng dồn chạy tiếp đúng, kết thúc ở 671 816, bằng pilot.
- Loss liền mạch qua chỗ nối: 0.818 ở step 50, 0.826 ở step 55.
- 6 snapshot đủ, chấm bình thường.

**Thời gian và tài nguyên (T4):**
- Train thuần (`step_seconds`, chỉ tính bên trong các step): lr 1e-5 mất 1076 giây; lr 5e-5 mất 420 + 664 = 1084 giây; lr 1e-4 mất 1068 giây. Cùng là 125 step × 32 = 4000 mẫu, tức khoảng 270 giây cho 1000 mẫu.
- Snapshot fp32, val loss và đẩy Hub tốn thêm khoảng 870 giây mỗi run.
- Eval dev (6 snapshot) cộng perplexity mất khoảng 37 phút.
- GPU đỉnh 11.07 GiB.
- Ước lượng cho run chính 20k mẫu × 2 epoch = 40k mẫu: train thuần khoảng 3 giờ, chưa tính snapshot và eval.

## 2026-10-26 — Dữ liệu: thêm Missing-info kiểu xLAM vào D_tool

**Loại hoạt động:** sửa dữ liệu train sau pilot, trước khi khóa giao thức (PLAN §5.1). Không đụng bộ eval.

**Lý do:** ở pilot, `clarify_correct` trên mẫu xLAM viết lại là 0/36, trên mẫu Glaive là 9/30. Train chỉ có Missing-info từ Glaive (694 câu mở đầu khác nhau), phần thiếu 306 mẫu đã bị chuyển sang nhóm gọi tool.

**Cách làm:** `src/data/train_sets.py`, config `train_sets.missing_info_rewrite`.
- 429 nháp từ câu hỏi xLAM single/select chưa dùng trong tập train.
- 4 subagent gán nhãn: câu hỏi viết lại, `missing_param`, từ đồng nghĩa, câu hỏi lại. Kết quả: 359 đạt, 70 bị loại. Lý do loại chủ yếu là tham số có giá trị mặc định ghi trong mô tả, giá trị phân trang hoặc chung chung, hoặc giá trị suy ra được từ phần còn lại.
- Mọi nhãn qua `apply_rewrite`, câu hỏi lại phải đạt `score_clarify`. 31 câu viết lại bị bỏ vì trùng prompt khác, lấy 306.
- Tự đọc rà 14 mẫu ngẫu nhiên: câu viết lại tự nhiên, câu hỏi lại hỏi đúng tham số thiếu. Có 352/359 câu hỏi lại khác nhau; khoảng 20% mở bằng "Sure. What is…".
- Nhãn ở `data/train/labels/missing_info_rewrite.jsonl` (git-ignore, có trên dataset Hub).

**Kết quả:**
- D_tool `70de4fa2…` (trước là `5467d878…`), có 1000 mẫu Missing-info: 694 Glaive và 306 xLAM viết lại.
- Bớt đúng 306 mẫu gọi tool (105 single, 104 select, 97 parallel); 9694 mẫu còn lại giữ nguyên từng byte.
- D_chat và nháp No-tool giữ nguyên hash.
- Token assistant của D_tool: 0.960 triệu (trước là 0.976 triệu).
- Đã đẩy lên dataset Hub `trantam31/sft-capability-data`.

**Ảnh hưởng tới pilot:** pilot E2 và sweep lr chạy trên D_tool cũ. Chênh lệch chỉ ở 306/10000 mẫu, nên lr đã chọn vẫn dùng được. Kết quả Missing-info của pilot không còn đại diện cho dữ liệu mới.

## 2026-10-26 — Khóa giao thức (PLAN §6, cuối Giai đoạn 2)

Từ mốc này mới được chạy test. Mọi thay đổi sau đây về dữ liệu, metric, ngưỡng hay siêu tham số phải ghi thành một mục mới, kèm lý do, và không được sửa ngược phần dưới.

**Mã và dữ liệu**
- Code: commit `6c919c8`, source bundle `b6232a2e…` (`scripts/build_playground_notebook.py`).
- Model: `HuggingFaceTB/SmolLM2-360M` @ `f8027fd0…` (`configs/models.yaml`). Giữ 360M theo tiêu chí đi/dừng.
- Train: D_chat `58674077…` (10 000 mẫu), D_tool `70de4fa2…` (10 000 mẫu, gồm 1000 Missing-info: 694 Glaive và 306 xLAM viết lại). Xem `data/TRAIN.md` và dataset Hub `trantam31/sft-capability-data`.
- Eval: dev/test đóng băng theo `eval_sets/MANIFEST.md` (`7aeaeb97…`, 11 file khớp); P `eval_sets/p.jsonl` (`54db40ea…`).

**Run (PLAN §5.1)**
- Mỗi mẫu được học đúng 2 epoch. Tổng khoảng 6.53 triệu token assistant mỗi run.
- E1 (Seq): D_chat × 2 epoch rồi D_tool × 2 epoch, một lịch lr toàn cục, optimizer liên tục. Seed 0, 1, 2.
- E2 (Mix): D_chat ∪ D_tool trộn ngẫu nhiên. Seed 0, 1, 2.
- E3 (LoRA, như E2): r = 64 với seed 0, 1, 2; r = 8 với seed 0. lr 2e-4, alpha = 2r, dropout 0.05, gắn vào q/k/v/o/gate/up/down của cả 32 layer, không gắn embedding (PLAN §5.2). lr của LoRA không qua sweep.
- Tổng 10 run.
- Giả thuyết của mỗi run phải ghi vào EXPERIMENTS.md trước khi chạy run đó.

**Siêu tham số full fine-tune** (`configs/train_base.yaml`): lr 5e-5 (chọn theo tiêu chí ghi trước khi sweep, xem mục "sweep lr"), cosine, warmup 3% toàn run, batch hiệu dụng 32 (8 × 4), seq len 1024, fp16 AMP, gradient checkpointing, grad clip 1.0, weight decay 0.

**Checkpoint (PLAN §3.7):** snapshot fp32 ở step đầu tiên mà token assistant cộng dồn chạm 25k, 50k, 100k… (nhân 2), cộng step cuối, khoảng 10 snapshot mỗi run. Trục x là `assistant_tokens_seen`.

**Eval (PLAN §3.3, §3.6)**
- Chấm test ở mọi snapshot.
- Greedy; `max_new_tokens` = 256, riêng bộ C là 512. Cấu hình dừng theo PLAN §2.4.
- Scorer ở commit trên: luật so số bỏ số 0 thừa ở phần thập phân; phủ định trong 3 từ trước giá trị; `clarify_correct` và `grounded_correct` yêu cầu dừng bằng `<|im_end|>`.
- Base được chấm zero-shot và few-shot; 360M-Instruct chấm bằng template riêng của nó.
- G: perplexity wikitext-2, KL tới base, ARC-Easy/HellaSwag ở các mốc chính. BFCL ở các mốc chính.

**Thống kê và ngưỡng (PLAN §3.5):** metric kỹ năng và θ giữ nguyên (θ = 0.5, riêng `decision_bacc` là 0.75; bộ θ kiểm độ nhạy như PLAN).
- t_reach: đạt θ và giữ ở 2 checkpoint kế tiếp.
- t_improve: hiệu số so với base zero-shot ≥ 0.2, cận dưới CI > 0, giữ ở 2 checkpoint kế tiếp.
- CI: bootstrap 95%, 2000 lần, theo cặp và theo cụm (`src/eval/stats.py`; cụm theo cùng lượt user đầu hoặc cùng tool quyết định đáp án).
- Biến thiên do seed báo riêng: độ lệch chuẩn giữa 3 seed.

**Ngân sách:** ước khoảng 6 giờ GPU T4 mỗi run full fine-tune (train khoảng 3.5 giờ, chấm test khoảng 2.5 giờ), tổng khoảng 55–60 giờ. Mỗi lần có hạn mức Colab miễn phí chỉ được khoảng 3 giờ GPU, nên một run phải chạy qua nhiều lần, nối lại bằng resume từ Hub (đã kiểm ở quy mô thật).

**Hạn chế đã biết lúc khóa (không sửa trước khi chạy):**
- `grounded_correct` chỉ kiểm giá trị có xuất hiện, không kiểm vai trò của giá trị. Trong 10 mẫu luật chấm đúng ở pilot, chấm tay thấy 1 mẫu sai.
- Không mẫu Multi-turn nào có nhãn giá trị cấm.
- Việc sinh `<tool_call>` trong bộ C chưa có metric. Nếu cần thì thêm sau bằng cách chấm lại `outputs.jsonl`; đó là phân tích phụ, không đổi metric chính.
- Missing-info của train có 31% kiểu xLAM, của eval khoảng 55%.
- T-unseen dễ hơn T-seen (ít tham số tùy chọn hơn). Kết luận RQ3 cần bản so khớp theo độ khó (PLAN §4.2 ý 5).

## 2026-10-27 — Run chính: config, LoRA, lịch snapshot của E1

**Loại hoạt động:** chuẩn bị Giai đoạn 3, sau khi khóa giao thức. Mục này bổ sung cho mục "Khóa giao thức", không sửa mục đó.

**Config và code**
- `configs/e1_seq.yaml`, `configs/e2_mix.yaml`, `configs/e3_lora_r64.yaml` và `configs/e3_lora_r8.yaml` đều kế thừa `train_base.yaml`. Seed truyền bằng `python -m src.train --config … --seed S`, và `{seed}` trong `run_name`, `output_dir`, `hub_model_id` được điền tự động (ví dụ `e2_mix_s0`).
- LoRA (`src/train.py`, khóa `lora`): PEFT, cài đặt như PLAN §5.2. Snapshot của E3 lưu trọng số đã merge vào một bản sao (fp32), nên được chấm và đo ΔW như full fine-tune; model đang train vẫn giữ adapter. Checkpoint để resume chỉ chứa adapter và trạng thái optimizer.
- Test: LoRA chỉ đổi trọng số adapter; snapshot đã merge cho cùng logits với model PEFT; resume LoRA trùng đường loss với run chạy liền. Chưa chạy LoRA trên GPU, nên chưa đo tốc độ và bộ nhớ.

**Lịch snapshot của E1: thêm mốc ranh giới và thang thứ hai**
- *Vấn đề (thấy khi tính lịch trên dữ liệu thật):* D_chat chiếm 4.61 triệu trong 6.53 triệu token assistant mỗi run (khoảng 70%). Với thang nhân đôi trên token cộng dồn, mốc 3.2M vẫn nằm trong giai đoạn chat, nên giai đoạn tool của E1 (step 626–1250, khoảng 1.92 triệu token) chỉ có snapshot ở step 1209 và 1250. Như vậy không đo được kỹ năng tool xuất hiện ra sao trong E1, trong khi so sánh E1 với E2 là mục đích của RQ về thứ tự dữ liệu.
- *Thay đổi:* E1 (`stage_snapshots: true`) lưu thêm hai loại snapshot:
  - Snapshot ở step chat cuối cùng (`boundary` trong `meta.json`), là điểm "chỉ học chat" mà PLAN §5.1 đã mô tả nhưng mục khóa chưa ghi.
  - Thang thứ hai tính token assistant kể từ ranh giới: 25k, 50k, 100k… (nhân 2), ghi `stage_target_tokens` trong `meta.json`. Một step thuộc cả hai thang chỉ lưu một lần, mang cả hai mốc.
- *Lịch thực tế (seed 0):* thang tích lũy ở step 4, 7, 15, 27, 54, 108, 218, 436; ranh giới ở step 625; thang từ ranh giới ở step 634, 643, 659, 691, 754, 886, 1145; rồi step 1209 (6.4M) và 1250 (cuối). Tổng 18 snapshot, so với 10 của E2. Seed khác lệch vài step.
- *Không đổi:* thang tích lũy, θ, luật t_reach/t_improve, cách tính CI. Trục x chính vẫn là `assistant_tokens_seen`. Thang thứ hai chỉ thêm điểm đo cho giai đoạn tool.
- *Đọc kết quả:* để so tốc độ học tool giữa E1 và E2, có thể vẽ thêm theo số token tool đã học (với E1 là token kể từ ranh giới; với E2 thì tính từ `order.json`). Đây là phân tích phụ. Khi áp luật t_reach (giữ ở 2 checkpoint kế tiếp) cho E1, dùng mọi snapshot theo thứ tự step.
- *Chi phí:* thêm 8 snapshot mỗi run E1. Ở mức khoảng 15 phút chấm test mỗi snapshot, mỗi run tốn thêm khoảng 2 giờ GPU, 3 seed khoảng 6 giờ. Tổng ngân sách khoảng 61–66 giờ GPU.

## 2026-10-27 — E0 trên test

**Loại hoạt động:** Giai đoạn 3, chạy sau mục "Khóa giao thức". Mốc so sánh cho t_improve.

**Thiết lập:** notebook `05_eval` tải lên mới từ repo, source bundle `acdaa2c3…` (commit `c7a7dea`), Colab T4. Trước khi chạy, `src.eval.manifest` báo 11 file khớp. `src/eval/run_eval.py --split test`: T-seen và T-unseen mỗi bộ 600 mẫu, hai mức T-rename mỗi mức 300, C 150, cùng bộ P (chỉ sinh). Tổng 2010 prompt. Greedy, batch 32, fp16; `max_new_tokens` 256, riêng C là 512. Base `f8027fd` chấm zero-shot và few-shot (template dự án); 360M-Instruct `a10cc15` dùng template riêng của nó. Perplexity và KL trên wikitext-2 không phụ thuộc split, nên dùng lại số ở mục "E0 trên dev". Output và metric ở `results/e0_test/` (git-ignore) và repo Hub private `trantam31/smollm2-360m-e0`.

**Kết quả (test, chưa có CI):**

| | stop_rate T-seen / T-unseen / C | call_exact | decision_bacc T-seen / T-unseen | clarify_correct T-seen / T-unseen | grounded_correct T-seen / T-unseen |
|---|---|---|---|---|---|
| base zero-shot | 0 / 0 / 0 | 0 | 0.50 / 0.50 | 0 / 0 | 0 / 0 |
| base few-shot | 0 / 0 / 0 | 0 | 0.50 / 0.49 | 0 / 0 | 0 / 0 |
| 360M-Instruct | 0.70 / 0.79 / 0.93 | 0 | 0.50 / 0.50 | 0.06 / 0.02 | 0.17 / 0.08 |

- Giống dev: không model nào gọi tool đúng (`call_exact` = 0 ở mọi bộ, kể cả T-rename). Base gần như không bao giờ dừng: `truncated_rate` 0.97–1.0 ở mọi bộ.
- 360M-Instruct dừng ở 0.54–0.55 mẫu T-rename. Nó bị cắt ở 0.07 mẫu C khi sinh 512 token.
- Base zero-shot đạt 0 ở mọi metric kỹ năng, nên với t_improve (hiệu số ≥ 0.2) mức sàn chính là ngưỡng tuyệt đối. Đây đúng là điều đã biết ở dev.
- **Thời gian sinh** (2010 prompt): base zero-shot 948 giây, 360M-Instruct 949 giây, base few-shot 1902 giây. Một checkpoint không bao giờ dừng tốn khoảng 16 phút chấm test, khớp ước tính 15 phút trong mục khóa giao thức.

## 2026-10-27 — Lưu trữ trên Hub cho run chính

**Loại hoạt động:** sửa hạ tầng, không đổi metric hay thiết lập train.

**Vấn đề:** dung lượng private trên Hub là 93.5/100 GB. Mỗi repo pilot tốn 20 GB vì ba lý do:
- Thư mục `hub/` (checkpoint tải về khi resume, 4.3 GB) bị đẩy lên cùng `output_dir`.
- Mỗi lần đẩy `last-checkpoint/` (4.3 GB) để lại một bản trong lịch sử repo.
- Bản thân snapshot.

Một run chính có khoảng 25 lần lưu checkpoint, nên riêng phần lịch sử đã khoảng 100 GB. Cả 10 run có 124 snapshot fp32, khoảng 180 GB.

**Quyết định (người dùng chọn):**
- Repo model của run chính để public: snapshot, checkpoint, metric train.
- Output eval và dataset giữ private, vì `outputs.jsonl` chứa nguyên văn dev/test rút từ dữ liệu gated. Kết quả nằm ở repo private `trantam31/smollm2-360m-results`, mỗi run một thư mục.
- Snapshot giữ fp32 như PLAN §3.7.

**Đã làm:**
- 4 repo pilot: chuyển `results/` sang repo kết quả private (đối chiếu đủ file), xóa khỏi repo pilot, squash lịch sử, rồi đặt public. Repo smoke cũng public.
- `src/train.py`:
  - Tự đẩy checkpoint qua `CheckpointPusher`: luồng nền, mỗi lần đẩy xong thì squash lịch sử, đợi đẩy xong trước khi checkpoint cũ bị xoay vòng và trước mỗi commit snapshot.
  - Hai ô `checkpoint-a/` và `checkpoint-b/` luân phiên, nên lúc nào cũng còn checkpoint ngay trước bản mới nhất. Cùng commit với checkpoint là `resume.json` (ô mới nhất, step) và `progress.json` (step, token, snapshot gần nhất, giờ đẩy): squash làm mất tên các commit cũ, nên theo dõi train bằng file này.
  - Resume từ Hub (`fetch_resume_checkpoint`) nạp ô mới nhất; nếu ô đó hỏng (trainer state không đọc được, safetensors không đọc được, file torch sai CRC) thì lùi sang ô kia. Repo cũ chỉ có `last-checkpoint/` vẫn resume được.
  - Run xong thì xóa cả hai ô và `resume.json`; giữ `progress.json`.
  - Checkpoint resume tải ra ngoài `output_dir`.
  - Cuối run chỉ đẩy `metrics*.json`, `config.json`, `order.json`, thay cho `trainer.push_to_hub`; `metrics.json` có thêm `completed`.
  - Khóa config `hub_private`: mặc định true; E1/E2/E3 đặt false.
- `run_checkpoints --results-repo`: tải `<run>/` từ repo kết quả trước, nên runtime mới cũng bỏ qua snapshot đã chấm; đẩy kết quả từng snapshot ngay sau khi chấm.
- Notebook `07_main`: một cell làm cả train (tự resume hoặc bỏ qua nếu đã xong), chấm test mọi snapshot, perplexity và KL của snapshot cuối, rồi đẩy kết quả. Notebook `06_pilot` cũng đẩy kết quả vào repo private.
- Cell cài thư viện của mọi notebook gỡ `torchao`: Colab cài sẵn bản 0.10, khiến `peft` từ chối tạo adapter LoRA (3 test LoRA hỏng khi chạy `05_eval`).

**Ước tính sau khi sửa:** mỗi repo run chính chỉ còn snapshot (E2 khoảng 15 GB, E1 khoảng 26 GB), thêm khoảng 8.6 GB cho hai ô checkpoint trong lúc train. Dung lượng private còn khoảng 2.4 GB cho dataset cộng vài chục MB kết quả mỗi run.

**Chưa kiểm trên GPU:** đẩy và squash checkpoint thật, tốc độ upload, LoRA trên T4. Cách kiểm đề xuất: chạy cell 6 của `07_main` với E2 seed 0, xem commit squash sau checkpoint đầu (step 50).

## 2026-10-27 — Giả thuyết cho E2 seed 0 (ghi trước khi chạy)

**Run:** `configs/e2_mix.yaml`, seed 0, chấm test ở 10 snapshot (25k, 50k, 100k, 200k, 400k, 800k, 1.6M, 3.2M, 6.4M token assistant và step cuối). Một seed nên mọi kết luận ở đây chỉ là đọc điểm, kèm CI bootstrap trong seed. Kết luận về H1–H3 đợi đủ 3 seed.

**Giả thuyết run này kiểm:** H1, H2, H3 (PLAN §1). H5 cần E3; H6 cần phân tích ΔW sau. H4 đã bị E0 bác bỏ: base few-shot có `format_valid` gần 0 trên cả dev và test.

**Dự đoán** (dựa trên pilot E2 10% ở lr 2e-5 và 5e-5; "xuất hiện" là t_reach với θ chính, trên test, T-seen):
1. **Thứ tự:** `stop_rate` và `format_valid` xuất hiện trước, ở mốc 100k hoặc 200k. Sau đó tới `grounded_correct` và `decision_bacc` (khoảng 200k). Rồi `call_exact` (400k–1.6M). `clarify_correct` xuất hiện muộn nhất, có thể không đạt 0.5 tới cuối run.
2. **H1** (format sớm hơn `call_exact` ít nhất một checkpoint): đúng, cách nhau ít nhất 2 checkpoint.
3. **H2, phần "`decision_bacc` muộn nhất":** sai. Ở pilot, `decision_bacc` đạt 0.80 ở khoảng 200k, trước `call_exact` và `clarify_correct`.
4. **H2, phần "`false_call_rate` vượt base rồi giảm":** đúng. Đỉnh ở 100k–200k token (pilot: 0.67–0.70), rồi giảm xuống dưới 0.3 ở cuối run.
5. **H3:** `call_exact` T-unseen cao hơn base few-shot (CI loại trừ 0); đúng, nhưng không đọc là tổng quát hóa vì T-unseen dễ hơn (hạn chế đã biết). Mức giảm T-rename vô nghĩa so với T-seen lớn hơn 10 điểm (pilot: 13 điểm), nên tiêu chí ≤ 10 điểm của H3 không đạt.
6. **Quên (mô tả, cho H5 về sau):** perplexity wikitext-2 cuối run trong khoảng 13.5–14.5 (base 13.01), KL tới base 0.03–0.10, `stop_rate` C ≥ 0.85.

**Kết quả coi là bất ngờ, phải kiểm hạ tầng trước khi đọc:** một metric đã đạt θ rồi rơi về gần 0; val loss tool tăng liên tục; `stop_rate` C dưới 0.5 ở cuối run; `call_exact` T-seen cuối run thấp hơn mức pilot 5e-5 ở 672k (0.55 trên dev).

## 2026-10-27 — Kết quả E2 seed 0 (test)

**Chạy:** `configs/e2_mix.yaml`, seed 0, notebook `07_main`. Step 1–50 chạy trên Colab T4 (bundle `75bdbd01…`, commit `88232e5`); vì hạn mức GPU Colab không đủ một run, run được chuyển sang Kaggle (Colab nối vào Kaggle Jupyter Server, T4, chỉ dùng một GPU; bundle `573a986e…`) và resume từ checkpoint step 50 trên Hub. Phiên Kaggle bị tắt một lần lúc đang chấm snapshot cuối (hộp "Are you still there?" hiện khi mở lại tab Kaggle); bật phiên mới và chạy lại cell 6, nó bỏ qua train và 9 snapshot đã chấm.
- Train: 1250 step, 6 527 256 token assistant (khớp số tính trước), khoảng 9.2 giây/step trên T4 Kaggle, GPU đỉnh 11.1 GiB. Không lần đẩy checkpoint nào lỗi; hai ô checkpoint đã được xóa khi run xong.
- Chấm test: khoảng 15–17 phút mỗi snapshot (2010 prompt).
- Trọng số: `trantam31/smollm2-360m-e2-mix-s0` (public). Output và metric: `trantam31/smollm2-360m-results`, thư mục `e2_mix_s0/` (private). CI: `python -m src.eval.stats --split test --baseline e0_test/base-zero-shot` → `results/{all,diff_base,emergence}_test.csv`.

**Snapshot cuối (step 1250), test, CI bootstrap 95% theo cụm:**

| | T-seen | T-unseen |
|---|---|---|
| `format_valid` | 1.00 | |
| `call_exact` | 0.66 [0.61, 0.72] | 0.80 [0.74, 0.86] |
| `call_exact` T-rename vô nghĩa / đồng nghĩa | 0.57 [0.51, 0.63] / 0.64 [0.58, 0.69] | |
| `decision_bacc` | 0.995 [0.98, 1.00] | |
| `false_call_rate` | 0.01 [0.00, 0.03] | |
| `clarify_correct` | 0.62 [0.49, 0.73] | 0.61 [0.40, 0.84] |
| `grounded_correct` | 0.97 [0.94, 1.00] | |
| `stop_rate` C | 0.93 [0.89, 0.97] | |

Base zero-shot và few-shot đều bằng 0 ở mọi metric trên (trừ `decision_bacc` = 0.5). Perplexity wikitext-2 14.12 (base 13.01), KL tới base 0.068 nat/token.

**t_reach (θ chính, T-seen; token assistant):** `decision_bacc` 104k → `stop_rate`, `format_valid`, `tool_name_acc`, `grounded_correct` 200k → `call_exact` 402k → `clarify_correct` 3.2M. T-unseen giống vậy, trừ `call_exact` đạt từ 200k và `decision_bacc` từ 200k. t_improve đạt ở cùng mốc hoặc sớm hơn (`clarify_correct` 402k).

**`false_call_rate` T-seen theo checkpoint:** 0, 0, 0.38 [0.29, 0.47] (104k), 0.30, 0.27, 0.17, 0.02, 0.04, 0.01, 0.01.

**Đối chiếu với dự đoán ghi trước khi chạy** (một seed, chưa phải kết luận về H1–H3):
1. Thứ tự: đúng phần lớn. Stop/format và grounded ở 200k, `call_exact` ở 400k, `clarify_correct` muộn nhất. Khác dự đoán: `decision_bacc` đến sớm nhất (104k), không phải cùng lúc grounded; `clarify_correct` có đạt 0.5 (ở 3.2M).
2. H1 (format trước `call_exact` ít nhất một checkpoint): đạt với θ chính trên T-seen nhưng chỉ cách một checkpoint, không phải hai như dự đoán. Không vững theo PLAN §3.5: ở θ = 0.3, `call_exact` cũng đạt ở 200k; trên T-unseen, `call_exact` đạt 0.5 cùng mốc với format.
3. H2 phần "`decision_bacc` muộn nhất": sai, như dự đoán (nó đến sớm nhất).
4. H2 phần "`false_call_rate` vượt base rồi giảm": đúng, như dự đoán. Đỉnh 0.38 ở 104k (CI loại trừ 0, base là 0), xuống 0.01 cuối run. Đỉnh thấp hơn pilot (0.67–0.70).
5. H3: `call_exact` T-unseen cao hơn base few-shot, CI loại trừ 0 (như dự đoán; vẫn không đọc là tổng quát hóa vì T-unseen dễ hơn). Mức giảm T-rename vô nghĩa so với T-seen khoảng 9 điểm (0.66 → 0.57, so điểm chưa ghép cặp); dự đoán là trên 10 điểm. Cần tính hiệu số ghép cặp trên đúng các prompt gốc kèm CI trước khi đọc tiêu chí ≤ 10 điểm và cận trên < 20 điểm.
6. Quên: perplexity 14.12, KL 0.068, `stop_rate` C 0.93, đều trong khoảng dự đoán.
- Không có dấu hiệu "bất ngờ" nào đã ghi (không metric nào rơi về 0 sau khi đạt; `call_exact` cuối 0.66 cao hơn mức pilot 0.55 trên dev).

**Việc còn lại cho run này:** hiệu số ghép cặp T-rename − T-seen kèm CI (H3); đọc output bộ P theo checkpoint (PLAN Giai đoạn 3).

## 2026-10-28 — E2 seed 0: hiệu số ghép cặp T-rename − T-seen (H3)

**Loại hoạt động:** phân tích thêm trên output đã chấm của E2 seed 0, không sinh lại, không đổi metric hay θ.

**Phép tính:** `src/eval/stats.py` có thêm `rename_differences`, ghi `results/rename_test.csv` (git-ignore). Mỗi prompt T-rename được ghép với đúng prompt T-seen gốc của nó (`renamed_from`), cùng checkpoint. Hai bên dùng chung các lần rút cụm của bộ T-rename (cụm theo quan hệ của mẫu gốc, như PLAN §3.5), nên CI là CI của chính hiệu số. Tính cho `format_valid`, `call_exact`, `tool_name_acc`, `args_f1`, trên cả ba nhóm gộp và từng nhóm. 300 mẫu T-rename mỗi mức phủ đúng 300 mẫu cần tool của T-seen, nên điểm T-seen ở đây trùng với điểm T-seen ở mục kết quả; cái mới là CI của hiệu số.

**`call_exact` cuối run (step 1250), ba nhóm gộp:** T-seen 0.663.

| | T-rename | Mức giảm so với T-seen [CI 95%] |
|---|---|---|
| Vô nghĩa | 0.573 | 9.0 điểm [4.5, 13.8] |
| Đồng nghĩa | 0.637 | 2.7 điểm [−0.7, 6.2] |

- **Tiêu chí H3 (mức vô nghĩa):** giảm ≤ 10 điểm và cận trên CI < 20 điểm. Cả hai đạt ở snapshot cuối (một seed). Dự đoán ghi trước ("giảm trên 10 điểm, tiêu chí không đạt") sai, dù sát ngưỡng.
- **Theo checkpoint (vô nghĩa):** 0 tới 104k; rồi 3.0, 7.3, 8.3, 10.7, 11.0, 9.0, 9.0 điểm (200k → cuối). Ở 1.6M và 3.2M mức giảm vượt 10 điểm; cận trên CI cao nhất là 15.8 điểm, luôn dưới 20. Cận dưới CI > 0 từ 402k, tức tên vô nghĩa làm giảm điểm thật, chỉ là ít.
- **Mức đồng nghĩa:** giảm 0.7–2.7 điểm từ 402k, CI luôn chứa 0.
- **Giảm ở đâu (vô nghĩa, cuối run):** `tool_name_acc` giảm 6.7 điểm [3.3, 10.3], nhiều nhất ở Parallel (14 điểm [5.2, 22.7]); `args_f1` giảm 9.3 điểm [5.9, 12.8]; `format_valid` giảm 1.7 điểm. Như vậy tên vô nghĩa làm model chọn sai tool và chép sai tên tham số nhiều hơn, chứ không làm hỏng format.
- **Đọc:** như PLAN §4.4, đây là bằng chứng hỗ trợ việc dùng schema và cho thấy mức phụ thuộc vào tên tool khoảng 9 điểm, không phân định "đọc schema" với "nhớ tên". Kết luận về H3 đợi đủ 3 seed.

## 2026-10-28 — Giả thuyết cho E1 seed 0 (ghi trước khi chạy)

**Run:** `configs/e1_seq.yaml`, seed 0: D_chat × 2 epoch (step 1–625) rồi D_tool × 2 epoch (step 626–1250), một lịch lr, optimizer liên tục. Chấm test ở 18 snapshot (mục "Run chính: config, LoRA, lịch snapshot của E1"): thang tích lũy tới 3.2M token, ranh giới (step 625), thang từ ranh giới 25k → 1.6M token, rồi 6.4M và step cuối. Một seed nên mọi điều ở đây chỉ là đọc điểm kèm CI trong seed.

**Giả thuyết run này kiểm:** H1, H2, H3 trên E1; và câu hỏi của PLAN §5.1 "học chat có tự sinh ra chút khả năng tool nào không" (snapshot ranh giới). So E1 với E2 là mô tả ở một seed; chưa có phép hiệu số ghép cặp giữa hai run (`stats.py` chỉ so với base), nên so cuối run dưới đây là so điểm.

**Căn cứ:**
- E2 seed 0, đổi trục sang token tool (D_tool chiếm khoảng 29.4% token assistant; tính gần đúng, chưa dùng `order.json`): `decision_bacc` đạt 0.77 ở khoảng 31k token tool; `format_valid` 0.96 ở 59k; `call_exact` 0.55 ở 118k; `clarify_correct` 0.56 ở khoảng 940k.
- lr của E1 ở ranh giới đã còn khoảng 52% đỉnh (warmup 37 step, cosine), và lr trung bình của giai đoạn tool chỉ khoảng 19% đỉnh. Ở E2, kỹ năng tool hình thành trong 400k token đầu, lúc lr gần đỉnh. Nên tính theo token tool, E1 dự kiến học tool chậm hơn E2.

**Dự đoán** ("xuất hiện" là t_reach với θ chính, trên test, T-seen; "token từ ranh giới" là thang thứ hai):
1. **Giai đoạn chat (tới ranh giới):**
   - `stop_rate` C đạt 0.5 ở mốc 100k hoặc 200k và ≥ 0.85 ở ranh giới. `stop_rate` T-seen ở ranh giới ≥ 0.7: model chỉ học chat vẫn trả lời bằng văn xuôi rồi dừng.
   - Ranh giới (chỉ học chat): `format_valid` < 0.05, `call_exact` = 0 (CI chứa 0 khi so với base), `decision_bacc` ≈ 0.5, `false_call_rate` ≈ 0. Tức học chat **không** tự sinh khả năng gọi tool.
   - `clarify_correct` ở ranh giới ≤ 0.1. `grounded_correct` ở ranh giới là điểm ít chắc nhất: model chat có thể chép giá trị từ kết quả tool đã có trong ngữ cảnh. Đoán 0.1–0.5 (360M-Instruct: 0.17).
2. **Giai đoạn tool, thứ tự:** `decision_bacc` và `format_valid` đạt ở 25k–100k token từ ranh giới; `call_exact` ở 100k–400k; `clarify_correct` muộn nhất, đạt 0.5 ở 800k token từ ranh giới trở đi, hoặc không đạt tới cuối run.
3. **Tốc độ so với E2 (trục token tool):** `call_exact` đạt 0.5 muộn hơn E2 (118k), ở khoảng 200k–400k token từ ranh giới, do lr thấp hơn.
4. **H1:** `stop_rate` đạt trong giai đoạn chat, trước `call_exact` nhiều checkpoint. Với `format_valid` so với `call_exact` thì đạt, cách ít nhất một checkpoint của thang thứ hai. Có vững ở cả ba θ không thì chưa chắc, như ở E2.
5. **H2:** "`decision_bacc` muộn nhất" sai, như E2. "`false_call_rate` vượt base rồi giảm" đúng: đỉnh ≥ 0.2 ở 25k–200k token từ ranh giới (CI loại trừ 0), xuống < 0.05 ở cuối run.
6. **H3:** `call_exact` T-unseen cao hơn base few-shot (CI loại trừ 0). Mức giảm ghép cặp T-rename vô nghĩa ở cuối run trong khoảng 5–15 điểm; mức đồng nghĩa < 5 điểm.
7. **Cuối run so với E2 seed 0 (so điểm):**
   - `call_exact` T-seen trong khoảng 0.58–0.70 (E2: 0.66). Lr thấp ở giai đoạn tool kéo xuống, còn việc tool là dữ liệu cuối cùng (không bị chat chen giữa) kéo lên. Đoán hai yếu tố gần bù nhau.
   - `clarify_correct` T-seen thấp hơn hoặc bằng E2 (0.62).
   - Quên: perplexity wikitext-2 trong khoảng 13.6–14.8 (E2: 14.12), KL tới base 0.04–0.12; `stop_rate` C ≥ 0.85 (D_tool có 20% No-tool trả lời văn xuôi, nên model không quên cách dừng).

**Kết quả coi là bất ngờ, phải kiểm hạ tầng trước khi đọc:**
- `call_exact` > 0.05 ở ranh giới: kiểm D_chat có lẫn mẫu tool không, và kiểm thứ tự trong `order.json`.
- Mọi metric kỹ năng rơi về gần 0 ngay sau ranh giới.
- `stop_rate` C < 0.5 ở cuối run.
- `call_exact` T-seen cuối run < 0.5.
- Số step hay token khác lịch đã tính (ranh giới phải ở step 625).

**Chạy:** trên Kaggle như E2. Sau khi nối Colab thì đóng hẳn tab Kaggle, để kiểm xem phiên còn sống không (CLAUDE.md). Ước tính khoảng 3.2 giờ train cộng khoảng 4.5 giờ chấm 18 snapshot, tức khoảng 7.7 giờ, nằm trong giới hạn 12 giờ của phiên. Hạn mức tuần sau run này khoảng 14/30 giờ.

## 2026-10-28 — Kết quả E1 seed 0 (test)

**Chạy:** `configs/e1_seq.yaml`, seed 0, notebook `07_main` (bundle `51099f04…`, commit `e447030`), Colab nối Kaggle Jupyter Server, T4, một GPU, tab Kaggle đóng suốt run. Phiên Kaggle bị hủy hai lần, cả hai lần khoảng 35–50 phút sau khi tab Colab mất kết nối runtime vì mạng phía máy người dùng: lần đầu sau checkpoint step 950 (resume từ đó, mất dưới 50 step), lần hai lúc đang chấm snapshot step 7 (chạy lại cell 6, nó bỏ qua train và các snapshot đã chấm). Chi tiết ở CLAUDE.md.
- Train: 1250 step, 6 527 256 token assistant; ranh giới chat → tool ở step 625 (4.61M token), đúng lịch. Khoảng 8.4 giây/step.
- Chấm test 18 snapshot, khoảng 14–17 phút mỗi snapshot. Hạn mức Kaggle cho run này khoảng 9 giờ (tính cả phần chạy lại).
- Trọng số: `trantam31/smollm2-360m-e1-seq-s0` (public). Output và metric: `trantam31/smollm2-360m-results`, thư mục `e1_seq_s0/` (private).

**Snapshot cuối (step 1250), test, CI bootstrap 95% theo cụm** (E2 seed 0 trong ngoặc, so điểm):

| | T-seen | T-unseen |
|---|---|---|
| `format_valid` | 1.00 (1.00) | |
| `call_exact` | 0.65 [0.59, 0.70] (0.66) | 0.81 [0.76, 0.87] (0.80) |
| `decision_bacc` | 0.99 [0.98, 1.00] (0.995) | |
| `false_call_rate` | 0.01 [0.00, 0.03] (0.01) | |
| `clarify_correct` | 0.50 [0.36, 0.62] (0.62) | |
| `grounded_correct` | 0.94 [0.89, 0.99] (0.97) | |
| `stop_rate` C | 0.94 [0.90, 0.98] (0.93) | |

Perplexity wikitext-2 14.10 (E2: 14.12; base 13.01), KL tới base 0.067 nat/token (E2: 0.068).

**Ranh giới (step 625, chỉ học chat):** `format_valid`, `tool_name_acc`, `call_exact` đều bằng 0 trên T-seen và T-unseen, và hiệu số so với base là 0 với CI [0, 0]; `decision_bacc` 0.5, `false_call_rate` 0. `stop_rate` C 0.90, T-seen 0.49 (Single 0.48, Select 0.30, Parallel 0.24, No-tool 0.59, Missing-info 0.36, Multi-turn 0.99). `clarify_correct` 0.01. `grounded_correct` 0.50 [0.34, 0.70].

**Giai đoạn tool, T-seen, theo token từ ranh giới** (25k, 50k, 100k, 200k, 400k, 800k, 1.6M):
- `format_valid` 0.87, 0.96, 0.98, 0.98, 0.98, 1.00, 1.00.
- `call_exact` 0.32, 0.47, 0.55, 0.57, 0.59, 0.64, 0.65.
- `decision_bacc` 0.68, 0.71, 0.88, 0.90, 0.94, 0.97, 0.99.
- `false_call_rate` 0.64 [0.54, 0.73], 0.58, 0.23, 0.19, 0.11, 0.05, 0.01.
- `clarify_correct` 0.00, 0.03, 0.10, 0.31, 0.41, 0.45, 0.51.

**t_reach (θ chính, T-seen):** `stop_rate` 404k (giai đoạn chat) → `grounded_correct` ranh giới → `format_valid`, `tool_name_acc` 25k từ ranh giới → `decision_bacc`, `call_exact` 100k → `clarify_correct` chỉ đạt ở checkpoint cuối ("chưa xác nhận"; T-unseen đạt ở 1.6M).

**Một điều thấy được ở giai đoạn chat (không nằm trong dự đoán):** `grounded_correct` T-seen là 0.81–0.83 ở 404k–800k, rơi xuống 0.24–0.26 ở 1.6M–3.2M, rồi 0.50 ở ranh giới. Không phải lỗi hạ tầng: D_chat không có `<tool_call>` nào, và ở cả bốn snapshot không mẫu Single/Select/Parallel nào sinh `<tool_call>`. Nhưng ở Multi-turn, nơi ngữ cảnh đã có một lượt assistant gọi tool, model chỉ học chat **chép lại lời gọi tool** thay vì trả lời bằng kết quả: tỷ lệ output có `<tool_call>` là 0.01 ở 800k, 0.65 ở 1.6M, 0.62 ở 3.2M, 0.32 ở ranh giới. Tức model chat bắt chước định dạng gọi tool có sẵn trong ngữ cảnh, dù chưa học gọi tool khi chỉ có schema.

**Đối chiếu với dự đoán ghi trước khi chạy** (một seed):
1. Giai đoạn chat: phần lớn đúng, hai chỗ sai.
   - Đúng: ở ranh giới, các metric gọi tool bằng 0 (học chat không tự sinh khả năng gọi tool); `stop_rate` C 0.90 ≥ 0.85; `clarify_correct` ≤ 0.1; `grounded_correct` 0.50, ở mép trên khoảng 0.1–0.5 đã đoán (nhưng không đơn điệu, xem trên).
   - Sai: `stop_rate` C đạt 0.5 ở 404k, muộn hơn mốc 100k–200k đã đoán. `stop_rate` T-seen ở ranh giới 0.49, dưới mức ≥ 0.7 đã đoán: khi system prompt có schema tool, model chỉ học chat dừng kém hơn hẳn, nhất là Select và Parallel.
2. Thứ tự ở giai đoạn tool: đúng. `format_valid` 25k, `decision_bacc` 100k (trong khoảng 25k–100k), `call_exact` 100k (mép dưới của 100k–400k), `clarify_correct` muộn nhất.
3. Tốc độ so với E2 theo token tool: sai. Dự đoán E1 chậm hơn (`call_exact` đạt 0.5 ở 200k–400k); thực tế đạt ở 100k token tool, so với khoảng 118k của E2 (thang snapshot của E2 thưa, 59k rồi 118k, nên chỉ nói được là E1 không chậm hơn). lr thấp hơn ở giai đoạn tool không làm chậm việc học tool.
4. H1: `stop_rate` đạt ở giai đoạn chat, trước `call_exact` 7 checkpoint. `format_valid` trước `call_exact` hai checkpoint (25k so với 100k) ở θ chính, nhưng không vững theo PLAN §3.5: ở θ = 0.3 `call_exact` cũng đạt ở 25k. Trên T-unseen chỉ cách một checkpoint. Giống E2.
5. H2: "`decision_bacc` muộn nhất" sai (cùng checkpoint với `call_exact`, trước `clarify_correct`). "`false_call_rate` vượt base rồi giảm" đúng: đỉnh 0.64 [0.54, 0.73] ở 25k từ ranh giới, cao hơn đỉnh của E2 (0.38), xuống 0.01 ở cuối.
6. H3: đúng. `call_exact` T-unseen 0.81, base few-shot 0. Mức giảm ghép cặp ở T-rename vô nghĩa là 8.3 điểm [3.8, 13.1], trong khoảng 5–15 và đạt tiêu chí H3; mức đồng nghĩa 3.0 điểm [−0.7, 6.8], dưới 5.
7. Cuối run so với E2: đúng. `call_exact` 0.65 trong khoảng 0.58–0.70; `clarify_correct` 0.50, thấp hơn E2 (0.62); perplexity 14.10, KL 0.067, `stop_rate` C 0.94 đều trong khoảng đã đoán. Chưa có hiệu số ghép cặp giữa hai run nên chưa nói được chênh lệch nào là thật.
- Không có dấu hiệu "bất ngờ" nào đã ghi: `call_exact` ở ranh giới là 0; không metric nào sụp sau ranh giới; `stop_rate` C cuối 0.94; `call_exact` cuối 0.65; ranh giới đúng step 625.

**Việc còn lại:** hiệu số ghép cặp giữa hai run (E1 − E2) trên cùng prompt, kèm CI; đọc output bộ P theo checkpoint.

## 2026-10-29 — E1 − E2 seed 0: hiệu số ghép cặp giữa hai run

**Loại hoạt động:** phân tích thêm trên output đã chấm của E1 và E2 seed 0. Không sinh lại, không đổi metric hay θ.

**Phép tính:** `src/eval/stats.py` có thêm `run_differences`. Lệnh `python -m src.eval.stats --split test --baseline e0_test/base-zero-shot --compare e1_seq_s0 e2_mix_s0` ghi `results/compare_e1_seq_s0_e2_mix_s0_test.csv` (git-ignore).
- Hai run được ghép ở những snapshot cả hai cùng có: cùng mốc token assistant cộng dồn (`target_tokens` 25k → 6.4M) và step cuối. Snapshot ranh giới của E1 và thang tính từ ranh giới không có cặp, nên không vào phép so này.
- Hai bên dùng chung các lần rút cụm của từng tập (cũng là các lần rút của `all.csv`), nên CI là CI của chính hiệu số, theo cặp prompt (PLAN §3.5, "giữa hai run").
- Một seed nên đây là bất định do tập prompt, chưa có biến thiên giữa các seed.

**Snapshot cuối (step 1250), E1 − E2, điểm [CI 95%]:**

| | T-seen | T-unseen |
|---|---|---|
| `format_valid` | −0.3 [−1.0, 0.0] | |
| `call_exact` | −1.3 [−3.8, +1.0] | +1.3 [−0.6, +3.6] |
| `tool_name_acc` | −0.3 [−1.8, +1.0] | |
| `args_f1` | −1.4 [−2.9, −0.1] | |
| `decision_bacc` | −0.2 [−1.7, +1.3] | |
| `false_call_rate` | 0.0 [−2.9, +3.0] | |
| `clarify_correct` | **−12.0 [−18.9, −6.2]** | **−9.0 [−14.7, −3.8]** |
| `grounded_correct` | −3.0 [−6.9, 0.0] | +3.0 [0.0, +8.6] |
| `call_exact` T-rename vô nghĩa / đồng nghĩa | −0.7 [−4.0, +2.4] / −1.7 [−4.4, +0.7] | |
| `stop_rate` C | +0.7 [−4.5, +5.6] | |

Snapshot 6.4M cho cùng bức tranh: `clarify_correct` −14.0 [−21.8, −7.4] (T-seen) và −9.0 [−14.7, −3.8] (T-unseen); các metric khác có CI chứa 0 hoặc lệch dưới 2 điểm.

**Đọc (một seed):**
- Cuối run, thứ tự dữ liệu không làm đổi điểm gọi tool. Hiệu số `call_exact`, `tool_name_acc`, `decision_bacc`, `false_call_rate` đều nhỏ hơn 2 điểm, CI chứa 0. `args_f1` T-seen thấp hơn 1.4 điểm, CI vừa loại trừ 0; mức này nhỏ, và với hàng trăm ô được so thì nên đọc là chưa chắc.
- Mức phụ thuộc vào tên tool cũng như nhau: hiệu số T-rename giữa hai run có CI chứa 0.
- Khác biệt duy nhất rõ và lặp lại là **hỏi lại khi thiếu thông tin**: E1 thấp hơn E2 khoảng 9–14 điểm, CI loại trừ 0 trên cả T-seen và T-unseen, ở cả 6.4M và step cuối. Ở mục kết quả E1, chênh lệch so điểm 0.50 so với 0.62 nay là hiệu số thật trong seed này. Không giống E1 chỉ chậm hơn: `clarify_correct` T-seen của E1 gần như đứng ở 0.45–0.51 từ 800k token tính từ ranh giới tới cuối run (0.45, 0.51, 0.48, 0.50), trong khi E2 lên 0.56 ở 3.2M token cộng dồn (khoảng 940k token tool) và 0.62 ở cuối. Cần seed 1–2 để biết hiệu số này có giữ qua seed không.
- Các snapshot giữa run (25k → 3.2M token cộng dồn) chỉ cho thấy điều đã biết: E1 lúc đó đang ở giai đoạn chat nên mọi metric gọi tool bằng 0, còn E2 đã học tool. Phép so theo trục token tool (E1 tính từ ranh giới so với E2 ở cùng lượng token tool) không làm được theo cặp, vì hai run không có snapshot ở cùng lượng token tool.

## 2026-10-29 — Giả thuyết cho E3 LoRA r64 seed 0 (ghi trước khi chạy)

**Run:** `configs/e3_lora_r64.yaml`, seed 0. Dữ liệu, số epoch, lịch lr (warmup 3% + cosine) và lịch snapshot như E2. Khác E2 ở hai điểm: chỉ train adapter LoRA r = 64 (alpha 128, dropout 0.05, gắn vào q/k/v/o/gate/up/down của 32 layer, base và embedding đứng yên), và lr 2e-4 thay vì 5e-5. Thứ tự mẫu sinh từ `seed` và `order` (`src/train_data.py`), nên E3 seed 0 học đúng thứ tự mẫu của E2 seed 0, và các snapshot rơi vào cùng step. Vì vậy E3 − E2 được ghép cặp ở cả 10 snapshot (`--compare e3_lora_r64_s0 e2_mix_s0`). Chênh lệch giữa hai run là do LoRA cộng với lr khác nhau, không tách được hai yếu tố này.

Adapter có khoảng 34.7 triệu tham số train được (khoảng 1.09 triệu mỗi layer), cỡ 10% model. Một seed nên mọi điều ở đây chỉ là đọc điểm kèm CI trong seed.

**Giả thuyết run này kiểm:** H5 (PLAN §1), ở một seed; H1–H3 trên E3.
- Phần "LoRA quên ít hơn" của H5: hiện chỉ có perplexity wikitext-2 và KL tới base ở snapshot cuối, mỗi thứ một con số, chưa có CI. ARC-Easy/HellaSwag (PLAN §3.3) chưa chạy cho run chính nào. Nên ở run này, phần quên chỉ là so điểm.
- Phần "LoRA kém không quá 5 điểm": PLAN đọc nó theo trung bình 3 seed. Ở đây báo hiệu số ghép cặp `call_exact` E3 − E2 kèm CI.

**Căn cứ:**
- E2 seed 0: `call_exact` T-seen 0.66, T-unseen 0.80; `clarify_correct` 0.62; perplexity 14.12, KL 0.068. t_reach: `decision_bacc` 104k, format 200k, `call_exact` 402k, `clarify_correct` 3.2M.
- Adapter gần 35 triệu tham số, lr 2e-4: đủ sức chứa cho một kỹ năng hẹp như định dạng gọi tool. Base và embedding đứng yên nên model ít lệch khỏi base hơn.
- Chưa có pilot LoRA nào, và LoRA chưa từng chạy trên GPU. Nên các khoảng dưới đây rộng hơn ở E1.

**Dự đoán** ("xuất hiện" là t_reach với θ chính, trên test, T-seen):
1. **Thứ tự** như E2: `decision_bacc` và format sớm, rồi `call_exact`, `clarify_correct` muộn nhất. Mỗi mốc lệch E2 không quá một checkpoint, sớm hay muộn đều được (lr cao hơn bù cho việc ít tham số hơn).
2. **H1:** đạt ở θ chính, không vững qua cả ba θ, như E1 và E2.
3. **H2:** "`decision_bacc` muộn nhất" sai. "`false_call_rate` vượt base rồi giảm" đúng: đỉnh ≥ 0.2 ở 50k–400k (CI loại trừ 0), xuống < 0.05 ở cuối run.
4. **H3:** `call_exact` T-unseen cao hơn base few-shot (CI loại trừ 0). Mức giảm ghép cặp khi đổi tên vô nghĩa nằm trong khoảng 5–15 điểm.
5. **H5, khả năng (E3 − E2, ghép cặp, snapshot cuối):** `call_exact` T-seen của E3 trong khoảng 0.55–0.68. Hiệu số so với E2 trong khoảng −8 tới +2 điểm. Đoán điểm là −3, và cận dưới CI có thể vượt quá −5. `clarify_correct` thấp hơn E2 (−15 tới 0 điểm), vì đây là kỹ năng đến muộn nhất, cần học nhiều nhất.
6. **H5, quên (so điểm):** perplexity wikitext-2 trong khoảng 13.3–13.9, thấp hơn E2 (14.12); KL tới base 0.02–0.05, thấp hơn E2 (0.068). `stop_rate` C ≥ 0.85.
7. **Tài nguyên:** bộ nhớ GPU đỉnh thấp hơn E2 (11.1 GiB), tốc độ mỗi step trong khoảng ±20% so với E2 (9.2 giây/step trên Kaggle), vì vẫn phải lan truyền ngược qua toàn bộ base.

**Kết quả coi là bất ngờ, phải kiểm hạ tầng trước khi đọc:**
- Loss NaN hoặc `grad_norm` vọt lên (fp16 AMP với adapter, lr 2e-4).
- Val loss tool ở step cuối cao hơn của E2 ở snapshot 400k.
- Metric của snapshot đã merge khác hẳn đường val loss lúc train (lỗi merge).
- `call_exact` T-seen cuối run < 0.4.
- Perplexity cao hơn E2.
- Số step hay token khác E2 (phải là 1250 step, 6 527 256 token assistant).

**Chạy:** trên Kaggle, notebook `07_main`, cell 6 với `CONFIG, SEED = "e3_lora_r64", 0`. Ước tính khoảng 3 giờ train cộng khoảng 2.7 giờ chấm 10 snapshot, tức khoảng 6 giờ, trong giới hạn 12 giờ của phiên. Hạn mức tuần đang dùng khoảng 15/30 giờ. Giữ tab Colab nối runtime suốt run (CLAUDE.md).

## 2026-10-29 — Kết quả E3 LoRA r64 seed 0 (test)

**Chạy:** `configs/e3_lora_r64.yaml`, seed 0, notebook `07_main` (bundle `6bd5ec75…`, commit `0af0c93`). Colab nối Kaggle Jupyter Server, T4, một GPU, tab Kaggle đóng suốt run. Tab Colab nối runtime liên tục, và phiên Kaggle không bị hủy lần nào (khoảng 6.9 giờ).
- Train: 1250 step, 6 527 256 token assistant, đúng như E2. Snapshot ở cùng step với E2 (5, 9, 19, 38, 77, 155, 307, 614, 1226, 1250), và số token ở từng snapshot khớp E2 đến từng token: E3 học đúng thứ tự mẫu của E2.
- Adapter có 34 734 080 tham số train được. Train mất khoảng 3.9 giờ (khoảng 10.8 giây/step, gồm cả thời gian merge và lưu snapshot), GPU đỉnh 8.8 GiB. Không lần đẩy checkpoint nào lỗi.
- Val loss ở step cuối: tool seen 0.232 (E2: 0.228), tool unseen 0.232 (0.231), chat 0.939 (0.937).
- Chấm test 10 snapshot, khoảng 15–17 phút mỗi snapshot.
- Trọng số: `trantam31/smollm2-360m-e3-lora-r64-s0` (public, snapshot đã merge). Output và metric: `trantam31/smollm2-360m-results`, thư mục `e3_lora_r64_s0/` (private).
- So với E2: `python -m src.eval.stats --split test --baseline e0_test/base-zero-shot --compare e3_lora_r64_s0 e2_mix_s0`.

**Snapshot cuối (step 1250), test, CI bootstrap 95% theo cụm; cột phải là hiệu số ghép cặp E3 − E2 (điểm):**

| | E3 T-seen | E3 − E2 T-seen | E3 T-unseen | E3 − E2 T-unseen |
|---|---|---|---|---|
| `format_valid` | 0.99 [0.98, 1.00] | −0.7 [−2, 0] | | |
| `call_exact` | 0.64 [0.59, 0.69] | −2.3 [−4.9, 0.0] | 0.81 [0.75, 0.86] | +1.0 [−1.1, +3.3] |
| `decision_bacc` | 1.00 | +0.5 [0, +2] | | |
| `false_call_rate` | 0.00 | −1.0 [−3, 0] | | |
| `clarify_correct` | 0.59 [0.47, 0.70] | −3.0 [−9.9, +4.4] | 0.61 [0.40, 0.84] | 0.0 [−4.6, +4.8] |
| `grounded_correct` | 0.96 [0.93, 1.00] | −1.0 [−2.5, 0.0] | 0.96 | +1.0 [0.0, +3.8] |
| `call_exact` T-rename vô nghĩa / đồng nghĩa | 0.56 / 0.61 | −1.3 [−4, +2] / −2.3 [−4.7, −0.3] | | |
| `stop_rate` C | 0.89 [0.84, 0.94] | −4.0 [−9.5, +1.4] | | |

Perplexity wikitext-2 13.63 (E2: 14.12; base 13.01), KL tới base 0.043 nat/token (E2: 0.068).

**t_reach (θ chính, T-seen; token assistant):** `format_valid`, `stop_rate`, `decision_bacc`, `grounded_correct` 200k → `tool_name_acc` 402k → `call_exact` 803k → `clarify_correct` 3.2M. T-unseen giống vậy, trừ `call_exact` đạt từ 402k. So với E2, mỗi mốc bằng hoặc muộn hơn một checkpoint.

**Giữa run, E3 chậm hơn rồi đuổi kịp** (E3 − E2, T-seen): ở 200k, `format_valid` −16.3 điểm [−21, −12], `call_exact` −15.0 [−20, −10], `tool_name_acc` −24.7 [−31, −18]; ở 400k còn −4 tới −7 điểm; từ 800k các hiệu số này dưới 3 điểm. `false_call_rate` của E3 cao hơn E2 ở 800k và 1.6M (+5 [+1, +10], +9 [+4, +15]), rồi cùng về 0. `stop_rate` C của E3 thấp hơn E2 khoảng 5–8 điểm suốt từ 400k tới 6.4M (CI loại trừ 0 ở vài snapshot), ở cuối −4.0 với CI chứa 0.

**`false_call_rate` T-seen theo checkpoint:** 0, 0, 0.02, 0.37 [0.28, 0.46] (200k), 0.24, 0.22, 0.11, 0.02, 0.00, 0.00.

**Đối chiếu với dự đoán ghi trước khi chạy** (một seed):
1. Thứ tự: đúng. Như E2, mỗi mốc lệch E2 không quá một checkpoint, và lệch đều về phía muộn hơn (`decision_bacc`, `tool_name_acc`, `call_exact`).
2. H1: sai theo hướng tốt hơn. Dự đoán là "đạt ở θ chính, không vững qua ba θ". Thực tế vững ở cả ba θ: `format_valid` đạt ở 200k với cả 0.3, 0.5, 0.7; `call_exact` đạt ở 402k (θ 0.3), 803k (θ 0.5), không đạt 0.7. Trên T-unseen cũng vững (200k so với 402k ở cả ba θ). Đây là run đầu tiên H1 vững theo tiêu chí PLAN §3.5. Một cách đọc: E3 học `call_exact` chậm hơn E2 nên khoảng cách với format giãn ra, chứ không phải format đến sớm hơn.
3. H2: đúng. "`decision_bacc` muộn nhất" sai (nó đạt ở 200k, trước `call_exact`). "`false_call_rate` vượt base rồi giảm" đúng: đỉnh 0.37 [0.28, 0.46] ở 200k, về 0.00 ở cuối.
4. H3: đúng. `call_exact` T-unseen 0.81, base few-shot 0. Mức giảm ghép cặp khi đổi tên vô nghĩa là 8.0 điểm [3.0, 12.9], trong khoảng 5–15 và đạt tiêu chí H3; mức đồng nghĩa 2.7 điểm [−1.0, 6.3].
5. H5, khả năng: đúng. `call_exact` T-seen 0.64, trong khoảng 0.55–0.68. Hiệu số E3 − E2 là −2.3 điểm [−4.9, 0.0], trong khoảng −8 tới +2 đã đoán. Cận dưới CI không vượt quá −5 như dự đoán là "có thể". Ở seed này, LoRA r64 kém full fine-tune không quá 5 điểm. PLAN đọc tiêu chí này theo trung bình 3 seed. `clarify_correct` thấp hơn 3 điểm, trong khoảng −15 tới 0 đã đoán, nhưng CI chứa 0.
6. H5, quên: đúng. Perplexity 13.63 trong khoảng 13.3–13.9, KL 0.043 trong khoảng 0.02–0.05, đều thấp hơn E2. Mới là so điểm ở một seed, chưa có CI, chưa có ARC-Easy/HellaSwag. `stop_rate` C 0.89 ≥ 0.85.
7. Tài nguyên: đúng. GPU đỉnh 8.8 GiB, thấp hơn E2 (11.1 GiB). Tốc độ chậm hơn E2 khoảng 17%, trong khoảng ±20%.
- Không có dấu hiệu "bất ngờ" nào đã ghi: không có NaN; val loss tool cuối run (0.232) thấp hơn hẳn mức của E2 ở 400k; điểm của snapshot đã merge đi cùng chiều với val loss; `call_exact` cuối 0.64; perplexity thấp hơn E2; số step và token khớp E2.

**Việc còn lại:** ARC-Easy/HellaSwag cho E1, E2, E3 ở snapshot cuối (phần "quên" của H5 theo PLAN §3.3); đọc output bộ P theo checkpoint; seed 1–2.

## 2026-10-29 — ARC-Easy và HellaSwag cho base, E1, E2, E3 seed 0 (snapshot cuối)

**Loại hoạt động:** đo thêm ở snapshot cuối của các run đã xong, cho phần "quên" của H5 (PLAN §1, §3.3). Không train lại, không đổi giao thức.

**Cách đo:** `src/eval/benchmarks.py` (lm-evaluation-harness 0.4.13), 500 câu đầu của mỗi bộ, 0-shot, chấm theo log-likelihood, fp16 trên T4 Kaggle. Notebook `07_main` có thêm cell 7 chạy bù cho base và các run đã xong (bundle `c0aae56c…`); từ nay cell 6 cũng chạy bước này trước perplexity/KL. Mỗi model khoảng một phút. File ghi điểm từng câu, nên hai model được so ghép cặp theo câu: `python -m src.eval.benchmarks --compare A.json B.json`, bootstrap 2000 lần trên cùng tập câu cho cả hai bên. Câu được rút độc lập, không theo cụm. Kết quả: `<run>/benchmarks-<snapshot>.json` trong repo kết quả, base ở `e0_test/benchmarks-base.json`.

**Điểm (acc / acc_norm):**

| | ARC-Easy | HellaSwag |
|---|---|---|
| Base | 0.700 / 0.668 | 0.442 / 0.548 |
| E1 seed 0 | 0.700 / 0.664 | 0.444 / 0.564 |
| E2 seed 0 | 0.716 / 0.664 | 0.448 / 0.556 |
| E3 LoRA r64 seed 0 | 0.716 / 0.674 | 0.452 / 0.560 |

**Hiệu số ghép cặp (điểm, CI 95%):**
- So với base, mọi hiệu số nằm trong khoảng −0.4 tới +1.6 điểm, và CI đều chứa 0. Ví dụ ARC-Easy acc: E1 0.0 [−2.0, +2.2], E2 +1.6 [−0.4, +3.4], E3 +1.6 [−0.4, +3.6]. HellaSwag acc_norm: E1 +1.6 [−0.4, +3.6], E2 +0.8 [−1.2, +2.8], E3 +1.2 [−0.4, +3.0].
- E2 − E3 (H5, full fine-tune so với LoRA): ARC-Easy acc 0.0 [−1.4, +1.4], acc_norm −1.0 [−2.8, +0.8]; HellaSwag acc −0.4 [−1.6, +0.8], acc_norm −0.4 [−2.0, +1.2].
- E1 − E2: ARC-Easy acc −1.6 [−3.2, 0.0]; các ô khác CI chứa 0.

**Đọc (một seed):**
- Không model nào giảm điểm ARC-Easy/HellaSwag so với base, kể cả full fine-tune. Với 500 câu, CI của hiệu số rộng khoảng ±2 điểm, nên một mức quên nhỏ hơn cỡ đó không thấy được.
- Phần "quên" của H5 theo benchmark: chưa có bằng chứng. Full fine-tune (E2) không giảm điểm nhiều hơn LoRA (E3); hiệu số E2 − E3 dưới 1 điểm, CI chứa 0.
- Như vậy, dấu hiệu duy nhất của khác biệt về quên vẫn là perplexity wikitext-2 (E2 14.12, E3 13.63, base 13.01) và KL, mới là so điểm, chưa có CI. Muốn đọc H5 theo đúng tiêu chí PLAN (CI của hiệu số), cần thêm CI cho perplexity: ví dụ lưu NLL theo từng cửa sổ trong `src/eval/knowledge.py` rồi bootstrap ghép cặp theo cửa sổ.

## 2026-10-29 — Perplexity có CI: base, E1, E2, E3 seed 0 (snapshot cuối)

**Loại hoạt động:** đo lại perplexity wikitext-2 và KL tới base ở snapshot cuối, để có CI của hiệu số (tiêu chí "quên" của H5, PLAN §1). Không đổi cách tính: cùng văn bản, cùng cửa sổ 1024 token. Con số trùng với lần đo trước tới từng chữ số (E1 14.102, E2 14.124, E3 13.628, base 13.008).

**Cách tính CI:** `src/eval/knowledge.py` nay lưu tổng NLL (và KL) của từng cửa sổ, kèm bài wikitext mà cửa sổ bắt đầu trong đó (theo tiêu đề cấp một " = … = "; tập test có 62 bài, 61 bài có cửa sổ bắt đầu). `python -m src.eval.knowledge --compare A.json B.json` bootstrap 2000 lần, lấy mẫu lại theo **bài** (cụm, như PLAN §3.5) cho cả hai bên trên cùng các cửa sổ, rồi tính lại perplexity mỗi bên. Base đo không kèm KL (`e0_test/knowledge-base.json`). Chạy bằng cell 7 của `07_main` trên T4 Kaggle (bundle `19cbe3d4…`), khoảng 3 phút mỗi model.

**Hiệu số perplexity, CI 95%:**

| | Hiệu số |
|---|---|
| E1 − base | +1.09 [+1.00, +1.20] |
| E2 − base | +1.12 [+1.02, +1.24] |
| E3 − base | +0.62 [+0.57, +0.68] |
| E2 − E3 | **+0.50 [+0.45, +0.56]** |
| E1 − E2 | −0.02 [−0.04, −0.005] |

KL tới base: E2 − E3 +0.025 [+0.024, +0.026] nat/token; E1 − E2 −0.0004 [−0.001, +0.0003].

**Đọc (một seed):**
- Cả ba run làm perplexity wikitext tăng so với base, CI loại trừ 0: SFT có làm model kém đi trên văn bản thô, dù ARC-Easy/HellaSwag không đổi (mục trước).
- **H5, phần quên:** full fine-tune (E2) làm perplexity tăng nhiều hơn LoRA r64 (E3) khoảng 0.5, CI của hiệu số loại trừ 0. Theo tiêu chí PLAN ("perplexity tăng nhiều hơn **hoặc** điểm benchmark giảm nhiều hơn"), phần này đạt ở seed 0, nhờ perplexity; benchmark không cho thấy khác biệt. Kết hợp với mục kết quả E3 (`call_exact` thấp hơn 2.3 điểm, cận dưới −4.9), seed 0 phù hợp với cả hai vế của H5. Kết luận chính thức đợi đủ seed.
- E3 cũng lệch khỏi base ít hơn (KL), như PLAN lưu ý thì đó chỉ là chỉ báo thay đổi, không phải bằng chứng quên.
- Thứ tự dữ liệu gần như không ảnh hưởng: E1 − E2 là −0.02, nhỏ tới mức không có ý nghĩa thực tế dù CI vừa loại trừ 0.

## 2026-10-29 — Bộ P qua các checkpoint: E1, E2, E3 seed 0

**Loại hoạt động:** đọc định tính output đã sinh của bộ P (60 prompt viết tay, PLAN §3.2) ở mọi snapshot của ba run (PLAN Giai đoạn 3). Không sinh lại. Output được gom theo từng prompt qua các checkpoint. Ba agent đọc song song, mỗi agent một run, theo cùng bộ câu hỏi. Các nhận xét chính dưới đây đã được kiểm lại trực tiếp trên `outputs.jsonl`, và các con số đếm là đếm bằng script. Bộ P nhỏ (mỗi nhóm 6–10 prompt), nên đây là mô tả, không phải số đo.

**Cùng một tiến trình ở cả ba run:**
- **Vỏ trước, ruột sau.** Thẻ `<tool_call>` xuất hiện trước khi nội dung đúng. Lúc đầu có các dạng: chép lại tool call có sẵn trong ngữ cảnh Multi-turn (từ snapshot đầu tiên); gọi lặp tới hết `max_new_tokens`; gộp nhiều lời gọi vào một JSON với khóa trùng (`"symbol": "AAPL", "symbol": "MSFT"`); tên tool bịa (`flight_price`, `is_virus`). Thẻ này còn bị áp lên prompt không có tool, với một chuỗi trơn bên trong (`<tool_call>"sleep_tips"</tool_call>`).
- **Gọi tool đúng đến sớm, khoảng 200k token** (E2 ở step 38: 7/7 Single đúng tool và đúng định dạng). Lỗi còn lại tới cuối run là giá trị: tham số tùy chọn bịa (`date`, `unit`, `season`), dao động giữa các snapshot chứ không hội tụ, và một lỗi chọn tool ổn định (Select 05 luôn gọi `send_email` thay vì `send_sms`, ở cả ba run).
- **Gọi tool thừa (No-tool):** cao nhất ở giai đoạn đầu học tool, rồi chuyển sang câu từ chối cố định của dữ liệu ("None of the tools I have can help…"). Riêng câu hỏi về "déjà vu" vẫn gọi `translate_text` tới 614 (E2, E3) hoặc tới cuối run (E1).
- **Hỏi lại khi thiếu thông tin là kỹ năng yếu nhất trên P.** Ở snapshot cuối, model chỉ hỏi lại ở 1/6 prompt (E2), 2/6 (E1, E3); ở các prompt còn lại nó điền giá trị mặc định rồi gọi luôn: "New York", USD→EUR, `bill_amount: 100`, "+1234567890". Model hỏi lại ở prompt gửi email/SMS, nơi không có giá trị nào hợp lý để bịa; không bao giờ hỏi ở thời tiết, tiền tệ, phim, tiền tip. Điều này khác xa `clarify_correct` 0.50–0.62 trên test: prompt Missing-info của P cố ý mơ hồ hơn (thiếu thành phố mà không có gợi ý nào), còn ở test thiếu một tham số bắt buộc của câu hỏi xLAM/Glaive. Gợi ý (chưa kiểm): model học "hỏi khi không đoán được", chưa học "hỏi khi thiếu".
- **Multi-turn:** từ khoảng 200k, cả 6 prompt trả lời từ kết quả tool, đôi khi bịa thêm chi tiết ("over the last 10 days") hoặc gộp mất thông tin.
- **Chat và kiến thức:** câu trả lời dài ra ở nửa sau run và kéo theo câu sai (Canberra "in the state of Victoria" ở E2, "New South Wales" ở E3; Thế chiến II kết thúc "15 May 1945" ở E2). Bài thơ limerick lặp tới hết token ở mọi run. Một seed và một prompt mỗi loại, nên không đọc được đây là quên hay là nhiễu.

**E3 so với E2** (cùng thứ tự mẫu, cùng step): E3 chậm khoảng một snapshot ở đầu. Ở step 19, E2 đã có `<tool_call>` ở 19/21 prompt cần tool, E3 chỉ 3/21 (và sai định dạng: khóa `"parameters"`, dạng giả code `get_stock_price(symbol: "AAPL")`). Ở step 38, E2 dừng bằng `<|im_end|>` ở 59/60 output, E3 ở 43/60. Từ khoảng 800k, hai run gần như trùng nhau, nhiều output cuối giống từng chữ. Khác biệt theo prompt ở cuối run không theo một hướng. Khớp với hiệu số ghép cặp trên test (E3 chậm hơn ở 100k–400k, đuổi kịp từ 800k).

**E1 (học chat trước, tool sau):**
- **Giai đoạn chat:** không có `<tool_call>` nào ở Single/Select/Parallel/No-tool/Missing-info. Model bịa kết quả bằng văn xuôi (giá NVDA "$1.00", rồi "$10.50", rồi "$1,200.00"), hoặc viết code gọi API (`requests` với `Bearer <KEY>`, `smtplib`) thay vì gọi tool. Không có câu từ chối nào. Ở Multi-turn, model chép lại tool call có sẵn trong ngữ cảnh không đơn điệu (2/6 ở 1.6M, 4/6 ở 3.2M, 1/6 ở ranh giới), khớp với `grounded_correct` dao động đã ghi ở mục kết quả E1.
- **Ngay sau ranh giới, chat và kiến thức bị kéo về định dạng tool.** Ở +25k token, 15/20 prompt chat/kiến thức của P có `<tool_call>`, 17/20 ở +50k, rồi giảm dần. Bộ C của test cho cùng hình dạng: 46/150 output có `<tool_call>` ở +25k, 45/150 ở +50k, 6/150 ở +100k, 1/150 từ +400k tới cuối. E2 và E3 cao nhất là 13–14/150 ở một snapshot sớm, về 0 từ 800k. `stop_rate` C không bắt được hiện tượng này, vì `<tool_call>"…"</tool_call><|im_end|>` vẫn dừng đúng.
- **Cuối run, P còn lỗi nặng hơn C:** 6/20 prompt chat/kiến thức của P vẫn hỏng ở step 1250. Trong đó, 3 câu thành tool call chứa chuỗi trơn ("What is the chemical symbol for gold?" → `<tool_call>"What is the chemical symbol for gold?"</tool_call>`), 2 câu bị từ chối "None of the tools I have can help…" (Pride and Prejudice, năm Thế chiến II kết thúc), 1 câu hỏi ngược lại thay vì trả lời. E2 và E3 trả lời bình thường cả 6. Trên C, E1 chỉ còn 1/150. Prompt của P ngắn và mang tính tra cứu, giống câu hỏi của dữ liệu tool hơn, nên có thể nhạy hơn. Học tool sau cùng (E1) để lại một dấu vết trên chat mà trộn (E2, E3) không để lại; độ lớn của nó phụ thuộc loại câu hỏi.
- **Định dạng đến gần như tức thì:** ở +25k, hầu hết prompt cần tool đã có JSON đúng định dạng, không có pha nửa văn xuôi. Phần đến chậm là hành vi: khi nào gọi, khi nào hỏi lại, giá trị tham số.

**Ghi chú đầy đủ** của ba lần đọc nằm ngoài repo (scratchpad của session 2026-10-29), không commit vì trích nguyên văn output; các điểm dùng ở đây đã được kiểm lại.

## 2026-11-03 — Giả thuyết cho E2 seed 1 (ghi trước khi chạy)

**Run:** `configs/e2_mix.yaml`, seed 1. Dữ liệu, siêu tham số, lịch lr và lịch snapshot y như E2 seed 0. Seed chỉ đổi thứ tự mẫu (`build_order`, `data_seed`): full fine-tune không có tham số khởi tạo ngẫu nhiên, và SmolLM2 không dùng dropout. Vẫn là 1250 step và 6 527 256 token assistant, vì 20 000 mẫu × 2 epoch chia hết cho batch 32, nên không mẫu nào bị bỏ. Snapshot lấy theo mốc token, nên có thể rơi vào step hơi khác seed 0.

**Vì sao chạy run này trước:** E2 là mốc so của cả E1 − E2 (thứ tự dữ liệu) và E3 − E2 (H5). Seed 1 của E2 cho ước lượng đầu tiên về biến thiên do seed (PLAN §3.5, nguồn bất định (b)). Với hai seed mới chỉ có khoảng chênh, chưa có độ lệch chuẩn; độ lệch chuẩn cần seed 2.

**Giả thuyết run này kiểm:** H1–H3 ở seed thứ hai của E2. Run này cũng kiểm xem các số đo của seed 0 có tái lập không. H5 và câu hỏi về thứ tự dữ liệu chỉ được đọc lại khi có E1 và E3 seed 1.

**So với seed 0:** `python -m src.eval.stats --split test --baseline e0_test/base-zero-shot --compare e2_mix_s1 e2_mix_s0`. Hai seed được ghép theo prompt ở các mốc token chung. CI của phép so này chỉ phản ánh bất định do tập prompt, không phải biến thiên do seed, nên chỉ dùng để mô tả.

**Căn cứ:** E2 seed 0 (mục "Kết quả E2 seed 0" và "hiệu số ghép cặp T-rename − T-seen"). E3 r64 seed 0 học đúng thứ tự mẫu của E2 seed 0, khác lr và cách train, nhưng ở cuối run hầu hết hiệu số so với E2 vẫn dưới 3 điểm. Do đó, một thay đổi nhỏ hơn, chỉ là thứ tự mẫu, nhiều khả năng cũng không làm lệch nhiều.

**Dự đoán** ("xuất hiện" là t_reach với θ chính, trên test, T-seen; "cuối" là step cuối):
1. **Thứ tự** như seed 0: `decision_bacc` sớm nhất (100k–200k); format, stop, `tool_name_acc`, `grounded_correct` ở 200k; `call_exact` ở 400k; `clarify_correct` muộn nhất (1.6M–6.4M). Mỗi mốc lệch seed 0 không quá một checkpoint.
2. **H1:** đạt ở θ chính nhưng không vững qua ba θ, như seed 0: ở θ 0.3, `call_exact` đạt cùng mốc với format.
3. **H2:** "`decision_bacc` muộn nhất" sai. "`false_call_rate` vượt base rồi giảm" đúng: đỉnh trong khoảng 0.25–0.50 ở 50k–400k (CI loại trừ 0), xuống < 0.05 ở cuối.
4. **H3:** `call_exact` T-unseen cao hơn base few-shot (CI loại trừ 0). Mức giảm ghép cặp khi đổi sang tên vô nghĩa ở cuối nằm trong 5–13 điểm, cận trên CI < 20; mức giảm với tên đồng nghĩa < 5 điểm.
5. **Khả năng ở cuối:** `call_exact` T-seen 0.62–0.70, T-unseen 0.75–0.85; `decision_bacc` ≥ 0.98; `grounded_correct` ≥ 0.93. `clarify_correct` T-seen 0.50–0.72: khoảng này rộng vì CI của seed 0 đã rộng [0.49, 0.73]. Hiệu số ghép cặp s1 − s0 dưới 4 điểm ở mọi metric gọi tool; với `clarify_correct`, CI chứa 0.
6. **Quên:** perplexity wikitext-2 13.9–14.4 (seed 0: 14.12; E3: 13.63), tức vẫn cao hơn E3 seed 0. KL tới base 0.055–0.08. ARC-Easy/HellaSwag lệch base trong ±2 điểm, CI chứa 0. `stop_rate` C ≥ 0.88.
7. **Tài nguyên:** khoảng 9.2 giây/step, GPU đỉnh khoảng 11.1 GiB như seed 0.

**Kết quả coi là bất ngờ, phải kiểm hạ tầng trước khi đọc:**
- Số step khác 1250 hoặc tổng token khác 6 527 256.
- Thứ tự mẫu trùng seed 0 (seed không được truyền vào).
- Loss NaN.
- Một metric đã đạt θ rồi rơi về gần 0.
- `call_exact` T-seen cuối < 0.55.
- Perplexity > 14.6 hoặc < 13.5.

**Chạy:** trên Kaggle, notebook `07_main`, cell 6 với `CONFIG, SEED = "e2_mix", 1`. Trước khi chạy, cập nhật bản trên Drive (cell cài thư viện, cell nạp source, cell 6) theo notebook trong repo rồi so SHA-256 của bundle. Ước tính khoảng 6.5 giờ (train khoảng 3.2 giờ, chấm 10 snapshot khoảng 2.7 giờ, thêm benchmark và perplexity), trong giới hạn 12 giờ của phiên. Giữ tab Colab nối runtime suốt run (CLAUDE.md).

## 2026-11-03 — Kết quả E2 seed 1 (test)

**Chạy:** `configs/e2_mix.yaml`, seed 1, notebook `07_main` trên Kaggle (Colab nối vào Kaggle Jupyter Server, T4, một GPU). Bundle `0ff9aeff…` (commit `2351ab5`), trùng với repo. Trước khi chạy: 212 test qua; `src.eval.manifest` khớp 11 file; hash D_chat và D_tool khớp Khóa giao thức. Run chạy một mạch, không bị ngắt, không phải resume.
- Train: 1250 step, 6 527 256 token assistant, khớp số tính trước. Thứ tự mẫu khác seed 0, nên snapshot rơi vào step 5, 11, 21, 41, 79, 156, 308, 614, 1227, 1250 (seed 0: 5, 9, 19, 38, 77, 155, 307, 614, 1226, 1250). Không lần đẩy checkpoint nào lỗi; `train_loss` 0.699, không có NaN.
- Khoảng 8.1 giây/step (seed 0: 9.2), GPU đỉnh 11.08 GiB. Train mất 3.0 giờ, chấm test khoảng 15 phút mỗi snapshot; cả run khoảng 5.5 giờ.
- Trọng số: `trantam31/smollm2-360m-e2-mix-s1`. Output và metric: repo kết quả, thư mục `e2_mix_s1/` (private).

**Snapshot cuối (step 1250), test, CI bootstrap 95% theo cụm; số trong ngoặc tròn là của seed 0:**

| | T-seen | T-unseen |
|---|---|---|
| `format_valid` | 0.99 [0.98, 1.00] (1.00) | 0.98 (0.97) |
| `call_exact` | 0.67 [0.61, 0.72] (0.66) | 0.81 [0.76, 0.87] (0.80) |
| `call_exact` T-rename vô nghĩa / đồng nghĩa | 0.58 / 0.65 (0.57 / 0.64) | |
| `decision_bacc` | 0.995 [0.98, 1.00] (0.995) | 0.997 (0.993) |
| `false_call_rate` | 0.01 [0.00, 0.03] (0.01) | 0.00 (0.00) |
| `clarify_correct` | 0.61 [0.48, 0.72] (0.62) | 0.58 [0.35, 0.83] (0.61) |
| `grounded_correct` | 0.95 [0.91, 1.00] (0.97) | 0.97 (0.95) |
| `stop_rate` C | 0.96 [0.92, 0.99] (0.93) | |

**t_reach (θ chính, T-seen; token assistant):** `stop_rate`, `format_valid`, `tool_name_acc`, `grounded_correct`, `decision_bacc` 204k → `call_exact` 401k → `clarify_correct` 6.4M. T-unseen giống vậy, trừ `call_exact` đạt từ 204k. `clarify_correct` đạt 0.5 ở 6.4M (0.61) nhưng `stats.py` ghi "unconfirmed", vì quy tắc cần điểm giữ trên θ thêm hai checkpoint sau đó mà chỉ còn snapshot cuối. Ở 3.2M nó là 0.46 (seed 0: 0.56).

**Theo checkpoint (T-seen; mốc 25k, 50k, 100k, 200k, 400k, 800k, 1.6M, 3.2M, 6.4M, cuối):**
- `false_call_rate`: 0, 0, **0.60 [0.50, 0.69]**, 0.29, 0.25, 0.21, 0.02, 0.00, 0.01, 0.01. Seed 0: đỉnh 0.38 cũng ở 100k.
- `call_exact`: 0, 0, 0.00, 0.46, 0.56, 0.57, 0.59, 0.65, 0.67, 0.67. Seed 0: 0, 0, 0.02, 0.37, 0.55, 0.58, 0.62, 0.66, 0.66, 0.66.
- `clarify_correct`: 0, 0, 0, 0.17, 0.09, 0.42, 0.43, 0.46, 0.61, 0.61. Có một lần giảm 0.17 → 0.09 ở 400k, CI chồng nhau.

**Hiệu số ghép cặp s1 − s0** (`--compare e2_mix_s1 e2_mix_s0`, ghép theo prompt ở cùng mốc token; CI chỉ phản ánh bất định do tập prompt, không phải do seed):
- Cuối run, mọi metric gọi tool chênh dưới 2 điểm: `call_exact` T-seen +0.7 [−1.6, +3.1], T-unseen +1.0 [−0.7, +2.9]; `tool_name_acc` T-unseen +1.3 [+0.3, +2.8]. `clarify_correct` T-seen −1.0 [−4.4, +2.1], T-unseen −3.0 [−6.3, 0.0]. `grounded_correct` T-seen −2.0 [−5.0, 0.0], T-unseen +2.0 [0.0, +6.2]. `stop_rate` C +2.7 [−2.1, +7.6].
- **Đầu run thì chênh lớn.** Ở 100k, seed 1 gọi tool thừa nhiều hơn (`false_call_rate` +22 điểm [+14, +30]) và phân biệt có/không cần tool kém hơn (`decision_bacc` −7.5 [−12.0, −3.3]). Ở 200k nó lại đi trước: `tool_name_acc` +11 điểm, `call_exact` +8.7 [+4.7, +12.7]. Từ 400k, các hiệu số về dưới 5 điểm.

**H3, hiệu số ghép cặp T-rename − T-seen (`call_exact`, ba nhóm gộp):**
- Vô nghĩa, cuối run: giảm 8.7 điểm [3.5, 13.7] (seed 0: 9.0 [4.5, 13.8]). Theo checkpoint từ 200k: 8.3, 10.0, 6.7, 7.0, 9.0, 8.7, 8.7. Cận trên CI cao nhất 14.4, luôn dưới 20.
- Đồng nghĩa, cuối run: giảm 2.3 điểm [−1.0, 5.9]. Riêng ở 200k giảm 5.0 [1.6, 8.7]; từ 400k CI luôn chứa 0.

**Quên (snapshot cuối):**
- Perplexity wikitext-2 14.12 (seed 0: 14.12). Hiệu số s1 − s0 là −0.001 [−0.017, +0.015]. s1 − base là +1.11 [+1.02, +1.24]. s1 − E3 r64 s0 là +0.49 [+0.45, +0.56], gần như trùng với E2 s0 − E3 s0 (+0.50).
- KL tới base 0.069 (seed 0: 0.068).
- ARC-Easy acc / acc_norm 0.708 / 0.664; HellaSwag 0.444 / 0.572. So với base, ba ô có CI chứa 0. Riêng HellaSwag acc_norm cao hơn base 2.4 điểm [+0.6, +4.4], và cao hơn seed 0 1.6 điểm [+0.4, +2.8].

**Đối chiếu với dự đoán ghi trước khi chạy:**
1. **Thứ tự:** đúng phần lớn. Format, stop, `tool_name_acc`, `grounded_correct` ở 200k; `call_exact` ở 400k; `clarify_correct` muộn nhất (6.4M). Mỗi mốc lệch seed 0 không quá một checkpoint. Khác dự đoán: `decision_bacc` không đến sớm nhất mà cùng mốc 200k với format. Ở 100k nó là 0.70, ngay dưới θ 0.75, trong khi seed 0 đã là 0.77.
2. **H1:** đúng như dự đoán. Đạt ở θ chính (format 204k, `call_exact` 401k, cách một checkpoint) nhưng không vững: ở θ 0.3, `call_exact` đạt cùng mốc với format; trên T-unseen thì đạt cùng mốc ngay ở θ chính.
3. **H2:**
   - "`decision_bacc` muộn nhất": sai, như dự đoán.
   - "`false_call_rate` vượt base rồi giảm": đúng hình dạng. Đỉnh ở 100k, CI loại trừ 0, cuối run 0.01.
   - Độ cao đỉnh **sai dự đoán**: 0.60, trên khoảng 0.25–0.50. Đỉnh này nằm giữa seed 0 (0.38) và pilot (0.67–0.70), nên độ cao đỉnh phụ thuộc thứ tự mẫu nhiều hơn tôi nghĩ.
4. **H3:** đúng dự đoán. Base few-shot ở `call_exact` T-unseen là 0, nên vế "cao hơn base" là hiển nhiên. Mức giảm với tên vô nghĩa 8.7 điểm, nằm trong 5–13, cận trên 13.7 < 20; mức giảm với tên đồng nghĩa 2.3 < 5. Tiêu chí H3 (giảm ≤ 10 điểm, cận trên < 20) đạt ở cả hai seed.
5. **Khả năng ở cuối:** cả năm khoảng dự đoán đều đúng. Hiệu số s1 − s0 dưới 4 điểm ở mọi metric gọi tool; `clarify_correct` có CI chứa 0.
6. **Quên:**
   - Đúng: perplexity 14.12 (khoảng 13.9–14.4, vẫn cao hơn E3), KL 0.069, `stop_rate` C 0.96.
   - Sai một ô: HellaSwag acc_norm cao hơn base 2.4 điểm, CI loại trừ 0, trong khi dự đoán là lệch trong ±2 và CI chứa 0. Hướng lệch là tốt lên chứ không phải quên. Ba ô còn lại khớp dự đoán.
7. **Tài nguyên:** GPU 11.08 GiB khớp; tốc độ nhanh hơn dự đoán (8.1 so với 9.2 giây/step). Có thể do máy Kaggle khác, vì cấu hình không đổi.
- Không có kết quả "bất ngờ" nào trong danh sách đã ghi.

**Đọc (hai seed của E2):**
- Ở cuối run, kết quả tái lập tốt: mọi metric chính lệch dưới 3 điểm, perplexity trùng tới hai chữ số thập phân. Biến thiên do thứ tự mẫu ở snapshot cuối nhỏ hơn CI do tập prompt.
- Ở giai đoạn đầu (100k–200k), thứ tự mẫu làm lệch tới 10–20 điểm ở cùng mốc token. Đây cũng là giai đoạn t_reach được đo. Vì vậy, hiệu số giữa các run ở các mốc đầu, như E3 chậm hơn E2 ở 100k–400k, cần đối chiếu với độ lệch này trước khi đọc là do cấu hình. Muốn có độ lệch chuẩn thì cần seed 2.
- Thứ tự mẫu không làm đổi thứ tự xuất hiện của các kỹ năng; chỉ dịch mốc tối đa một checkpoint.

**Việc còn lại:** bộ P của seed 1 (chưa đọc). E1 và E3 seed 1 để đọc lại H5 và câu hỏi về thứ tự dữ liệu.

## 2026-11-03 — Giả thuyết cho E3 LoRA r64 seed 1 (ghi trước khi chạy)

**Run:** `configs/e3_lora_r64.yaml`, seed 1. Cấu hình y như E3 seed 0: LoRA r 64, alpha 128, lr 2e-4, cùng dữ liệu, lịch lr và lịch snapshot. Thứ tự mẫu chỉ phụ thuộc seed, nên E3 seed 1 học đúng thứ tự của E2 seed 1, và snapshot phải rơi vào cùng step (5, 11, 21, 41, 79, 156, 308, 614, 1227, 1250). Vì vậy E3 − E2 được ghép cặp ở cả 10 snapshot của seed 1, như ở seed 0.

**Vì sao chạy run này:**
- E3 − E2 seed 1 là lần lặp đầu tiên của phép so H5.
- Nó cũng kiểm xem E3 có chậm hơn E2 ở 200k–400k như seed 0 không. Kết quả E2 seed 1 cho thấy chỉ riêng thứ tự mẫu đã làm lệch 10–20 điểm ở 100k–200k. Vì vậy cần một seed nữa, với thứ tự mẫu khác, mới đọc được độ chậm đó là do LoRA (cộng lr) hay do thứ tự mẫu.

**Giả thuyết run này kiểm:** H5 ở seed thứ hai; H1–H3 trên E3.

**So sánh:**
- `python -m src.eval.stats --split test --baseline e0_test/base-zero-shot --compare e3_lora_r64_s1 e2_mix_s1`: H5, ghép cặp theo prompt và cùng step.
- `--compare e3_lora_r64_s1 e3_lora_r64_s0`: mô tả, như s1 − s0 của E2.
- Quên: `src.eval.knowledge --compare` và `src.eval.benchmarks --compare` giữa E3 s1 và E2 s1.

**Căn cứ:**
- E3 seed 0: ở 200k, `format_valid` −16 điểm và `call_exact` −15 điểm so với E2; từ 800k các hiệu số dưới 3 điểm. Cuối run, `call_exact` −2.3 [−4.9, 0.0]; perplexity 13.63, thấp hơn E2 0.50 [0.45, 0.56].
- E2 seed 1: `call_exact` 0.67 / 0.81; đỉnh `false_call_rate` 0.60 ở 100k; perplexity 14.12; s1 − s0 dưới 2 điểm ở cuối run.
- Cơ chế đoán cho việc E3 chậm ở đầu: adapter khởi tạo B = 0, nên ở những step đầu thay đổi của model nhỏ hơn full fine-tune, kể cả với lr cao hơn. Cơ chế này không phụ thuộc thứ tự mẫu, nên tôi đoán độ chậm sẽ lặp lại.

**Dự đoán** ("xuất hiện" là t_reach với θ chính, trên test, T-seen; hiệu số là E3 s1 − E2 s1, ghép cặp):
1. **Độ chậm ở đầu lặp lại:** ở mốc 200k, `call_exact` và `format_valid` thấp hơn E2 s1 từ 5 tới 25 điểm, CI loại trừ 0. Từ 800k, các hiệu số này dưới 4 điểm.
2. **Thứ tự** như E2 và E3 seed 0. Mỗi mốc bằng hoặc muộn hơn E2 s1 tối đa một checkpoint, không mốc nào sớm hơn.
3. **H1:** đạt ở θ chính. Vững qua ba θ như E3 seed 0. Kém chắc hơn các dự đoán khác: ở seed 0, H1 vững chỉ vì `call_exact` đến chậm.
4. **H2:**
   - "`decision_bacc` muộn nhất": sai.
   - "`false_call_rate` vượt base rồi giảm": đúng. Đỉnh trong khoảng 0.25–0.65 ở 100k–400k, CI loại trừ 0; cuối run < 0.05.
5. **H3:** mức giảm ghép cặp khi đổi sang tên vô nghĩa ở cuối nằm trong 5–13 điểm, cận trên CI < 20. Mức giảm với tên đồng nghĩa < 5 điểm.
6. **H5, khả năng (cuối run):** `call_exact` T-seen 0.60–0.69, T-unseen 0.75–0.86. Hiệu số `call_exact` T-seen trong khoảng −6 tới +2 điểm (đoán điểm −2). `clarify_correct` có CI chứa 0.
7. **H5, quên:**
   - Perplexity wikitext-2 13.4–13.9, thấp hơn E2 s1 từ 0.3 tới 0.7, CI loại trừ 0.
   - KL tới base 0.035–0.05.
   - ARC-Easy/HellaSwag lệch E2 s1 dưới 2 điểm, CI chứa 0.
   - `stop_rate` C ≥ 0.85.
8. **Tài nguyên:** GPU đỉnh khoảng 8.8 GiB. Tốc độ khoảng 9.5–11 giây/step: chậm hơn E2 s1 (8.1) khoảng 17% như ở seed 0.

**Kết quả coi là bất ngờ, phải kiểm hạ tầng trước khi đọc:**
- Số step khác 1250, tổng token khác 6 527 256, hoặc số token ở từng snapshot không khớp E2 s1 (thứ tự mẫu không khớp).
- Loss NaN, hoặc `grad_norm` vọt lên.
- `call_exact` T-seen cuối < 0.5.
- Perplexity cao hơn E2 s1 hoặc thấp hơn 13.2.

**Chạy:** trên Kaggle, notebook `07_main`, cell 6 với `CONFIG, SEED = "e3_lora_r64", 1`. Ước tính khoảng 3.5–4 giờ train và khoảng 2.7 giờ chấm, tổng khoảng 6.5–7 giờ, trong giới hạn 12 giờ của phiên. Kiểm hạn mức GPU tuần trên Kaggle trước khi bật phiên. Giữ tab Colab nối runtime suốt run.

## 2026-11-03 — Kết quả E3 LoRA r64 seed 1 (test)

**Chạy:** `configs/e3_lora_r64.yaml`, seed 1, notebook `07_main` trên Kaggle (Colab nối vào Kaggle Jupyter Server, T4, một GPU). Bundle `0ff9aeff…`, trùng với repo. Trước khi chạy: `src.eval.manifest` khớp 11 file; hash D_chat và D_tool khớp Khóa giao thức. Run chạy một mạch, không bị ngắt, không phải resume.
- Train: 1250 step, 6 527 256 token assistant. Số token ở cả 10 snapshot trùng E2 s1 (step 5, 11, 21, 41, 79, 156, 308, 614, 1227, 1250), nên thứ tự mẫu khớp và phép ghép cặp E3 − E2 hợp lệ. Không lần đẩy checkpoint nào lỗi; `train_loss` 0.749; không có NaN; `grad_norm` lớn nhất 0.40 (step 5), trung vị 0.16.
- Khoảng 11.2 giây/step (E3 s0: 10.8), GPU đỉnh 8.79 GiB. Train mất 4.0 giờ, chấm test khoảng 15 phút mỗi snapshot; cả run khoảng 6.7 giờ.
- Trọng số: `trantam31/smollm2-360m-e3-lora-r64-s1`. Output và metric: repo kết quả, thư mục `e3_lora_r64_s1/` (private).

**Snapshot cuối (step 1250), test, CI bootstrap 95% theo cụm; số trong ngoặc tròn là E2 s1 / E3 s0:**

| | T-seen | T-unseen |
|---|---|---|
| `format_valid` | 0.99 [0.98, 1.00] (0.99 / 0.99) | 0.98 (0.98 / 0.98) |
| `call_exact` | 0.66 [0.61, 0.72] (0.67 / 0.64) | 0.80 (0.81 / 0.81) |
| `decision_bacc` | 0.995 (0.995 / 1.000) | |
| `false_call_rate` | 0.01 [0.00, 0.03] (0.01 / 0.00) | 0.00 |
| `clarify_correct` | 0.59 [0.46, 0.70] (0.61 / 0.59) | 0.57 (0.58 / 0.61) |
| `grounded_correct` | 0.96 (0.95 / 0.96) | |
| `stop_rate` C | 0.93 [0.88, 0.97] (0.96 / 0.89) | |

**t_reach (θ chính, T-seen; token assistant):** `decision_bacc` 105k → `stop_rate`, `format_valid`, `tool_name_acc`, `grounded_correct` 204k → `call_exact` 401k → `clarify_correct` 6.4M ("unconfirmed", như E2 s1: chỉ còn snapshot cuối sau đó). So với E2 s1, mọi mốc trùng, trừ `decision_bacc` sớm hơn một checkpoint (E2 s1: 204k).

**Theo checkpoint (T-seen; mốc 25k, 50k, 100k, 200k, 400k, 800k, 1.6M, 3.2M, 6.4M, cuối):**
- `false_call_rate`: 0, 0, 0.18, **0.41 [0.32, 0.50]**, 0.28, 0.23, 0.07, 0.01, 0.01, 0.01. Đỉnh ở 200k (E2 s1: 0.60 ở 100k; E3 s0: 0.37 ở 200k).
- `call_exact`: 0, 0, 0.15, 0.37, 0.50, 0.56, 0.56, 0.63, 0.66, 0.66.
- `format_valid`: 0, 0, 0.33, 0.94, 0.96, 1.00, 0.99, 1.00, 0.99, 0.99.
- `clarify_correct`: 0, 0, 0, 0, 0.04, 0.34, 0.45, 0.46, 0.59, 0.59.

**H5, hiệu số ghép cặp E3 s1 − E2 s1** (`--compare e3_lora_r64_s1 e2_mix_s1`, cùng step, cùng prompt):
- **Ở 100k, E3 đi trước:** `format_valid` +22.7 điểm [+16.0, +28.9], `call_exact` +15.0 [+10.9, +19.5], `tool_name_acc` +22.3, `decision_bacc` +14.7 [+9.7, +19.8], `false_call_rate` −42 [−52, −33]. Nhưng E3 chưa biết dừng: `stop_rate` T-seen 0.00, so với 0.19 của E2 (riêng nhóm nhiều lượt 0.00 so với 0.99). Hai run đi hai đường khác nhau: E2 học dừng trước, E3 học định dạng lời gọi trước.
- **Ở 200k–400k, E3 chậm hơn, nhưng ít hơn seed 0:** 200k `call_exact` −8.3 [−11.8, −4.9], `format_valid` −3.0 [−5.6, −0.3], `tool_name_acc` −8.0, `grounded_correct` −13 [−23, −6]; 400k `call_exact` −6.0 [−8.9, −3.4]. Ở seed 0, mốc 200k là −15 và −16.
- `clarify_correct` chậm lâu nhất: −17 [−25, −10] ở 200k, −5 ở 400k, −8.0 [−14.3, −2.9] ở 800k; từ 1.6M CI chứa 0.
- Từ 800k, `call_exact` và `format_valid` chênh dưới 4 điểm ở mọi mốc. Cuối run: `call_exact` T-seen −0.7 [−2.8, +1.3], T-unseen −1.3 [−3.7, +0.6]; `clarify_correct` T-seen −2.0 [−7.1, +2.7], T-unseen −1.0 [−8.1, +3.6]; `grounded_correct` +1.0 [0.0, +2.5].

**Hiệu số ghép cặp E3 s1 − E3 s0** (`--compare e3_lora_r64_s1 e3_lora_r64_s0`, ghép theo prompt ở mốc token chung; chỉ để mô tả):
- Cuối run, mọi metric chênh dưới 4.5 điểm: `call_exact` T-seen +2.3 [+0.3, +4.5], T-unseen −1.3 [−3.3, +0.4]; `clarify_correct` T-seen 0.0, T-unseen −4.0 [−12.0, 0.0]; `stop_rate` C +3.3 [−2.1, +8.8].
- Đầu run thì seed 1 đi trước seed 0 rất xa: ở 100k `format_valid` +31.7 [+26.0, +37.0], `decision_bacc` +24.2; ở 200k `tool_name_acc` +28.0, `call_exact` +15.3 [+9.8, +21.0]. `clarify_correct` thì ngược lại: −25 ở 400k, −12 [−18, −7] ở 3.2M.

**H3, hiệu số ghép cặp T-rename − T-seen (`call_exact`, cuối run):** tên vô nghĩa giảm 11.3 điểm [6.5, 16.4] (E2 s1: 8.7; E3 s0: 8.0); tên đồng nghĩa giảm 3.7 [0.0, 7.5].

**Quên (snapshot cuối):**
- Perplexity wikitext-2 13.64. So với E2 s1: −0.48 [−0.54, −0.44]. So với E3 s0 (13.63): +0.013 [+0.003, +0.023]. So với base: +0.63 [+0.58, +0.70].
- KL tới base 0.044 (E2 s1: 0.069; E3 s0: 0.043); E3 s1 − E2 s1 là −0.025 [−0.026, −0.024].
- ARC-Easy acc / acc_norm 0.712 / 0.682; HellaSwag 0.460 / 0.560. So với E2 s1: ARC-Easy acc_norm +1.8 [+0.2, +3.4] và HellaSwag acc +1.6 [+0.4, +3.0], CI loại trừ 0; hai ô còn lại CI chứa 0 (HellaSwag acc_norm −1.2 [−2.6, +0.2]). So với E3 s0: cả bốn ô CI chứa 0. So với base: HellaSwag acc +1.8 [+0.2, +3.4]; ba ô còn lại CI chứa 0.

**Đối chiếu với dự đoán ghi trước khi chạy:**
1. **Độ chậm ở đầu:** sai một phần.
   - Ở 200k, `call_exact` −8.3, nằm trong khoảng 5–25, CI loại trừ 0: đúng.
   - `format_valid` chỉ −3.0, dưới khoảng 5–25: **sai**.
   - Ngoài dự đoán: ở 100k E3 đi trước E2 tới 15–23 điểm.
   - Từ 800k, chênh dưới 4 điểm: đúng.
2. **Thứ tự:** đúng phần lớn. Thứ tự kỹ năng như E2 và E3 seed 0, và các mốc trùng E2 s1. Riêng `decision_bacc` sớm hơn E2 s1 một checkpoint, trái với vế "không mốc nào sớm hơn".
3. **H1:** đúng trên T-seen. Format đến trước `call_exact` ở cả ba θ: 105k so với 204k ở θ 0.3, 204k so với 401k ở θ chính, còn ở θ 0.7 `call_exact` không đạt. Trên T-unseen ở θ chính, hai mốc trùng nhau ở 204k, như E2 s1 nhưng khác E3 s0.
4. **H2:** đúng.
   - `decision_bacc` không đến muộn nhất mà sớm nhất (105k).
   - `false_call_rate` đạt đỉnh 0.41 [0.32, 0.50] ở 200k, nằm trong khoảng dự đoán; cuối run 0.01.
5. **H3:** đúng. Mức giảm với tên vô nghĩa là 11.3, trong khoảng 5–13; cận trên 16.4 < 20. Với tên đồng nghĩa, ước lượng điểm 3.7 < 5, nhưng cận trên CI 7.5.
6. **H5, khả năng:** cả bốn vế đúng.
   - `call_exact` T-seen 0.66 và T-unseen 0.80, đều trong khoảng.
   - Hiệu số T-seen −0.7, trong khoảng −6 tới +2.
   - `clarify_correct` có CI chứa 0.
7. **H5, quên:** đúng ba vế, sai một vế.
   - Đúng: perplexity 13.64, thấp hơn E2 s1 0.48 với CI loại trừ 0; KL 0.044; `stop_rate` C 0.93.
   - Sai: ARC-Easy/HellaSwag. Cả bốn ô lệch E2 s1 dưới 2 điểm, nhưng hai ô có CI loại trừ 0, ở hướng E3 tốt hơn.
8. **Tài nguyên:** GPU 8.79 GiB, khớp. Tốc độ 11.2 giây/step, trên khoảng 9.5–11 một chút: chậm hơn E2 s1 38%, không phải 17%. Tỉ lệ này lệch vì E2 s1 chạy nhanh bất thường, còn E3 s1 gần E3 s0 (10.8).
- Không có kết quả "bất ngờ" nào trong danh sách đã ghi.

**Đọc (hai seed của E3, H5 ở hai seed):**
- **Cuối run, H5 lặp lại.** LoRA r64 đạt khả năng như full fine-tune: `call_exact` chênh dưới 2.5 điểm ở cả hai seed, CI chạm hoặc chứa 0. Nó cũng quên ít hơn: perplexity thấp hơn khoảng 0.5 ở cả hai seed (0.50 và 0.48), KL bằng khoảng hai phần ba. Hai seed của E3 trùng perplexity tới 0.013.
- **Độ chậm của E3 ở đầu run không lặp lại như ở seed 0,** nên giả thuyết B = 0 không đủ giải thích.
  - Ở seed 1, E3 đi trước E2 ở 100k, chậm hơn ở 200k–400k nhưng ít hơn seed 0, rồi bắt kịp từ 800k.
  - Ở cùng mốc token, chênh E3 s1 − E3 s0 (tới 32 điểm) lớn hơn chênh E3 − E2 trong cùng seed. Thứ tự mẫu, có lẽ cộng với tương tác giữa thứ tự và cấu hình, chi phối giai đoạn đầu nhiều hơn chính cấu hình.
  - Một điểm lặp lại: `clarify_correct` của E3 chậm hơn E2 tới 800k ở cả hai seed.
- **Thứ tự xuất hiện của kỹ năng vững qua cả cấu hình lẫn seed:** format/stop/tool → `call_exact` → `clarify`. Các mốc dịch tối đa một checkpoint.

**Việc còn lại:** bộ P của E2 s1 và E3 s1 (chưa đọc). E1 seed 1 cho câu hỏi về thứ tự dữ liệu. Seed 2 nếu muốn tách được độ lệch do thứ tự mẫu ở giai đoạn đầu.

## 2026-11-03 — Giả thuyết cho E1 seed 1 (ghi trước khi chạy)

**Run:** `configs/e1_seq.yaml`, seed 1. Cấu hình y như E1 seed 0: D_chat × 2 epoch (step 1–625), rồi D_tool × 2 epoch (step 626–1250), một lịch lr, optimizer liên tục. Seed chỉ đổi thứ tự mẫu trong mỗi giai đoạn, nên ranh giới vẫn ở step 625. Các snapshot khác có thể lệch vài step so với seed 0. Chấm test ở khoảng 18 snapshot, như seed 0: thang tích lũy, ranh giới, thang từ ranh giới, rồi 6.4M và step cuối.

**Vì sao chạy run này:**
- Ở seed 0, khác biệt rõ duy nhất do thứ tự dữ liệu là `clarify_correct`. E1 thấp hơn E2 9–14 điểm, CI loại trừ 0, ở cả 6.4M lẫn step cuối. Mọi metric gọi tool khác chênh dưới 2 điểm.
- E2 và E3 seed 1 cho thấy thứ tự mẫu làm lệch tới 10–30 điểm ở các snapshot đầu, nhưng dưới khoảng 4 điểm ở snapshot cuối. Khoảng chênh `clarify_correct` của E1 nằm ở snapshot cuối, nên seed 1 là phép thử đầu tiên xem nó có giữ qua thứ tự mẫu khác không.
- Run này cũng lặp lại các quan sát của seed 0 ở ranh giới và ngay sau đó, khi chat bị kéo về định dạng tool.

**Giả thuyết run này kiểm:** câu hỏi về thứ tự dữ liệu (E1 − E2) ở seed thứ hai; H1–H3 trên E1.

**So sánh:**
- `python -m src.eval.stats --split test --baseline e0_test/base-zero-shot --compare e1_seq_s1 e2_mix_s1`: ghép theo prompt ở các mốc token cộng dồn chung và step cuối. Snapshot ranh giới và thang từ ranh giới không có cặp.
- `--compare e1_seq_s1 e1_seq_s0`: mô tả, như s1 − s0 của E2 và E3. CI chỉ phản ánh bất định do tập prompt.
- Quên: `src.eval.knowledge --compare` và `src.eval.benchmarks --compare` giữa E1 s1 và E2 s1, và giữa E1 s1 và E1 s0.

**Căn cứ:**
- E1 seed 0, ở ranh giới:
  - mọi metric gọi tool bằng 0;
  - `stop_rate` C 0.90, T-seen 0.49;
  - `grounded_correct` 0.50, dao động trong giai đoạn chat (0.81 → 0.25 → 0.50) vì model chép lại tool call có trong ngữ cảnh.
- E1 seed 0, giai đoạn tool, tính bằng token từ ranh giới:
  - `format_valid` đạt ở 25k;
  - `call_exact` và `decision_bacc` đạt ở 100k;
  - `false_call_rate` lên đỉnh 0.64 ở 25k;
  - `clarify_correct` đứng ở 0.45–0.51 từ 800k.
- E1 seed 0, cuối run: `call_exact` 0.65 / 0.81; perplexity 14.10; KL 0.067. Ngay sau ranh giới, 46/150 output của bộ C có `<tool_call>` ở +25k và 45/150 ở +50k; từ +400k chỉ còn 1/150.
- E2 seed 1, cuối run: `call_exact` 0.67 / 0.81; `clarify_correct` 0.61 / 0.58; perplexity 14.12.

**Dự đoán** ("xuất hiện" là t_reach với θ chính, trên test, T-seen; "token từ ranh giới" là thang thứ hai):
1. **Ranh giới (chỉ học chat):**
   - `format_valid` và `call_exact` bằng 0, CI so với base chứa 0; `decision_bacc` ≈ 0.5.
   - `stop_rate` C ≥ 0.85, `stop_rate` T-seen trong khoảng 0.35–0.65.
   - `clarify_correct` ≤ 0.05.
   - `grounded_correct` ở ranh giới thì khó đoán, trong khoảng 0.2–0.85. Nó dao động không đơn điệu qua các snapshot của giai đoạn chat, như seed 0.
2. **Giai đoạn tool, thứ tự:**
   - `format_valid` đạt ở 25k token từ ranh giới.
   - `call_exact` ở 25k–100k; `decision_bacc` ở 50k–200k.
   - `clarify_correct` đạt muộn nhất: từ 800k trở đi, hoặc chỉ ở snapshot cuối, hoặc không đạt.
3. **H1:** đạt ở θ chính: `format_valid` trước `call_exact` ít nhất một checkpoint của thang thứ hai. Không vững qua ba θ, như seed 0.
4. **H2:**
   - "`decision_bacc` muộn nhất": sai.
   - "`false_call_rate` vượt base rồi giảm": đúng. Đỉnh trong khoảng 0.40–0.80 ở 25k–50k token từ ranh giới, CI loại trừ 0; cuối run < 0.05.
5. **H3:** mức giảm ghép cặp khi đổi sang tên vô nghĩa ở cuối nằm trong 5–13 điểm, cận trên CI < 20. Mức giảm với tên đồng nghĩa < 5.
6. **Thứ tự dữ liệu, cuối run (E1 s1 − E2 s1, ghép cặp):**
   - **`clarify_correct` T-seen thấp hơn từ 6 tới 16 điểm, CI loại trừ 0.** Đây là dự đoán chính của run. T-unseen cùng hướng, nhưng CI có thể chạm 0. Từ 800k token từ ranh giới tới cuối, `clarify_correct` T-seen của E1 đứng trong khoảng 0.40–0.56.
   - `call_exact` T-seen và T-unseen chênh trong ±4 điểm, CI chứa 0. `decision_bacc` và `false_call_rate` chênh dưới 2 điểm.
7. **Chat bị kéo về định dạng tool sau ranh giới**, đếm bằng script trên `outputs.jsonl` của bộ C:
   - ≥ 20/150 output có `<tool_call>` ở +25k hoặc +50k;
   - ≤ 3/150 từ +400k tới cuối.
8. **Quên:**
   - Perplexity wikitext-2 13.9–14.3; hiệu số so với E2 s1 trong ±0.1.
   - KL tới base 0.06–0.075.
   - ARC-Easy/HellaSwag lệch E2 s1 dưới 2 điểm.
   - `stop_rate` C cuối run ≥ 0.88.
9. **Tài nguyên:** khoảng 8–9 giây/step, GPU đỉnh khoảng 11 GiB. Train khoảng 3 giờ, chấm khoảng 18 × 15 phút; tổng khoảng 7.5–8.5 giờ.

**Kết quả coi là bất ngờ, phải kiểm hạ tầng trước khi đọc:**
- Ranh giới không ở step 625; tổng token khác 6 527 256.
- `call_exact` > 0.05 ở ranh giới: kiểm D_chat có lẫn mẫu tool không, và kiểm `order.json`.
- Mọi metric kỹ năng rơi về gần 0 ngay sau ranh giới.
- Loss NaN, hoặc `grad_norm` vọt lên.
- `call_exact` T-seen cuối < 0.5, hoặc `stop_rate` C cuối < 0.5.

**Chạy:** trên Kaggle, notebook `07_main`, cell 6 với `CONFIG, SEED = "e1_seq", 1`. Tổng khoảng 8 giờ, trong giới hạn 12 giờ của phiên. Kiểm hạn mức GPU tuần trên Kaggle trước khi bật phiên: tuần này đã dùng khoảng 13 giờ, nên còn đủ. Giữ tab Colab nối runtime suốt run.

## 2026-11-04 — Kết quả E1 seed 1 (test)

**Chạy:** `configs/e1_seq.yaml`, seed 1, notebook `07_main` trên Kaggle (Colab nối vào Kaggle Jupyter Server, T4, một GPU). Bundle `0ff9aeff…`, trùng với repo.
- Train: 1250 step, 6 527 256 token assistant. Ranh giới ở step 625 (4 607 572 token, đúng như seed 0). `train_loss` 0.598; không có NaN; `grad_norm` lớn nhất 1.82. Loss giảm mạch lạc qua ranh giới (0.82 ở step 625 → 0.46 ở step 640), không vọt lên. Không lần đẩy checkpoint nào lỗi.
- 8.0 giây/step, GPU đỉnh 11.09 GiB. Train mất 3.1 giờ, xong lúc 01:31.
- **Phiên Kaggle bị hủy ("Cancelled") khoảng 04:50,** sau khoảng 8 giờ, khi đã chấm 13/18 snapshot. Tab Colab báo "Runtime disconnected"; không rõ nguyên nhân (tab Kaggle nằm ở trang settings, không mở trình soạn). Bật lại phiên, nối lại, chạy lại cell 1–5 rồi cell 6: nó bỏ qua train và các snapshot đã chấm, chấm nốt 5 snapshot cùng benchmark/knowledge trong khoảng 1 giờ 15 phút. Lần nối lại này, nút copy URL của Kaggle không đưa được gì vào clipboard (ba lần, có lẽ vì cửa sổ Chrome không có focus); người dùng phải tự dán.
- Trọng số: `trantam31/smollm2-360m-e1-seq-s1`. Output và metric: repo kết quả, thư mục `e1_seq_s1/` (private).

**Snapshot cuối (step 1250), test, CI bootstrap 95% theo cụm; số trong ngoặc tròn là E2 s1 / E1 s0:**

| | T-seen | T-unseen |
|---|---|---|
| `format_valid` | 0.99 [0.98, 1.00] (0.99 / 1.00) | 0.97 (0.98 / 0.98) |
| `call_exact` | 0.66 [0.60, 0.71] (0.67 / 0.65) | 0.81 (0.81 / 0.81) |
| `decision_bacc` | 0.993 (0.995 / 0.993) | 0.993 |
| `false_call_rate` | 0.01 [0.00, 0.03] (0.01 / 0.01) | 0.00 |
| `clarify_correct` | 0.46 [0.32, 0.58] (0.61 / 0.50) | 0.49 (0.58 / 0.52) |
| `grounded_correct` | 0.95 (0.95 / 0.94) | 0.96 |
| `stop_rate` C | 0.96 [0.93, 0.99] (0.96 / 0.94) | |

**Ranh giới (step 625, chỉ mới học chat):** `format_valid`, `call_exact`, `tool_name_acc`, `false_call_rate` đều bằng 0, CI so với base là [0, 0]; `decision_bacc` 0.50. `stop_rate` C 0.94 [0.90, 0.97], T-seen 0.575. `clarify_correct` 0.03. `grounded_correct` 0.27 [0.11, 0.50]; trong giai đoạn chat nó đi 0.26 → 0.46 → 0.20 → 0.29 → 0.08 → 0.27 (step 28 → 625), không đơn điệu, như seed 0.

**Giai đoạn tool, theo token từ ranh giới (T-seen; mốc +28k, +52k, +103k, +201k, +401k, +800k, +1.6M, +1.79M, cuối):**
- `false_call_rate`: **0.63**, 0.40, 0.24, 0.23, 0.18, 0.01, 0.01, 0.01, 0.01. Đỉnh ở +28k, cận dưới CI so với base +0.54 (seed 0: 0.64 ở +25k).
- `format_valid`: 0.48, 0.92, 0.99, 0.98, 0.99, 0.98, 0.99, 0.99, 0.99. Seed 0 ở +25k đã 0.87.
- `call_exact`: 0.15, 0.42, 0.53, 0.58, 0.59, 0.64, 0.65, 0.65, 0.66. Seed 0 ở +25k là 0.32.
- `decision_bacc`: 0.69, 0.80, 0.88, 0.88, 0.91, 0.99, 0.99, 0.99, 0.99.
- `clarify_correct`: 0.00, 0.01, 0.23, 0.38, 0.46, 0.44, 0.52, 0.46, 0.46.
- `grounded_correct` nhảy từ 0.38 (+28k) lên 0.83 (+52k), rồi 0.88–0.95.
- `stop_rate` C hụt xuống 0.83 ở +28k rồi về 0.93–0.97.

**t_reach (θ chính, T-seen, token cộng dồn):** `format_valid`, `tool_name_acc`, `decision_bacc`, `grounded_correct` 4.66M (+52k) → `call_exact` 4.71M (+103k) → `clarify_correct` không đạt θ 0.5 (θ 0.3 ở +201k). Seed 0: `call_exact` và `decision_bacc` cùng ở +100k; `clarify_correct` "unconfirmed" ở snapshot cuối.

**Bộ C, số output có `<tool_call>` trên 150 (đếm trên `outputs.jsonl`):** 0 ở mọi snapshot giai đoạn chat; **38** ở +28k, 19 ở +52k, 4 ở +103k, 3 ở +201k, 1 từ +401k tới cuối. Seed 0: 46, 45, 6, 3, rồi 1.

**Thứ tự dữ liệu, hiệu số ghép cặp E1 s1 − E2 s1** (`--compare e1_seq_s1 e2_mix_s1`, cùng mốc token cộng dồn, cùng prompt):
- **Cuối run, `clarify_correct` T-seen −15.0 điểm [−22.6, −8.6]; T-unseen −9.0 [−17.5, −3.0].** Ở 6.4M: −15.0 [−22.6, −8.6] và −10.0 [−17.9, −4.4]. Seed 0 (E1 s0 − E2 s0): −12.0 [−18.9, −6.2] và −9.0.
- Các metric gọi tool khác bằng nhau: `call_exact` T-seen −1.3 [−3.1, +0.4], T-unseen −0.3 [−2.4, +1.5]; `decision_bacc` −0.2 / −0.3; `false_call_rate` 0.0 / 0.0.
- Trước 6.4M, mọi hiệu số là "E1 chưa học tool" (vd. `call_exact` −45 tới −81 ở 200k–3.2M), không mang thông tin về thứ tự dữ liệu.

**Hiệu số ghép cặp E1 s1 − E1 s0** (`--compare e1_seq_s1 e1_seq_s0`; chỉ để mô tả): cuối run mọi metric chênh dưới 4 điểm. `call_exact` T-seen +0.7 [−1.0, +2.3], T-unseen −0.7 [−2.4, +0.9]; `clarify_correct` T-seen −4.0 [−9.5, 0.0], T-unseen −3.0 [−9.5, +1.0]; `decision_bacc` và `false_call_rate` dưới 1 điểm.

**H3, hiệu số ghép cặp T-rename − T-seen (`call_exact`, cuối run):** tên vô nghĩa giảm 10.0 điểm [5.3, 14.8] (E1 s0: 8.3; E2 s1: 8.7); tên đồng nghĩa giảm 2.3 [−1.0, 5.7].

**Quên (snapshot cuối):**
- Perplexity wikitext-2 14.10. So với E2 s1: −0.028 [−0.044, −0.011]. So với E1 s0: −0.007 [−0.017, +0.005]. So với base: +1.09 [+1.00, +1.20].
- KL tới base 0.067 (E2 s1: 0.069; E1 s0: 0.067); E1 s1 − E2 s1 là −0.0016 [−0.0020, −0.0012].
- ARC-Easy acc / acc_norm 0.706 / 0.672; HellaSwag 0.446 / 0.562. So với E2 s1: cả bốn ô CI chứa 0 (HellaSwag acc_norm −1.0 [−2.2, 0.0]). So với E1 s0: ARC-Easy acc_norm +0.8 [+0.2, +1.6], ba ô còn lại CI chứa 0. So với base: cả bốn ô CI chứa 0.

**Đối chiếu với dự đoán ghi trước khi chạy:**
1. **Ranh giới:** đúng cả bốn vế. Metric gọi tool bằng 0, CI so với base chứa 0; `decision_bacc` 0.50; `stop_rate` C 0.94, T-seen 0.575; `clarify_correct` 0.03; `grounded_correct` 0.27, dao động không đơn điệu.
2. **Thứ tự giai đoạn tool:** đúng phần lớn.
   - `format_valid` **không** đạt ở +25k: 0.48 ở +28k, sát dưới θ 0.5; đạt ở +52k. **Sai**, sát nút.
   - `call_exact` ở +103k (mốc 100k của thang từ ranh giới), `decision_bacc` ở +52k: đúng.
   - `clarify_correct` đến muộn nhất, không đạt θ chính: đúng.
3. **H1:** đúng ở θ chính trên T-seen: format (+52k) trước `call_exact` (+103k) một checkpoint. Ở θ 0.3 cũng trước một checkpoint (+28k so với +52k); ở θ 0.7 `call_exact` T-seen không đạt. Trên T-unseen, ở θ 0.3 và θ chính hai mốc trùng nhau (+52k). Vậy "không vững qua θ" đúng nếu tính cả T-unseen; riêng T-seen thì vững hơn seed 0.
4. **H2:** đúng. `decision_bacc` không muộn nhất. `false_call_rate` lên đỉnh 0.63 ở +28k, CI loại trừ 0; cuối run 0.01.
5. **H3:** đúng. Tên vô nghĩa giảm 10.0, trong 5–13; cận trên 14.8 < 20. Tên đồng nghĩa 2.3 < 5 (cận trên 5.7).
6. **Thứ tự dữ liệu:** đúng cả ba vế.
   - **`clarify_correct` T-seen −15.0 [−22.6, −8.6], trong khoảng −6 tới −16, CI loại trừ 0.** T-unseen cùng hướng, và CI cũng loại trừ 0.
   - Từ +800k tới cuối, `clarify_correct` T-seen của E1 đứng ở 0.44–0.52, trong khoảng 0.40–0.56.
   - `call_exact` chênh trong ±4, CI chứa 0; `decision_bacc` và `false_call_rate` chênh dưới 2 điểm.
7. **Chat bị kéo về định dạng tool:** đúng. 38/150 ở +28k (≥ 20); 1/150 từ +401k (≤ 3).
8. **Quên:** đúng cả bốn vế. Perplexity 14.10, hiệu số với E2 s1 −0.03, trong ±0.1 (CI loại trừ 0 nhưng rất nhỏ); KL 0.067; ARC/HellaSwag lệch E2 s1 dưới 1 điểm; `stop_rate` C 0.96.
9. **Tài nguyên:** 8.0 giây/step và 11.09 GiB, khớp. Tổng thời gian dài hơn dự đoán vì phiên bị hủy và phải nối lại.
- Không có kết quả "bất ngờ" nào trong danh sách đã ghi.

**Đọc (hai seed của E1, câu hỏi thứ tự dữ liệu ở hai seed):**
- **Khoảng chênh `clarify_correct` của E1 so với E2 lặp lại qua seed:** −12.0 ở seed 0, −15.0 ở seed 1 (T-seen), CI của cả hai loại trừ 0; T-unseen −9.0 ở cả hai seed. Đây là hiệu ứng thứ tự dữ liệu duy nhất ở cuối run, và nó lớn hơn độ lệch do thứ tự mẫu (E1 s1 − E1 s0 chỉ −4.0 [−9.5, 0.0]). Học chat trước rồi mới học tool làm model hỏi lại kém hơn khi thiếu thông tin, dù cùng dữ liệu và cùng số token.
- Mọi metric gọi tool khác ở cuối run không phụ thuộc thứ tự dữ liệu: `call_exact`, `decision_bacc`, `false_call_rate`, `grounded_correct` chênh dưới 2 điểm ở cả hai seed. Quên cũng vậy: perplexity và KL của E1 và E2 gần như trùng.
- Diễn biến sau ranh giới lặp lại gần như y hệt seed 0: `false_call_rate` lên đỉnh khoảng 0.63 rồi về 0.01 trong khoảng 800k token; chat bị kéo về định dạng tool (38–46/150) rồi hồi lại trong khoảng 400k. Seed 1 học định dạng chậm hơn seed 0 một chút ở +25k.

**Việc còn lại:** bộ P của E1 s1, E2 s1 và E3 s1 (chưa đọc). Các run seed 2 (E1, E2, E3 r64) và E3 r8 s0.
