# PLAN — Theo dõi model học được gì qua SFT

> Model: `HuggingFaceTB/SmolLM2-360M` (base)
> Mốc tham chiếu: `SmolLM2-360M-Instruct` (cho chat) · `SmolLM2-1.7B-Instruct` (cho tool, tùy chọn)
> Môi trường: Colab (chính, chạy từng cell) · máy local CPU (thử nhanh) · Kaggle (job dài / dự phòng)

---

## 1. Mục đích

Đo và hiểu **SFT thay đổi model như thế nào** khi dạy một base model khả năng hội thoại và tool calling:

- Model học thêm những khả năng gì, học theo thứ tự nào, nhanh hay chậm.
- Model mất đi những gì (quên kiến thức, đổi văn phong).
- Bên trong model, phần nào thay đổi (trọng số, xác suất, biểu diễn ẩn).

Vì vậy giao thức đánh giá được **chốt trước khi train** (mục 3), và mọi checkpoint đều được đo bằng đúng giao thức đó.

### Câu hỏi nghiên cứu (RQ)

| # | Câu hỏi | Đo bằng |
|---|---|---|
| RQ1 | SFT dạy *hình thức* (format, lượt hội thoại, biết dừng) hay dạy *năng lực*? Cái nào đến trước? | Đường cong metric theo lượng dữ liệu đã học |
| RQ2 | Các kỹ năng con của tool calling (đúng format, chọn đúng tool, điền đúng tham số, biết **không** gọi tool, hỏi lại khi thiếu thông tin, dùng kết quả tool) xuất hiện theo thứ tự nào? | Metric theo từng nhóm prompt |
| RQ3 | Model học **kỹ năng đọc schema** hay **nhớ tool**? | Nhóm chức năng chưa thấy + bài kiểm tra đổi tên |
| RQ4 | Model trả giá gì (quên kiến thức chung)? | Perplexity, benchmark nhỏ, KL so với base |
| RQ5 | Thay đổi nằm ở đâu bên trong model? | ΔW theo layer/module, embedding, linear probe |
| RQ6 | Full fine-tune khác LoRA thế nào về năng lực lẫn mức quên? | Lặp lại RQ1–RQ5 với LoRA |
| RQ7 *(tùy chọn)* | Cần bao nhiêu dữ liệu? Có đúng là "~1000 mẫu tốt là đủ" (LIMA)? | Đường cong theo số mẫu |

### Giả thuyết (ghi trước, kiểm chứng sau)

Mỗi giả thuyết có tiêu chí xác nhận. "Xuất hiện" được định nghĩa ở mục 3.5.

Mọi so sánh ở dưới đều dùng **khoảng tin cậy 95% của chính hiệu số** (bootstrap theo cặp/cụm, mục 3.5), không so hai khoảng tin cậy riêng lẻ với nhau.

- **H1** — Kỹ năng *format* (`format_valid`, `stop_rate`) xuất hiện sớm hơn kỹ năng *nội dung* (`call_exact`), cách nhau ít nhất một mốc checkpoint.
- **H2** — Trong các kỹ năng tool, *phân biệt khi nào gọi / không gọi* (`decision_bacc`) xuất hiện muộn nhất. Giai đoạn giữa train, `false_call_rate` tăng lên **cao hơn base** trước khi giảm xuống (model "gọi tool cho mọi thứ").
- **H3** — Trên T-unseen, `call_exact` cao hơn base few-shot (CI của hiệu số nằm hoàn toàn trên 0). Trên T-rename (đổi tên vô nghĩa), mức giảm so với T-seen **≤ 10 điểm** và cận trên CI của mức giảm **< 20 điểm**. Kết quả này được đọc là bằng chứng *hỗ trợ* khả năng sử dụng schema và cho biết mức phụ thuộc vào tên tool, **không** phải phân định dứt khoát giữa "đọc schema" và "nhớ tên" (xem mục 4.4).
- **H4** — Base model với few-shot đã làm được một phần format. SFT chủ yếu biến khả năng tiềm ẩn đó thành hành vi mặc định.
- **H5** — Full fine-tune làm model quên nhiều hơn LoRA. Bằng chứng *suy giảm* là perplexity wikitext tăng nhiều hơn **hoặc** điểm ARC-Easy/HellaSwag giảm nhiều hơn (CI của hiệu số loại trừ 0). KL so với base chỉ là chỉ báo đi kèm về độ lệch khỏi base: KL tăng cho thấy model *thay đổi*, chưa chứng minh model *quên*. LoRA r=64 đạt `call_exact` thấp hơn full fine-tune **không quá 5 điểm** (trung bình trên các seed).
- **H6** *(mô tả, không phải nhân quả)* — ΔW của full fine-tune tập trung năng lượng vào ít hướng: ở phần lớn module, rank-k với k ≤ 64 giữ ≥ 90% năng lượng. Kiểm tra mạnh hơn: thay ΔW bằng bản xấp xỉ rank-k rồi eval lại (mục 3.4). Kết quả LoRA **không** được dùng để xác nhận H6, vì rank thấp là do chính cấu trúc của adapter.

---

## 2. Thiết lập

### 2.1 Các model

| Vai trò | Model | Ghi chú |
|---|---|---|
| Điểm xuất phát | `SmolLM2-360M` (base) | |
| Mốc tham chiếu chat | `SmolLM2-360M-Instruct` | Model card **không** ghi bản 360M hỗ trợ function calling. Chỉ dùng làm mốc cho chat và mức quên. |
| Mốc tham chiếu tool *(tùy chọn)* | `SmolLM2-1.7B-Instruct` | Đạt 27% trên BFCL theo model card. Eval bằng prompt format riêng của nó, chỉ để định vị kết quả. |
| Các model ta tự train | E1 … E6 (mục 5) | |

### 2.2 Ngôn ngữ

- **Thí nghiệm chính dùng tiếng Anh**, vì SmolLM2 được huấn luyện chủ yếu bằng tiếng Anh. Dùng tiếng Việt sẽ trộn lẫn năng lực ngôn ngữ với năng lực tool calling.
- *(Tùy chọn)* Một bộ eval tiếng Việt riêng, báo cáo tách biệt, không dùng cho các kết luận chính.

### 2.3 Định dạng hội thoại (cố định cho mọi thí nghiệm)

Dùng kiểu ChatML. Tokenizer base đã có sẵn `<|im_start|>` và `<|im_end|>`, nhưng **không có** chat template.

System prompt có một **chính sách gọi tool cố định**, dùng chung cho dữ liệu train lẫn eval:

> *Call a tool only when the request needs information or an action that one of the provided tools offers. If you can answer directly from general knowledge, answer directly. If a required argument is missing, ask the user for it instead of guessing. If no provided tool fits a request that needs external information, say so.*

```
<|im_start|>system
You are a helpful assistant. {TOOL POLICY ở trên}
You have access to the following tools:
<tools>
[{"name": "get_weather", "description": "...", "parameters": {...}},
 {"name": "get_time", "description": "...", "parameters": {...}}]
</tools><|im_end|>
<|im_start|>user
What's the weather and local time in Hanoi?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "Hanoi"}}
</tool_call>
<tool_call>
{"name": "get_time", "arguments": {"city": "Hanoi"}}
</tool_call><|im_end|>
<|im_start|>tool
<tool_response>
{"name": "get_weather", "content": {"temp_c": 31, "condition": "sunny"}}
</tool_response>
<tool_response>
{"name": "get_time", "content": {"time": "14:05"}}
</tool_response><|im_end|>
<|im_start|>assistant
It's sunny and 31°C in Hanoi, and the local time is 14:05.<|im_end|>
```

Các quy định:

- **Gọi nhiều tool (parallel):** mỗi lời gọi nằm trong một tag `<tool_call>` riêng, các tag đặt liên tiếp nhau. Khi chuyển đổi, mảng `answers` của xLAM được tách thành nhiều tag theo đúng thứ tự gốc.
- **Thứ tự khi chấm:** các lời gọi song song được so sánh như **multiset**, không phụ thuộc thứ tự. Lời gọi tuần tự (lời gọi sau phụ thuộc kết quả lời gọi trước) nằm ở các lượt assistant khác nhau.
- **Kết quả tool:** gom vào **một** lượt `tool`, gồm nhiều khối `<tool_response>` theo đúng thứ tự lời gọi, mỗi khối có kèm `name`.
- **Thứ tự danh sách tool** trong system prompt được xáo ngẫu nhiên (với seed cố định) để model không học theo vị trí.
- `<tool_call>` và `<tool_response>` để ở dạng **text thường**. Biến thể thêm chúng thành special token là thí nghiệm E6.
- **Loss mask:** chỉ tính loss trên token của assistant, **gồm cả `<|im_end|>`** ở cuối mỗi lượt assistant. Toàn bộ system, user và tool được mask.

### 2.4 Cấu hình token dừng (bắt buộc, áp dụng cho mọi lần generate)

Base model mặc định có `eos_token = <|endoftext|>` (id 0), và `generation_config` cũng dừng ở id 0. Nếu không cấu hình lại, model sẽ không dừng ở `<|im_end|>`, và `stop_rate` sẽ phản ánh cấu hình thay vì hành vi mà model học được.

- Khi train: đặt `tokenizer.eos_token = "<|im_end|>"`. `pad_token` dùng lại token **có sẵn** `<|endoftext|>`, **không thêm token mới** (thêm token sẽ phải resize embedding, làm thay đổi điều kiện so sánh). Vị trí pad bị loại bằng attention mask và label = -100. Chat template được gắn vào tokenizer trong code.
- Khi generate (cho **mọi** model, kể cả base): `eos_token_id = [id(<|im_end|>), id(<|endoftext|>)]`.
- `stop_rate` **chỉ** đếm trường hợp dừng bằng `<|im_end|>`. Dừng bằng `<|endoftext|>` được đếm riêng (`eot_stop_rate`). Bị cắt vì chạm `max_new_tokens` là `truncated_rate`.

### 2.5 Quy tắc độ dài

- `max_seq_len = 1024` token, đo **sau khi** render template đầy đủ.
- Mẫu vượt giới hạn bị **loại bỏ, không cắt bớt**, vì cắt có thể làm mất schema, JSON hoặc `<|im_end|>`.
- Báo cáo tỉ lệ mẫu bị loại theo từng nguồn dữ liệu. Nếu một nguồn mất hơn 15% số mẫu, xem lại giới hạn trước khi train.
- Prompt eval cũng tuân theo cùng giới hạn này.

### 2.6 Môi trường

- **Local (CPU):** viết code, unit test, chạy thử train vài step (`MAX_STEPS=5`), phân tích kết quả.
- **Colab T4:** train và eval. Agent điều khiển qua Colab MCP theo từng cell. **Checkpoint được đẩy lên HF Hub hoặc Drive ngay khi lưu.**
- **Kaggle:** dùng cho job dài hoặc khi Colab hết quota.
- T4 **không hỗ trợ bf16** → dùng fp16 AMP (trọng số master vẫn ở fp32).

---

## 3. Hệ thống đánh giá (phần quan trọng nhất)

### 3.1 Dev và test

Đóng băng bộ eval **không** thay thế được việc tách dev và test:

- **Dev:** dùng để debug, kiểm tra metric và chọn cấu hình (lr, số epoch…). Được xem bao nhiêu lần cũng được.
- **Test:** dùng để đưa ra kết luận. **Không chạy test cho đến khi giao thức được khóa** (sau pilot, mục 6). Sau đó test được đo ở mọi checkpoint, nhưng không được dùng để chỉnh dữ liệu hay hyperparameter.
- Nếu buộc phải thay đổi thiết kế sau khi đã xem test: chạy lại các run **không** xóa được ảnh hưởng của việc đã xem. Ghi lý do vào `EXPERIMENTS.md`, và trong báo cáo đánh dấu những kết quả liên quan là **thăm dò (exploratory)**, không phải xác nhận.

### 3.2 Các bộ eval

Kích thước dưới đây là cho **test**. Dev có cùng cấu trúc, cỡ khoảng 1/3.

| Bộ | Kích thước (test) | Nội dung | Phục vụ |
|---|---|---|---|
| **P — Probe cố định** | ~60 prompt | Bộ prompt đọc bằng mắt qua mọi checkpoint (10 chat, 10 kiến thức, 40 tool chia theo 6 nhóm) | Quan sát định tính |
| **T-seen** | ~100 / nhóm | Tool calling với các nhóm chức năng **có** trong train (nhưng câu hỏi khác) | RQ2 |
| **T-unseen** | ~100 / nhóm | Tool calling với các **nhóm chức năng không hề có** trong train | RQ3 |
| **T-rename** | Bằng T-seen | T-seen sau khi đổi tên tool và tên tham số một cách nhất quán (mục 4.3) | RQ3 |
| **C — Chat** | ~150 | Câu hỏi và lệnh thông thường, held-out từ smoltalk | RQ1 |
| **G — Kiến thức chung** | wikitext-2 test + ARC-Easy / HellaSwag (~500 câu mỗi bộ) | Văn bản và benchmark chung | RQ4 |
| **BFCL** | Các nhóm simple / multiple / parallel / irrelevance | Benchmark chuẩn bên ngoài, chỉ chạy ở mốc chính | Định vị so với cộng đồng |

**6 nhóm prompt tool:**

1. **Single** — 1 tool, 1 lời gọi.
2. **Select** — nhiều tool, phải chọn đúng 1.
3. **Parallel** — gọi nhiều tool hoặc gọi nhiều lần trong một lượt.
4. **No-tool** — có cung cấp tool nhưng câu hỏi **không cần** dùng.
5. **Missing-info** — thiếu một tham số bắt buộc. Mỗi mẫu có nhãn `missing_param`, tức tên tham số đang thiếu.
6. **Multi-turn** — đã có kết quả tool. Mỗi mẫu có **đáp án có cấu trúc**: các giá trị cần xuất hiện, các giá trị không được xuất hiện, và một câu trả lời mẫu.

Nhóm Multi-turn và Missing-info lấy từ Glaive nhưng nhãn phải được viết hoặc rà lại bằng tay, vì đây là hai nhóm khó chấm tự động nhất.

*Quyết định (2026-10-23):* Glaive lặp prompt Missing-info nhiều (phía unseen chỉ khoảng 80 câu khác nhau), nên Missing-info lấy cùng số mẫu Glaive ở hai phía rồi bù bằng câu hỏi xLAM viết lại cho thiếu một tham số bắt buộc, đủ 133 mẫu mỗi phía. Seen và unseen nhờ đó có cùng tỉ lệ nguồn, test và dev giữ tỉ lệ này. Không để seen toàn Glaive còn unseen nửa xLAM, vì khác biệt nguồn sẽ lẫn vào so sánh seen/unseen.

### 3.3 Metric tự động

Quy ước: **mẫu số luôn là toàn bộ mẫu của nhóm**. Output lỗi hoặc không parse được tính là sai, không bị loại khỏi phép tính.

**Format / hành vi**
- `format_valid`: trên mọi mẫu *cần tool*, tỉ lệ output có ít nhất một `<tool_call>` và mọi `<tool_call>` đều chứa JSON hợp lệ với đủ key `name` và `arguments`.
- `stop_rate`, `eot_stop_rate`, `truncated_rate` (mục 2.4).
- `role_break_rate`: tỉ lệ model tự viết thêm lượt `<|im_start|>user` hoặc `system`.
- `avg_len`, `repetition_rate` (tỉ lệ 4-gram lặp lại).

**Tool calling** (chấm kiểu AST giống BFCL: so sánh cấu trúc, chuẩn hóa kiểu dữ liệu, không so chuỗi thô)
- **`call_exact`** *(metric chính)*: trên mọi mẫu cần tool, tập lời gọi khớp **hoàn toàn** với đáp án (tên tool, tham số, số lượng lời gọi, so dạng multiset).
- `tool_name_acc`: tập tên tool khớp đáp án. Mẫu số là toàn bộ mẫu cần tool.
- `args_f1`: F1 theo từng cặp key–value. Mẫu số là toàn bộ mẫu cần tool; mẫu không parse được tính F1 = 0.
- `hallucinated_tool`: tỉ lệ gọi tool **không có trong danh sách đã cung cấp** cho prompt đó.
- `false_call_rate`: nhóm No-tool, tỉ lệ có gọi tool. `missed_call_rate`: nhóm cần tool, tỉ lệ không gọi gì.
- **`decision_bacc`**: balanced accuracy của quyết định gọi / không gọi = ½ × (tỉ lệ gọi đúng ở nhóm cần tool + tỉ lệ không gọi ở nhóm No-tool). Một model không bao giờ gọi tool (như base zero-shot) hay luôn gọi tool đều chỉ được 0.5. Đây là metric chính cho kỹ năng "biết khi nào gọi", thay cho việc nhìn riêng `false_call_rate`.
- `clarify_correct` (nhóm Missing-info): đúng khi thỏa **cả ba**: (a) không gọi tool, (b) có hỏi lại, (c) nhắc đúng `missing_param` hoặc một từ đồng nghĩa trong danh sách. Chấm bằng luật; nếu không chắc chắn thì dùng LLM judge với rubric cố định. Tỉ lệ đồng ý giữa judge và chấm tay được kiểm tra trên 50 mẫu.
- `grounded_correct` (nhóm Multi-turn): đúng khi câu trả lời chứa đủ các giá trị bắt buộc, không chứa giá trị cấm, và không phủ định giá trị đúng. Chấm bằng luật kết hợp LLM judge theo rubric, kiểm định trên 50 mẫu như trên. Báo thêm điểm phụ chỉ trên các mẫu `meta.derivable = false`, tức các mẫu mà đáp án không tự tính được từ câu hỏi (xem quyết định dưới đây).
- *Quyết định (2026-10-23):* tool unseen của Glaive phần lớn là tính toán (tip, giảm giá, GCD, Fibonacci…), nên đa số mẫu Multi-turn unseen tự tính được mà không cần đọc kết quả tool. Hai phía nhắm cùng một tỉ lệ: số mẫu cần đọc kết quả tool bằng mức phía unseen có, phần còn lại là mẫu tự tính được. Nếu một phía không đủ mẫu tự tính được thì bù bằng mẫu cần đọc kết quả tool. Bản hiện tại: seen có 88 mẫu tự tính được trên 133, unseen có 99 trên 133 (nháp seen đã vét hết pool). Điểm phụ trên phần cần đọc kết quả tool mới là thước đo grounding thật; phần này ít mẫu nên khoảng tin cậy rộng. Đã cân nhắc đặt trần 25% mẫu tự tính được nhưng không khả thi: cả pool unseen chỉ có khoảng 87 prompt khác nhau loại cần đọc kết quả tool mà có giá trị chấm được.
- *Câu phủ định (2026-10-24):* một giá trị chỉ được tính là "có nêu" khi có ít nhất một lần xuất hiện mà không có từ phủ định (`not`, `no`, `never`, `n't`, `cannot`, `unable`…) trong 3 từ đứng ngay trước nó, cùng mệnh đề. Ví dụ "It is not 31 C" hay "Hotel A is not available" bị chấm sai; "The tip is $10, not $15" vẫn đúng với 10. Giá trị cấm bị phủ định ("not going to rain") không tính là vi phạm. Luật không bắt được phủ định đặt sau giá trị ("31 is wrong"). Trên câu trả lời mẫu, cả 266 mẫu Multi-turn đều chấm đúng. Phải so luật với chấm tay trên output thật ở pilot (50 mẫu, như trên) trước khi kết luận từ `grounded_correct`. Điểm phụ là `grounded_correct_not_derivable`.
- *Phải dừng đúng (2026-10-24):* `clarify_correct` và `grounded_correct` chỉ tính đúng khi output dừng bằng `<|im_end|>`. Output bị cắt ở `max_new_tokens` hoặc dừng bằng `<|endoftext|>` bị chấm sai. Lý do (E0 trên dev): base không sinh token điều khiển thật. Ở chỗ cần `<|im_end|>` / `<|im_start|>` nó sinh token khác như `tolist`, `ipython`, rồi tự viết tiếp các lượt user/system giả. Luật cũ chấm trên toàn bộ output nên cộng điểm cho cả chữ của lượt giả (ví dụ chỉ lặp lại câu hỏi của user cũng thành "hỏi lại đúng"). Cũng vì vậy `role_break_rate` không bắt được role break của base: luật chỉ nhận token thật. Câu trả lời mẫu viết tay được coi là đã dừng.
- Metric phụ của No-tool 3b là `no_fitting_said`: không gọi tool và nói rõ là không có tool phù hợp (bắt bằng mẫu câu như "none of the tools", "no suitable tool", "I don't have access to"…). Tính trên các mẫu `no_fitting_tool`. Câu trả lời mẫu của cả 44 mẫu 3b đều được nhận ra; 222 câu trả lời mẫu `plain`/`related` không câu nào bị nhận nhầm.
- *Lựa chọn LLM (2026-10-23):* các việc cần LLM khi dựng dữ liệu (lọc prompt No-tool, kiểm tool không phù hợp, tên đồng nghĩa cho T-rename, nhãn Missing-info/Multi-turn) do session agent gán nhãn một lần, không dùng model hay API nào khác. Vì không có judge chạy lại tự động theo checkpoint, `clarify_correct` và `grounded_correct` dựa vào luật; nhãn phải đủ chặt để luật chấm được (danh sách từ đồng nghĩa, giá trị cấm). Nếu luật không đủ, phần judge được bàn lại ở bước 5.

**Chat**
- `stop_rate`, `role_break_rate`, `avg_len`.
- *(Tùy chọn)* LLM judge chấm theo cặp so với bản 360M-Instruct trên bộ C (tỉ lệ thắng).

**Kiến thức chung / quên**
- Perplexity trên wikitext-2 test, tính trên văn bản thô, không qua template.
- ARC-Easy và HellaSwag (subset ~500 câu) bằng `lm-evaluation-harness`. Hai bộ này chấm theo log-likelihood nên công bằng với cả base model.
- KL(model_t ‖ base) trung bình theo token trên wikitext. Đây là chỉ báo **độ lệch** khỏi base, không phải thước đo quên. Kết luận về quên dựa trên perplexity và điểm benchmark.

### 3.4 Đo bên trong model

| Phép đo | Cách làm | Trả lời |
|---|---|---|
| **Likelihood của prefix chuẩn** | Ngay sau `<|im_start|>assistant\n`, tính log P của chuỗi chuẩn `<tool_call>\n{"name": "` (teacher forcing), tách theo nhóm cần tool và No-tool. Đại lượng này gộp cả việc học **cú pháp serialize**, nên **không** được diễn giải là "xác suất quyết định gọi tool". Luôn vẽ kèm tỉ lệ gọi tool thực tế (greedy) để đối chiếu. | RQ2 (mô tả) |
| **Likelihood dừng** | log P(`<|im_end|>`) tại vị trí kết thúc câu trả lời chuẩn | RQ1 |
| **ΔW theo layer** | ‖W_t − W_0‖_F / ‖W_0‖_F cho từng layer (32 layer) × module (q, k, v, o, gate, up, down, các norm), cộng thêm embedding → heatmap. SmolLM2-360M có `tie_word_embeddings = true`, nên embedding và `lm_head` là **một** ma trận và chỉ được tính một lần. | RQ5 |
| **Phổ của ΔW** | SVD của ΔW, đo tỉ lệ năng lượng mà k hướng đầu giữ lại (k = 8, 16, 64) | H6 (mô tả) |
| **Cắt rank ΔW** | Thay ΔW bằng bản xấp xỉ rank-k (k = 8, 64), eval lại `call_exact` và perplexity | H6 (kiểm tra mạnh hơn) |
| **Embedding drift** | Khoảng cách cosine giữa embedding trước và sau train của `<|im_start|>`, `<|im_end|>` và các token tạo nên `<tool_call>`, so với phân phối drift của toàn bộ vocab | RQ5 |
| **Linear probe** *(làm sau)* | Logistic regression trên hidden state để dự đoán "cần tool hay không", huấn luyện ở base và ở từng checkpoint | H4 |

### 3.5 Định nghĩa "kỹ năng xuất hiện" và khoảng tin cậy

Chốt trước khi train:

**Metric kỹ năng.** Mỗi kỹ năng được theo dõi bằng **một** metric, cao hơn là tốt hơn. Ngưỡng được đặt theo **ý nghĩa** của từng metric: cùng nằm trong [0, 1] không có nghĩa là cùng một ngưỡng mang cùng ý nghĩa. Riêng `decision_bacc` có mức ngẫu nhiên là 0.5 (chính sách luôn gọi hoặc không bao giờ gọi), nên ngưỡng của nó được đặt ở giữa mức ngẫu nhiên và mức hoàn hảo.

| Kỹ năng | Metric | θ chính | Bộ θ kiểm tra độ nhạy |
|---|---|---|---|
| Format | `format_valid` | 0.5 | {0.3, 0.5, 0.7} |
| Biết dừng | `stop_rate` | 0.5 | {0.3, 0.5, 0.7} |
| Chọn tool | `tool_name_acc` | 0.5 | {0.3, 0.5, 0.7} |
| Gọi đúng hoàn toàn | `call_exact` | 0.5 | {0.3, 0.5, 0.7} |
| Biết khi nào gọi | `decision_bacc` | **0.75** | {0.65, 0.75, 0.85} |
| Hỏi lại khi thiếu thông tin | `clarify_correct` | 0.5 | {0.3, 0.5, 0.7} |
| Dùng kết quả tool | `grounded_correct` | 0.5 | {0.3, 0.5, 0.7} |

**Hai mốc thời gian, báo cáo riêng, không bắt buộc phải có cả hai:**

- **t_reach — đạt ngưỡng hành vi:** checkpoint đầu tiên có metric ≥ θ và giữ được ở 2 checkpoint kế tiếp. Có thể bằng 0 (base đã đạt sẵn), và được báo cáo đúng như vậy.
- **t_improve — cải thiện nhờ SFT:** checkpoint đầu tiên có metric(t) − metric(base zero-shot) ≥ **0.2**, cận dưới CI của hiệu số > 0, và giữ được ở 2 checkpoint kế tiếp. Nếu base đã ≥ 0.8 (không còn đủ chỗ để tăng 0.2), ghi là **"gần trần ở base"** thay vì "không cải thiện". So thêm với base few-shot như một phân tích phụ.

Nếu điều kiện chỉ được thỏa ở một trong hai checkpoint cuối cùng (không còn đủ checkpoint để xác nhận), ghi là **"chưa xác nhận"**, không kết luận rằng kỹ năng chưa đạt.

Trong H1 và H2, "xuất hiện" nghĩa là **t_reach**. t_improve được báo cáo song song.

**Độ nhạy:** một kết luận về *thứ tự* xuất hiện (H1, H2) chỉ được coi là vững nếu thứ tự giữ nguyên ở cả ba ngưỡng trong bộ θ của từng metric.

**Khoảng tin cậy:**
- Bootstrap 95%, 2000 lần lấy mẫu lại. Mọi so sánh dùng **CI của hiệu số**.
- **Theo cặp:** so sánh trên cùng tập prompt (giữa hai checkpoint, giữa T-seen và T-rename là cùng prompt gốc, giữa hai run) thì lấy mẫu lại *cùng một tập chỉ số prompt* cho cả hai bên.
- **Theo cụm:** các prompt chung tool hoặc chung template nguồn được lấy mẫu lại **theo cụm** (cluster bootstrap), vì chúng không độc lập.
- **Tách hai nguồn bất định:** (a) bất định do tập prompt → bootstrap ở trên, trong từng seed; (b) biến thiên do seed huấn luyện → độ lệch chuẩn giữa 3 seed. Báo cáo cả hai, không gộp làm một.

### 3.6 Quy tắc chạy eval

- Decoding **greedy**, `max_new_tokens=256`, cấu hình dừng theo mục 2.4. Batch generate với left padding.
- Base model được eval theo hai cách:
  - **zero-shot** với template;
  - **few-shot** với 3 ví dụ cố định. Các ví dụ này dùng tool nằm ngoài mọi bộ eval.
- Các model Instruct được eval bằng chat template **riêng** của chúng. Ghi rõ trong bảng kết quả rằng format khác nhau.
- Kết quả được ghi vào `results/<run>/<checkpoint>/metrics.json`, sau đó gộp thành `results/all.csv` với các cột: `run, seed, step, tokens_seen, assistant_tokens_seen, split, eval_set, group, metric, value, ci_low, ci_high`.
- Output thô được lưu vào `results/<run>/<checkpoint>/outputs.jsonl` để xem lại và chấm lại khi cần.
- **Thời gian eval:** mục tiêu ≤ 10 phút mỗi checkpoint (bộ P + dev, hoặc test không gồm BFCL và G). Con số này **phải được đo ở pilot** (mục 6, Giai đoạn 2). Nếu vượt, giảm số checkpoint được chạy test đầy đủ, không giảm cỡ mẫu.

### 3.7 Lịch checkpoint và logging

- Checkpoint lưu theo thang log của **số token assistant đã học**: tương ứng khoảng step 0, 10, 25, 50, 100, 200, 400, 800, 1600, …, final.
- Trục x của mọi đường cong là **`assistant_tokens_seen`**, không dùng step, để so sánh được giữa các run có batch hoặc độ dài mẫu khác nhau. Mỗi checkpoint log cả `step`, `tokens_seen` và `assistant_tokens_seen`.
- Khi train, log (W&B hoặc TensorBoard và CSV): train loss, grad norm, lr, và **val loss tách riêng** cho chat dev và tool dev.
- *Cách làm (2026-10-24):* thứ tự mẫu của cả run được dựng trước (`src/train_data.py`, ghi `order.json`), nên biết trước số token assistant mỗi step. Snapshot để eval lưu ở step đầu tiên mà token assistant cộng dồn chạm 25k, 50k, 100k… (nhân 2, theo token tuyệt đối nên các run so được với nhau) và ở step cuối. Snapshot chỉ gồm trọng số fp32 (fp16 làm mất ΔW nhỏ), kèm `meta.json`, đẩy lên `snapshots/step-NNNNN/` của repo run. Val loss tính trên các mẫu dev có câu trả lời mẫu, tách chat (C), tool seen và tool unseen, ở cùng các mốc. `python -m src.eval.run_checkpoints` chấm lần lượt mọi snapshot trên dev; notebook `06_pilot` gọi cả hai bước. Pilot E2 10% có 125 step và 6 snapshot (step 5, 9, 19, 37, 76, 125).

---

## 4. Dữ liệu

### 4.1 Nguồn

| Nguồn | Dùng cho | Ghi chú |
|---|---|---|
| `HuggingFaceTB/smoltalk` | Chat | Lấy subset, chỉ tiếng Anh |
| `Salesforce/xlam-function-calling-60k` | Single / Select / Parallel | Gated. Mỗi mẫu gồm `query`, `tools`, `answers` (mảng các lời gọi). |
| `glaiveai/glaive-function-calling-v2` | Multi-turn, Missing-info | Định dạng khác, phải chuyển đổi và lọc kỹ |
| Tự tạo | No-tool | Xem mục 4.3. Không ghép tool ngẫu nhiên một cách mù quáng. |

### 4.2 Tách seen / unseen theo **nhóm chức năng**

Tách theo tên tool là chưa đủ, vì có những tool cùng chức năng nhưng khác tên (`get_weather` và `fetch_forecast` chẳng hạn).

1. Gom **mọi** tool của cả xLAM và Glaive vào một bảng. Chuẩn hóa tên (bỏ tiền tố, chuyển snake/camel case).
2. Embed chuỗi `name + description + tên tham số`, rồi phân cụm thành các **nhóm chức năng** (ví dụ: thời tiết, tài chính, toán học, tìm kiếm…). Rà lại bằng tay các cụm lớn.
3. Chọn khoảng 15% số cụm làm **unseen**. Chọn sao cho phân phối độ khó tương đương với phần seen: cùng phân phối số tham số, số tool trong danh sách và tỉ lệ các nhóm 1–6.
4. Loại khỏi tập train **mọi** mẫu có tool thuộc cụm unseen, **kể cả khi tool đó chỉ nằm trong danh sách được cung cấp mà không được gọi**.
5. Khi so sánh seen và unseen, báo cáo thêm kết quả sau khi **khớp theo độ khó** (stratify theo số tham số và số tool trong danh sách), để tách riêng ảnh hưởng của độ khó.
6. Embedding không tách sạch tuyệt đối các tool cùng chủ đề. Vì vậy khi dựng T-unseen từ pool `unseen_eval`, **bỏ mọi mẫu có tool nằm sát một tool seen** (cosine ≥ 0,85, cùng embedder). Bước lọc này chỉ áp dụng cho eval, không lấy thêm dữ liệu của train.

*Trần mỗi cụm cho T-unseen (2026-10-24):* pool `unseen_eval` phần lớn là cụm 29 (bài tính toán: sắp xếp, độ lệch chuẩn, số chính phương…), nên trước đây khoảng 60–73% mẫu single/select/parallel của T-unseen gọi tool cụm 29. Từ nay mỗi cụm chiếm tối đa **40%** mẫu của một nhóm, tính riêng test và dev, theo cụm của **tool được gọi** (tool nhiễu không tính). Mẫu đã chọn được giữ nếu còn trong trần. Phần thiếu được rút thêm từ pool sau mọi bước khác, nên nháp, nhãn và các bộ khác không đổi. Pool đủ mẫu: ngoài cụm 29, select còn khoảng 390 mẫu, single khoảng 540, parallel khoảng 940. Missing-info và Multi-turn không đặt trần, vì đã dưới 40% (38% và 26%). Kết quả nằm trong `eval_sets/REPORT.md`, mục "Cụm của tool được gọi".

Cách làm thực tế (cụm ghim sau khi duyệt tay, khối cụm, các cụm tạp bị loại, các luật kéo tool sang unseen) và kết quả kiểm tra nằm trong `configs/data.yaml` (mục `split`) và `data/SPLIT.md`.

### 4.3 Dữ liệu No-tool (cho cả train và eval)

Nếu chỉ ghép prompt chat với tool ngẫu nhiên rồi giữ nguyên đáp án, nhãn có thể **sai** theo hai cách: (a) tool ngẫu nhiên lại tình cờ phù hợp với yêu cầu, hoặc (b) câu hỏi thực ra cần thông tin bên ngoài (giá hiện tại, tin tức…), nên câu trả lời gốc là bịa. Khi đó model bị dạy bỏ qua tool một cách sai lệch.

**Cách xây:**
1. **Lọc prompt:** chỉ giữ prompt smoltalk trả lời được bằng kiến thức chung. Loại các prompt cần dữ liệu thời gian thực, dữ liệu cá nhân hoặc hành động bên ngoài. Lọc bằng LLM judge với rubric cố định.
2. **Chọn tool:** lấy danh sách tool từ các cụm chức năng **khác** chủ đề của prompt. LLM judge kiểm tra lại rằng *không tool nào trong danh sách giúp trả lời được yêu cầu*.
3. **Thêm ca khó**, chiếm khoảng 1/3 dữ liệu No-tool:
   - **Có vẻ liên quan nhưng không cần dùng:** tool cùng chủ đề nhưng câu hỏi mang tính khái niệm. Ví dụ có tool `get_weather` nhưng câu hỏi là "Why is the sky blue?" → trả lời trực tiếp.
   - **Cần thông tin bên ngoài nhưng không có tool phù hợp:** ví dụ hỏi giá cổ phiếu khi chỉ có tool thời tiết → nói rõ là không có công cụ phù hợp, theo đúng chính sách trong system prompt (mục 2.3). Câu trả lời mẫu được viết theo một mẫu cố định.
4. **Rà bằng tay:** kiểm tra ngẫu nhiên 200 mẫu train và **toàn bộ** mẫu No-tool trong dev/test. Ghi tỉ lệ nhãn sai tìm thấy vào `data/REPORT.md`. Nếu tỉ lệ sai trên 5%, sửa lại bộ lọc trước khi train.

Ca khó (3b) được tính vào nhóm No-tool khi chấm (đúng = không gọi tool). Tỉ lệ nói đúng là "không có tool phù hợp" được báo cáo thêm như một metric phụ.

*Bộ eval (2026-10-23):* mỗi phía T-seen/T-unseen có 133 mẫu No-tool, gồm 89 `plain`, 22 `related` (3a) và 22 `no_fitting_tool` (3b). Test và dev giữ tỉ lệ này. Tool đưa ra luôn lấy từ pool eval của đúng phía, và mỗi tên tool chỉ xuất hiện một lần trong một mẫu.
- **Prompt:** lấy từ chat SmolTalk giữ riêng, chỉ dùng lượt đầu. Session rà cả 800 prompt, giữ 704. Prompt bị bỏ là loại cần dữ liệu theo thời gian, dữ liệu cá nhân, thiếu ngữ cảnh, hoặc chỉ là câu chào.
- **Chọn tool:** `plain` lấy tool ở các cụm xa prompt nhất (theo embedding). `related` lấy tool ở cụm gần nhất. Glaive `no_call` không dùng được, vì cả pool chỉ có 23/35 prompt khác nhau, toàn kiểu "đặt vé, gọi pizza".
- **Câu hỏi 3b:** lấy câu hỏi xLAM cần tool từ pool seen, cho **cả hai phía**, kèm tool ở cụm xa tool gốc. Pool unseen gần như toàn bài tính toán, mà loại đó model tự làm được.
- **Rà từng nháp:** đạt khi không tool nào giúp trả lời được. Với 3b còn phải là câu hỏi thật sự cần dữ liệu hoặc hành động bên ngoài. Toàn bộ mẫu dùng trong dev/test đã được rà.
- **Tỉ lệ nháp bị bỏ:** `related` khoảng 2–9%, vì tool cùng cụm đôi khi làm được việc (giải phương trình, dịch, kiểm ngữ pháp). `no_fitting_tool` khoảng 1/3, vì câu hỏi tự trả lời được.
- **Hạn chế:** phía unseen có ít cụm không phải tính toán, nên tool trong `plain` unseen lặp lại khá nhiều.

*Tập train (2026-10-24):* No-tool chiếm 20% D_tool (2000 mẫu: 1334 `plain`, 333 `related`, 333 `no_fitting_tool`), dựng bằng đúng hàm nháp của bộ eval nhưng chỉ dùng tool seen trong pool train, prompt chat ngoài D_chat và câu hỏi xLAM ngoài D_tool. Lọc prompt và kiểm tool gộp làm một lượt gán nhãn: subagent duyệt từng nháp theo tiêu chí ở trên. Nhãn gắn với id cùng tên các tool được đưa ra (`data/train/labels/no_tool.jsonl`, trên Hub ở `train/labels/`), vì nhãn chỉ đúng cho bộ tool mà người duyệt đã thấy. Mỗi nháp có RNG riêng theo id, nên thay đổi ở nhóm khác không xáo lại các nháp. Số liệu nằm trong `data/TRAIN.md`.

### 4.4 Bài kiểm tra đổi tên (T-rename)

Lấy T-seen, đổi tên tool và tên tham số **một cách nhất quán** trong schema, câu hỏi (nếu có nhắc tới) và đáp án. Có hai mức:

- **Đổi sang từ đồng nghĩa:** ví dụ `get_weather → fetch_forecast`, `city → location`.
- **Đổi sang tên vô nghĩa:** ví dụ `get_weather → fn_k3x`, `city → p1`. Phần description vẫn giữ nguyên.

*Mức đồng nghĩa (2026-10-24):* session gán một lần bảng `rename_synonyms` (nhãn trên Hub, `eval_sets/labels/`) cho mọi tên trong single/select/parallel của T-seen: 666 tool và 685 tham số. Mỗi tên tham số có một tên mới dùng chung. Có 18 chỗ ghi đè riêng theo tool, khi nghĩa khác nhau (ví dụ `b` là cận trên của tích phân nhưng là kênh xanh lam trong `rgb_to_hsl`). Tên mới đổi các từ chính, chỉ giữ tên riêng và con số (`tiktok`, `54`). Bộ được dựng từ đúng các mẫu T-seen, nên so theo cặp được với T-seen và T-rename mức vô nghĩa. Bước dựng chặn upload khi bảng thiếu tên, khi tên mới chỉ là cách viết khác của tên cũ (cùng tập từ), hoặc khi hai tên gốc khác nhau trong một mẫu nhận cùng một tên mới. Mẫu nguồn vốn có hai tool trùng tên thì được giữ trùng.

Kết quả được diễn giải là bằng chứng **hỗ trợ** khả năng sử dụng schema và cho biết **mức phụ thuộc vào tên tool**, không phải phân định dứt khoát. Lý do: điểm giảm có thể đến từ việc tên vô nghĩa khó tokenize hoặc khó sao chép, và điểm giữ nguyên cũng chưa loại trừ việc model nhớ phần mô tả hoặc cấu trúc schema.

### 4.5 Quy tắc chung

- Khử trùng lặp (exact và near-duplicate theo MinHash) giữa train và **mọi** bộ dev/test, kể cả bộ P. Bước tách hiện mới khử trùng train với các pool eval, nên khi dựng D_chat/D_tool phải khử trùng thêm với `eval_sets/p.jsonl`. P không trùng prompt nào của các bộ eval.
- Áp dụng quy tắc độ dài ở mục 2.5, ghi tỉ lệ bị loại vào `data/REPORT.md`.
- Tất cả được chuyển về một format chung và lưu thành `data/processed/*.jsonl`. Ghi lại seed và hash cho từng file. Tập train cố định D_chat/D_tool nằm ở `data/train/{d_chat,d_tool}.jsonl` (`python -m src.data.train_sets`, báo cáo và hash ở `data/TRAIN.md`), vì `data/processed/` đã chứa dữ liệu nguồn đã chuyển đổi.
- Unit test bao gồm: render template, mask (có `<|im_end|>` của assistant), parser `<tool_call>` (một lời gọi, nhiều lời gọi, JSON hỏng), và scorer AST.

---

## 5. Thí nghiệm

### 5.1 Kiểm soát điều kiện so sánh

Hai tập dữ liệu train được **cố định một lần** và dùng chung cho mọi thí nghiệm:

- **D_chat**: N_chat mẫu chat.
- **D_tool**: N_tool mẫu tool, gồm khoảng 20% No-tool, kèm Multi-turn và Missing-info.

Số lượng cụ thể sẽ chốt sau pilot, dự kiến N_chat ≈ N_tool ≈ 10k. Mọi run đều có **mỗi mẫu được học đúng 2 epoch**.

| ID | Khởi đầu từ | Dữ liệu (thứ tự) | Optimizer / scheduler |
|---|---|---|---|
| **E0** | — | Không train. Eval base (zero-shot và few-shot) và các model Instruct. | — |
| **E1 — Seq** | base | **Một run duy nhất**: D_chat × 2 epoch, **sau đó** D_tool × 2 epoch | **Một** lịch lr toàn cục (warmup 3% + cosine trên toàn bộ run), optimizer **giữ nguyên** qua ranh giới chat → tool |
| **E2 — Mix** | base | D_chat ∪ D_tool trộn ngẫu nhiên, 2 epoch | Đúng lịch lr toàn cục như E1 |
| **E3 — LoRA** | base | Như E2 | Như E2, lr riêng cho LoRA. r ∈ {8, 64} |
| **E4** *(tùy chọn)* | base → checkpoint chat | Quy trình hai giai đoạn thực tế: train D_chat với lịch riêng đến hết, sau đó **reset** optimizer và scheduler rồi train D_tool | Hai chu kỳ warmup + cosine |
| **E5** *(tùy chọn)* | base | D_chat ∪ subset D_tool ∈ {100, 500, 2k, N_tool} | Như E2 |
| **E6** *(tùy chọn)* | base | Như E2, thêm `<tool_call>` và `</tool_call>` làm special token | Như E2 |

**Diễn giải các so sánh:**

- **E1 và E2** có cùng dữ liệu, cùng số lần tiếp xúc, cùng tổng số token, cùng lịch lr và cùng một optimizer liên tục. Chỉ khác **thứ tự dữ liệu**, nên chênh lệch giữa hai run được quy cho thứ tự.
- **E1 và E2 dùng chung mọi thiết lập huấn luyện**, ghi trong cùng một file config gốc mà hai run kế thừa: cùng tập mẫu, cùng effective batch, cùng cách chuẩn hóa loss (tổng loss chia cho số token assistant trong batch), **không packing**, cùng xử lý batch cuối (`drop_last=True` ở từng giai đoạn). Chỉ khác **thứ tự** đưa mẫu vào: E1 xáo trộn trong từng giai đoạn (chat, rồi tool), E2 xáo trộn toàn bộ. Mỗi seed lưu lại danh sách thứ tự mẫu để kiểm chứng.
- **Checkpoint tại ranh giới chat → tool của E1** chính là điểm "chỉ học chat" (trả lời câu hỏi: học chat có tự sinh ra chút khả năng tool nào không). Lưu ý: tại điểm này lr chưa giảm hết, khác với một run chỉ học chat được train trọn vẹn.
- **E4 so với E2** là so sánh **hai quy trình huấn luyện**: khác cả thứ tự lẫn động lực tối ưu (hai chu kỳ lr, reset optimizer). Không được quy toàn bộ chênh lệch cho thứ tự.
- **Bộ đếm token là cộng dồn** qua mọi giai đoạn: các checkpoint giai đoạn tool của E1 và E4 có `assistant_tokens_seen` gồm cả phần chat đã học.
- **Seed:** mỗi seed của E1 hay E4 là **toàn bộ quy trình** từ base, gồm cả giai đoạn chat. Không dùng chung một checkpoint chat cho nhiều seed của giai đoạn tool.

### 5.2 Siêu tham số mặc định

- **Full fine-tune:** lr 2e-5, cosine, warmup 3%, seq len 1024, effective batch 32, fp16 AMP, gradient checkpointing, grad clip 1.0.
- **LoRA:** lr 2e-4, alpha = 2r, dropout 0.05. Gắn vào **`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`** của cả 32 layer. **Không** gắn vào embedding / `lm_head` (hai ma trận này là một vì tied), và cũng không train chúng (không dùng `modules_to_save`). Đây là điểm khác cố ý so với full fine-tune, nơi embedding có thay đổi. Phải nhắc tới điều này khi so sánh embedding drift.
- Chọn lr bằng một sweep nhỏ **trên dev** (ví dụ {1e-5, 2e-5, 5e-5} cho full fine-tune), sau đó cố định cho mọi run.

### 5.3 Khả năng tái lập

- Ghi lại: `requirements.lock` (phiên bản chính xác của `torch`, `transformers`, `trl`, `peft`, `datasets`…), **revision (commit hash)** của từng model và dataset trên HF Hub, cùng config đầy đủ của từng run trong `results/<run>/config.yaml`.
- Checkpoint để resume lưu **đầy đủ trạng thái**: trọng số, optimizer, scheduler, GradScaler (fp16), trạng thái RNG (Python, NumPy, torch, CUDA), vị trí trong data loader. Đặt `save_only_model=False` và kiểm tra trong pilot rằng một run bị dừng giữa chừng rồi resume cho đường loss trùng với run chạy liền mạch.
- Checkpoint chỉ dùng để eval có thể lưu chỉ trọng số, để tiết kiệm dung lượng.

---

## 6. Các giai đoạn và checklist

### Giai đoạn 0 — Hạ tầng
- [x] Khung repo (mục 8), `requirements.txt`, `requirements.lock`, config YAML (`configs/models.yaml`, `configs/smoke.yaml`; config E1–E6 viết khi thiết kế xong ở Phase 1–2)
- [x] Load base và các model Instruct, chạy thử generate trên CPU **với cấu hình dừng ở mục 2.4** (chạy trên Colab CPU; máy local không đủ RAM)
- [x] Module chat template, loss mask, parser, scorer cấu trúc ban đầu, kèm unit test (chưa phải evaluator BFCL đầy đủ; quy tắc metric nghiên cứu sẽ khóa sau pilot)
- [x] `train.py` chạy được `MAX_STEPS=5`, đẩy checkpoint lên Hub, resume được, log token đã học (local chỉ test với Llama nhỏ vì thiếu RAM; chạy thật trên Colab T4)
- [x] Notebook Colab dùng chung code trong `src/` (không viết logic train trong notebook)

### Giai đoạn 1 — Dữ liệu và bộ eval
- [x] Tải và chuyển đổi dữ liệu, áp quy tắc độ dài, ghi `data/REPORT.md`
- [x] Phân cụm nhóm chức năng, tách seen/unseen, khử trùng lặp
- [x] Tạo T-rename (hai mức), viết bộ P bằng tay (`src/eval/probe_set.py` → `eval_sets/p.jsonl`), gán nhãn hoặc rà nhãn cho Missing-info và Multi-turn
- [x] Viết các metric ở mục 3.3, test trên output viết tay (có cả đúng lẫn sai). Không dùng LLM judge (quyết định ở mục 3.3); việc so luật với chấm tay trên 50 output thật dời sang pilot. Phép đo bên trong model (mục 3.4) viết khi có checkpoint.
- [x] Cài `lm-evaluation-harness` (`src/eval/benchmarks.py`, notebook `03_eval`), chạy được trên SmolLM2-360M-Instruct
- [x] Cài BFCL (`src/eval/bfcl.py`, notebook `04_bfcl`), chạy được trên SmolLM2-360M-Instruct. BFCL không có handler SmolLM2, nên dự án đăng ký một handler dùng đúng prompt tool của SFT; sinh output bằng vLLM, chấm bằng AST checker chính thức. Trên T4 mất khoảng 20 phút cho 1040 mẫu. Với prompt này, 360M-Instruct không xuất `<tool_call>` lần nào: simple/multiple/parallel đạt 0%, irrelevance 100%. Phép kiểm đối chứng (đáp án BFCL viết thành `<tool_call>`) đạt 98.5–100%; phần trượt là do cách dựng đáp án đối chứng, không phải do decode.
- [x] **Đóng băng** dev/test và P: ghi hash vào `eval_sets/MANIFEST.md` (`python -m src.eval.manifest` để kiểm)

### Giai đoạn 2 — Pilot (E0 + E2 ngắn, **chỉ dùng dev**)
Mục đích: kiểm tra hạ tầng và kiểm tra **scorer chấm đúng**, **chưa** để kết luận. Pilot không đặt yêu cầu nào về kết quả học: một pilot ngắn không cải thiện gì cũng là kết quả hợp lệ, và không phải lý do để sửa metric. Ngoại lệ duy nhất là **tiêu chí đi/dừng về độ đủ mạnh của model** dưới đây, chốt từ trước pilot: nó chỉ quyết định dùng model nào, không bao giờ dùng để sửa metric hay ngưỡng.
- [x] Eval E0 trên dev (base zero-shot / few-shot, các model Instruct)
- [x] Chạy E2 với khoảng 10% dữ liệu, 1 seed, eval dev ở các checkpoint (EXPERIMENTS.md, "Pilot: E2 10%")
- [ ] Đo: thời gian train mỗi 1k mẫu, thời gian eval mỗi checkpoint, bộ nhớ GPU
- [ ] Kiểm tra bằng mắt: mask đúng, model dừng đúng `<|im_end|>`, parser không bỏ sót
- [ ] **Kiểm tra độ nhạy của scorer** bằng một bộ output **dựng có chủ đích** cho từng nhóm: output đúng, và output mắc từng lỗi đã biết (sai tên tool, thiếu hoặc thừa tham số, sai kiểu dữ liệu, JSON hỏng, gọi tool khi không cần, hỏi lại sai tham số, trả lời phủ định giá trị đúng…). Mỗi lỗi phải bị chấm sai, mỗi output đúng phải được chấm đúng.
- [ ] Rà bằng tay khoảng 50 output thật của pilot, so điểm của scorer với điểm chấm tay
- [ ] Kiểm tra resume (mục 5.3)
- [ ] **Tiêu chí đi/dừng cho model** (chỉ dùng dev, đọc ở checkpoint cuối của pilot E2, so với base zero-shot):
  - **(a) Lõi:** `format_valid` và `call_exact` của nhóm Single trên T-seen đều tăng ≥ 0.2, cận dưới CI của hiệu số > 0.
  - **(b) Nhóm khó:** ít nhất một trong `call_exact` T-unseen (Single), `clarify_correct`, `grounded_correct` có cận dưới CI của hiệu số > 0.
  - **Đạt (a) và (b):** giữ SmolLM2-360M.
  - **Đạt (a), không đạt (b):** giữ 360M, nhưng đánh dấu RQ3 và các kỹ năng khó là **"nguy cơ sàn"**; chạy thêm một pilot ngắn với phương án dự phòng 1 để so trước khi khóa giao thức.
  - **Không đạt (a):** trước hết tìm lỗi hạ tầng (mask, template, cấu hình dừng, parser). Chỉ khi hạ tầng đã đúng mà vẫn không đạt thì đổi model.
  - **Phương án dự phòng:** (1) SmolLM2-1.7B base với LoRA: giữ nguyên tokenizer, ChatML và pipeline; full fine-tune 1.7B không vừa T4 (trọng số + Adam ≈ 27 GB) nên RQ6 cần GPU lớn hơn hoặc bị thu hẹp. (2) Một base model khác cỡ ~0.5B: phải kiểm tra lại tokenizer, token dừng và chạy lại các kiểm tra Phase 0.
  - Đổi model thì chạy lại E0 và pilot trên model mới, ghi quyết định và lý do vào `EXPERIMENTS.md` trước khi khóa giao thức.
- [ ] Chốt N_chat, N_tool, lr, lịch checkpoint, các ngưỡng ở mục 3.5
- [ ] **Khóa giao thức:** ghi toàn bộ quyết định vào `EXPERIMENTS.md` kèm ngày và commit hash. Chỉ từ thời điểm này mới được chạy test.

### Giai đoạn 3 — Thí nghiệm chính (E0 test, E1 → E3)
- [ ] Eval E0 trên test
- [ ] Với mỗi run: ghi giả thuyết vào `EXPERIMENTS.md` **trước** khi chạy
- [ ] Train, eval test ở mọi checkpoint, ghi vào `results/all.csv`
- [ ] Đọc output bộ P qua các checkpoint, ghi chú lúc kỹ năng bắt đầu xuất hiện
- [ ] Chạy 3 seed cho E1, E2, E3 (mỗi seed là toàn bộ quy trình từ base)

### Giai đoạn 4 — Mở rộng *(nếu còn thời gian)*: E4, E5, E6, linear probe, bộ eval tiếng Việt

### Giai đoạn 5 — Phân tích và báo cáo
- [ ] Notebook phân tích, sinh toàn bộ biểu đồ từ `results/`
- [ ] Thí nghiệm cắt rank ΔW (H6)
- [ ] `REPORT.md`: trả lời từng RQ, đối chiếu H1–H6 (đúng / sai / chưa rõ), kèm khoảng tin cậy

---

## 7. Kết quả cần có cuối cùng (Definition of Done)

1. **Bảng tổng:** base (zero-shot / few-shot) · E1 · E2 · E3 (· E4 nếu có) · các model Instruct, so trên mọi metric chính, có khoảng tin cậy và độ lệch giữa các seed.
2. **Đường cong năng lực:** trục x là `assistant_tokens_seen` (thang log, cộng dồn), mỗi đường là một kỹ năng con, đánh dấu điểm xuất hiện và độ nhạy theo ngưỡng (mục 3.5). Đây là biểu đồ trung tâm, trả lời RQ1 và RQ2.
3. **Likelihood của prefix chuẩn và tỉ lệ gọi thực tế**, cùng `decision_bacc`: nhóm cần tool so với nhóm No-tool, theo trục token.
4. **Seen / unseen / rename:** CI của hiệu số (bootstrap theo cặp và theo cụm), kèm bản đã khớp theo độ khó (RQ3).
5. **Năng lực và mức quên:** `call_exact` so với perplexity và điểm benchmark (mức quên), kèm KL (độ lệch khỏi base), cho E1, E2, E3 (RQ4, RQ6). Biểu đồ riêng E1 so với E2 cho ảnh hưởng của thứ tự dữ liệu.
6. **Heatmap ΔW**, phổ singular value, và kết quả eval sau khi cắt rank (RQ5, H6).
7. **Bảng so sánh định tính:** khoảng 5 prompt tiêu biểu, đặt output qua các checkpoint cạnh nhau.
8. `REPORT.md`: kết luận cho từng RQ, mỗi kết luận đi kèm biểu đồ làm bằng chứng.

---

## 8. Cấu trúc repo dự kiến

```
PLAN.md                 kế hoạch này
EXPERIMENTS.md          nhật ký: giả thuyết trước run → kết quả sau run
REPORT.md               báo cáo cuối
configs/                models.yaml (model + revision), smoke.yaml, e1_seq.yaml, e2_mix.yaml, …
data/
  REPORT.md             thống kê dữ liệu, tỉ lệ mẫu bị loại, kết quả phân cụm
src/
  data/                 tải, chuyển format, phân cụm và tách seen/unseen, rename, khử trùng lặp
  chat_format.py        system prompt, tool policy, ChatML cho một lượt
  sft_data.py           chat template, loss mask, kiểm tra độ dài, padding
  tool_calls.py         serialize, parse và chấm cấu trúc <tool_call>
  playground.py         nạp model theo configs/models.yaml, generate với cấu hình dừng
  train.py              SFT (transformers Trainer), đẩy checkpoint lên Hub, resume, đếm token
  eval/
    generate.py         generate theo batch cho P / T / C
    scorers.py          scorer AST, clarify, grounded (luật + LLM judge)
    logprobs.py         likelihood prefix, likelihood dừng, KL
    forgetting.py       perplexity, gọi lm-eval-harness
    weights.py          ΔW, SVD, cắt rank, embedding drift
    stats.py            bootstrap CI, phát hiện điểm xuất hiện
    probe.py            linear probe
  run_eval.py           chạy toàn bộ eval cho một checkpoint, ghi vào results/
eval_sets/              dev/ và test/ đã đóng băng + MANIFEST.md
notebooks/
  00_base_playground.ipynb, 01_train_smoke.ipynb   sinh từ src/ bằng scripts/build_playground_notebook.py
  analysis.ipynb        sinh biểu đồ cho báo cáo
results/                metrics.json, outputs.jsonl, all.csv
tests/                  unit test cho format, mask, parser, scorer
```

---

## 9. Rủi ro và cách xử lý

| Rủi ro | Cách xử lý |
|---|---|
| `stop_rate` phản ánh cấu hình thay vì hành vi học được | Cấu hình dừng thống nhất (mục 2.4), đếm riêng từng kiểu dừng |
| Chênh lệch Seq/Mix đến từ lượng dữ liệu hoặc lịch lr | E1 và E2 dùng cùng dữ liệu, cùng epoch, cùng một lịch lr toàn cục và optimizer liên tục. Quy trình hai giai đoạn (E4) được tách riêng và diễn giải là so sánh quy trình (mục 5.1). |
| Pilot kéo metric theo kết quả mong muốn | Pilot chỉ kiểm tra scorer bằng output dựng có chủ đích và chấm tay. Không yêu cầu pilot phải cải thiện. |
| Nhãn No-tool sai | Lọc prompt, chọn tool khác cụm, kiểm bằng judge, rà bằng tay (mục 4.3) |
| Không resume được đúng trạng thái | Lưu đầy đủ trạng thái optimizer/scheduler/scaler/RNG, kiểm tra resume trong pilot (mục 5.3) |
| Unseen bị lọt hoặc khó hơn seen | Tách theo cụm chức năng, kiểm tra cả danh sách tool, khớp theo độ khó, thêm T-rename |
| Metric đẹp nhờ loại bỏ output lỗi | Mẫu số luôn là toàn bộ mẫu. Metric chính là `call_exact`. |
| Overfit vào bộ test do xem đường cong nhiều lần | Chọn cấu hình trên dev. Thay đổi sau khi đã xem test phải ghi lại và chạy lại. |
| Kết luận dựa trên nhiễu | Bootstrap CI, 3 seed cho so sánh chính |
| Colab ngắt session, hết quota | Đẩy checkpoint lên Hub/Drive ngay khi lưu, train hỗ trợ resume, có Kaggle dự phòng |
| Glaive khác format, nhãn nhiễu | Lọc mạnh, rà nhãn bằng tay cho các bộ eval. Nếu vẫn kém thì tự sinh một tập nhỏ. |
| Eval quá lâu | Đo ở pilot. Nếu vượt, giảm số checkpoint chạy test đầy đủ, không giảm cỡ mẫu. |
| fp16 không ổn định (loss NaN) | Giảm lr, grad clip 1.0, trọng số master ở fp32 |
| Phạm vi phình to | Giai đoạn 4 là tùy chọn. Hoàn thành E0–E3 trước. |
| Model quá yếu: metric nằm sát đáy ở mọi checkpoint (floor effect), nhất là T-unseen, Missing-info, Multi-turn | Tiêu chí đi/dừng chốt trước pilot (Giai đoạn 2), chỉ dùng dev; phương án dự phòng SmolLM2-1.7B + LoRA hoặc base ~0.5B khác. Kỹ năng vẫn sát đáy ở thí nghiệm chính được báo cáo là "không phân giải được ở cỡ model này", không phải "không học được". |
