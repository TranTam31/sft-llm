# Phase 0 — Format dữ liệu SFT và loss mask

Phần này nối tiếp [playground base model](phase-0-base-playground.md), tương ứng các mục **9–12** trong [notebook Phase 0](../notebooks/00_base_playground.ipynb). Mục tiêu là kiểm tra dữ liệu mà model sẽ học trước khi viết vòng lặp train.

## Dữ liệu gồm những gì?

Mỗi ví dụ có danh sách schema `tools` và các lượt `turns`. Mỗi lượt gồm `role` và `content`. Hàm `prepare_messages()` thêm system prompt với cùng chính sách gọi tool dùng trong playground.

Các ví dụ viết tay nằm trong `src/sft_examples.py`:

- `chat`: một yêu cầu và câu trả lời thông thường.
- `tool_result`: assistant gọi tool, nhận kết quả và trả lời người dùng.
- `parallel`: assistant gọi hai tool trong cùng lượt; kết quả được gom vào một lượt tool.
- `missing_info`: assistant hỏi lại thay vì đoán tham số còn thiếu.

Kết quả thời tiết và thời gian trong các ví dụ được giả lập. Các ví dụ này dùng để kiểm tra format, không phải dữ liệu thời gian thực hoặc bộ train/dev/test đã chốt.

## Model nhìn thấy gì và học gì?

`input_ids` chứa toàn bộ hội thoại đã render: system, user, assistant và tool. `attention_mask` giữ các token thật và loại vị trí pad khỏi attention.

`labels` quyết định những vị trí được tính loss:

- Nội dung assistant và `<|im_end|>` kết thúc lượt assistant: label bằng token ID gốc.
- System, user, tool, các role header và newline phân cách: label `-100`.
- Vị trí padding: label `-100` và attention mask 0.

Mask loss không xóa ngữ cảnh. Model vẫn nhìn thấy yêu cầu và kết quả tool để học cách trả lời dựa trên chúng. Chỉ phần dự đoán tại những vị trí bị mask không đóng góp trực tiếp vào loss.

Labels giữ cùng chỉ số với input. Causal language model tự dịch labels khi tính loss: dự đoán token tiếp theo từ những token trước đó. Không shift labels thêm một lần trong bộ chuẩn bị dữ liệu.

## Template và token dừng

`src/sft_data.py` gắn một chat template có vùng `{% generation %}` cho từng nội dung assistant, gồm cả `<|im_end|>`. Tokenizer trả assistant mask từ vùng này; code dùng mask để tạo labels.

Tokenizer dùng các token có sẵn:

- EOS cho dữ liệu SFT: `<|im_end|>`.
- Padding: `<|endoftext|>`.
- `<tool_call>` và `<tool_response>` là text thường.

Không thêm token hoặc resize embedding. Phần kiểm tra SFT dùng bản copy tokenizer để giữ tokenizer đang dùng trong playground độc lập.

## Thử trong Colab

Notebook: [00_base_playground.ipynb](../notebooks/00_base_playground.ipynb), tải lên Colab như hướng dẫn playground.

1. Chạy các mục chuẩn bị mã nguồn, kiểm tra code và nạp base như hướng dẫn playground. Nếu notebook đã có `lab`, có thể tiếp tục tại mục 9 sau khi source bundle đã được cập nhật.
2. Tại mục **9**, chọn `EXAMPLE_INDEX`: 0 là chat, 1 là tool result, 2 là parallel, 3 là missing-info.
3. Chạy mục **10** để xem token `LEARN` và `CONTEXT`. Thay `START` / `STOP` để xem vùng khác của hội thoại.
4. Đọc danh sách các vị trí `<|im_end|>`: với ví dụ tool result có năm lượt tổng cộng, chỉ hai token kết thúc lượt assistant mang nhãn `LEARN`.
5. Chạy mục **11** để kiểm tra batch có độ dài bằng nhau và các vị trí pad không tham gia loss.
6. Chạy mục **12** để xem scorer nhận lời gọi đúng và từ chối các lỗi được dựng sẵn.

Chạy các mục này không cập nhật trọng số model. Việc encode dữ liệu và tính mask chỉ cần tokenizer.

## Quy tắc độ dài và padding

Độ dài được đo sau khi render đầy đủ hội thoại. `encode_sft()` mặc định giới hạn 1024 token và ném `SequenceTooLongError` khi mẫu vượt giới hạn. Caller phải thống kê và loại nguyên mẫu, không cắt schema, JSON hay câu trả lời.

`pad_batch()` pad bên phải tới độ dài mẫu dài nhất trong batch. Mask theo vị trí padding, không thay mọi label có cùng ID với pad token. Hàm trả các danh sách Python; vòng lặp train sau này sẽ chuyển chúng thành tensor.

## Hợp đồng parser và scorer hiện tại

`src/tool_calls.py` định nghĩa scorer cấu trúc ban đầu của dự án:

- Lời gọi nằm trong các khối `<tool_call>…</tool_call>`, chỉ cho phép whitespace giữa các khối.
- Mỗi JSON object có đúng hai key `name` và `arguments`; name là chuỗi không rỗng, arguments là object.
- Chấp nhận một token kết thúc `<|im_end|>` hoặc `<|endoftext|>` ở cuối output.
- JSON hỏng, tag thiếu, key trùng, NaN/Infinity và văn bản thừa ngoài các khối làm lời gọi không hợp lệ.
- Không có marker lời gọi: parser trả danh sách rỗng. Khi đáp án yêu cầu gọi tool, trường hợp này bị chấm sai.
- Parallel được so như multiset: bỏ qua thứ tự nhưng giữ số lần xuất hiện và tổng số lời gọi.
- Key của object không phụ thuộc thứ tự; array giữ thứ tự.
- Số JSON tương đương như `1` và `1.0` khớp nhau. Boolean khác số, chuỗi khác số, `null` khác chuỗi `"null"`.
- Parser giữ số thập phân dưới dạng `Decimal` để so sánh cấu trúc; `format_tool_calls()` nhận các object JSON đầu vào thông thường, như các ví dụ viết tay.

Scorer này chưa xử lý toàn bộ JSON Schema, tương đương ngữ nghĩa của từng API hoặc evaluator BFCL. Các metric và rubric nghiên cứu sẽ được cụ thể hóa và khóa ở những phase sau.

## Kiểm tra và mã nguồn

Từ thư mục gốc repo:

```powershell
python -m unittest discover -s tests -v
```

Các test thuần Python chạy local. Integration test dùng tokenizer SmolLM2 đã cache và một Llama ngẫu nhiên nhỏ khi có các thư viện cần thiết. Notebook chạy cùng bộ test trên Colab; không tải trọng số mới để kiểm tra mask.

- `src/chat_format.py`: system prompt và prompt cho playground.
- `src/sft_data.py`: template, labels, kiểm tra độ dài và padding.
- `src/sft_examples.py`: các ví dụ viết tay dùng quan sát.
- `src/tool_calls.py`: serialize, parse và so sánh cấu trúc lời gọi.
- `tests/test_sft_format.py`: kiểm tra format, padding, parser và scorer.
- `tests/test_sft_tokenizer.py`: kiểm tra vùng assistant và EOS bằng tokenizer thật.

Tham khảo API: [Transformers — apply_chat_template và assistant token mask](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.apply_chat_template).
