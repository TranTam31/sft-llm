# SFT Capability Tracking

Dự án thực nghiệm nhằm tìm hiểu **supervised fine-tuning (SFT) thay đổi một base language model như thế nào** khi học hội thoại và tool calling. Điểm xuất phát là `HuggingFaceTB/SmolLM2-360M`; quá trình học được quan sát qua hành vi đầu ra, các phép đo năng lực và thay đổi bên trong model.

Mục tiêu trước mắt là học và hiểu từng bước của pipeline: chuẩn bị dữ liệu, huấn luyện, đánh giá và phân tích kết quả. Quy mô model nhỏ giúp chạy các thí nghiệm có kiểm soát với tài nguyên giới hạn.

## Các câu hỏi chính

- SFT dạy model format trả lời, năng lực thực hiện yêu cầu, hay cả hai?
- Các kỹ năng tool calling xuất hiện như thế nào: chọn tool, điền tham số, hỏi lại và biết khi nào không gọi?
- Model có sử dụng được schema của tool chưa gặp trong dữ liệu SFT không?
- Việc học nhiệm vụ mới ảnh hưởng thế nào đến năng lực đã có?
- Full fine-tune và LoRA khác nhau ra sao về khả năng học, mức quên và thay đổi trọng số?

## Thiết kế thực nghiệm

Kế hoạch sử dụng dữ liệu hội thoại và function calling, so sánh cách học tuần tự với trộn dữ liệu, cùng full fine-tune và LoRA. Các checkpoint sẽ được đánh giá bằng một giao thức cố định để theo dõi quá trình học.

Dữ liệu chính dự kiến gồm SmolTalk, xLAM function-calling 60k và Glaive. Đánh giá tập trung vào tính đúng của lời gọi tool, khả năng khái quát, chất lượng hội thoại và mức giữ lại năng lực chung. Chi tiết câu hỏi nghiên cứu, giả thuyết, metric và điều kiện so sánh nằm trong [PLAN.md](PLAN.md).

## Trạng thái triển khai

Hiện tại dự án có **playground của Phase 0** để quan sát base model trước SFT, cùng các công cụ chuẩn bị format, loss mask, padding và chấm cấu trúc lời gọi tool. Playground đã được chạy trên Colab CPU; quá trình xây bộ dữ liệu, train SFT và đánh giá nghiên cứu đang nằm trong lộ trình.

Các phase triển khai theo kế hoạch:

1. **Phase 0 — Hạ tầng:** công cụ khám phá base, định dạng hội thoại và nền tảng train/eval.
2. **Phase 1 — Dữ liệu và eval:** chuyển đổi dữ liệu, chia tập và xây các phép chấm.
3. **Phase 2 — Pilot:** kiểm tra pipeline, scorer, tài nguyên và chốt giao thức.
4. **Phase 3 — Thí nghiệm chính:** chạy baseline và các cấu hình SFT đã chọn.
5. **Phase 4 — Mở rộng:** các thí nghiệm bổ sung khi pipeline chính đã ổn định.
6. **Phase 5 — Phân tích và báo cáo:** tổng hợp đường cong năng lực, mức quên và thay đổi trọng số.

Phase là giai đoạn triển khai. Các mã E0–E6 trong kế hoạch là những thiết lập thí nghiệm, có thể được thực hiện trong các phase tương ứng.

## Mã nguồn và môi trường

- `src/`: mã Python dùng chung cho local và Colab.
- `notebooks/`: notebook chạy và quan sát kết quả theo từng cell.
- `scripts/`: công cụ tạo notebook từ mã nguồn trong repo.
- `tests/`: kiểm tra format, phép đo và pipeline inference.
- `docs/`: hướng dẫn sử dụng cho từng phase.

Playground hiện sử dụng PyTorch và Hugging Face Transformers. Colab là môi trường chính để chạy model; local dùng để phát triển, kiểm tra code và thử suy luận khi đủ bộ nhớ. Notebook nhúng mã nguồn để có thể chia sẻ bằng một file `.ipynb`.

## Tài liệu

- [PLAN.md](PLAN.md): thiết kế nghiên cứu và lộ trình triển khai.
- [REFERENCES.md](REFERENCES.md): paper tham khảo, điểm mạnh và nội dung nên học từ từng bài.
- [EXPERIMENTS.md](EXPERIMENTS.md): nhật ký triển khai, thiết lập và kết quả đã chạy.
- [Hướng dẫn Phase 0](docs/phase-0-base-playground.md): chạy playground trên Colab hoặc local và đọc các phép đo.
- [Format dữ liệu và loss mask](docs/phase-0-sft-format.md): xem token được học, padding và hợp đồng parser/scorer.
- [Notebook Phase 0](notebooks/00_base_playground.ipynb): quan sát base model trước SFT.
