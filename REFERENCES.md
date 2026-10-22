# Tài liệu tham khảo — Học và hiểu SFT cho hội thoại và tool calling

Tài liệu này dành cho người mới triển khai SFT, với mục tiêu hiểu dữ liệu, quá trình huấn luyện và cách đo hành vi của model. Các bài được sắp theo thứ tự ưu tiên đọc cho dự án dùng SmolLM2-360M: hiểu model gốc → hiểu vai trò của SFT → hiểu bài kiểm tra và dữ liệu tool calling → tìm hiểu mức quên và phân tích sâu hơn.

Mỗi mục tách riêng nội dung nghiên cứu, điểm mạnh và những điều có thể học để áp dụng. Các gợi ý áp dụng là hướng đọc của tài liệu này, không phải khẳng định rằng paper sử dụng đúng thiết kế của dự án. Tài liệu không bao gồm phần đánh giá hoặc đề xuất sửa `PLAN.md`.

## Cách dùng tài liệu

- **Đọc trước khi làm:** bài 1–3 để hiểu model, SFT và cách đánh giá tool calling.
- **Đọc khi làm dữ liệu:** bài 4–5 để hiểu nguồn xLAM và khả năng sử dụng schema.
- **Đọc khi đã có run đầu tiên:** bài 6–7 để hiểu học, quên và tổ chức thí nghiệm.
- **Đọc sâu sau khi nắm pipeline:** bài 8–9 để theo dõi động lực học và tìm hiểu quy trình phát triển model lớn.

Không cần đọc toàn bộ phụ lục ngay. Với mỗi bài, nên bắt đầu từ abstract, hình tổng quan, phần thiết lập thí nghiệm và kết luận; sau đó đọc các mục được chỉ dẫn bên dưới.

Một số thuật ngữ thường gặp:

- **SFT — supervised fine-tuning:** huấn luyện model trên các ví dụ đầu vào và câu trả lời mong muốn.
- **Ablation:** thay hoặc bỏ một thành phần để đo ảnh hưởng của thành phần đó.
- **ID / OOD:** dữ liệu trong / ngoài phân phối của bài toán huấn luyện; ý nghĩa cụ thể phụ thuộc cách từng nghiên cứu định nghĩa.
- **Checkpoint:** trạng thái model được lưu tại một thời điểm trong quá trình train.
- **LoRA:** fine-tune thông qua các ma trận cập nhật có rank thấp, thay vì cập nhật toàn bộ trọng số.
- **SVD:** phân rã ma trận thành các hướng và độ lớn tương ứng, thường dùng để phân tích thay đổi trọng số.

## 1. SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model

**Nhóm tác giả:** Hugging Face và cộng sự. **Năm:** 2025.

**Nguồn:** [Paper và các phiên bản](https://arxiv.org/abs/2502.02737) · [Bản HTML được dẫn trong tài liệu này, v1](https://arxiv.org/html/2502.02737v1).

### Bài báo làm gì, triển khai những gì?

Mô tả quá trình xây dựng SmolLM2, trọng tâm là model 1.7B. Nghiên cứu trình bày lựa chọn dữ liệu, pretraining nhiều giai đoạn và phát triển phiên bản instruction-tuned. Các dữ liệu được giới thiệu gồm FineMath, Stack-Edu và SmolTalk. Paper cũng có phần riêng cho các bản 135M và 360M.

### Điểm mạnh

Đây là nguồn gốc trực tiếp của họ model đang sử dụng. Cách trình bày nối lựa chọn dữ liệu với thiết lập train và kết quả đánh giá, giúp người đọc hiểu một model được xây dựng như thế nào.

### Có thể học gì?

Hiểu vai trò của SmolTalk, phân biệt base với Instruct và tra thiết lập dành cho model nhỏ. Khi đọc, tách thông tin của 1.7B khỏi 360M.

**Đọc ưu tiên:** mục 5 — Post-training; mục 6 — SmolLM2 135M and 360M. Các số mục này theo bản v1.

## 2. LIMA: Less Is More for Alignment

**Tác giả:** Chunting Zhou và cộng sự. **Công bố:** NeurIPS 2023.

**Nguồn:** [Paper trên arXiv](https://arxiv.org/abs/2305.11206) · [Bản hội nghị](https://papers.nips.cc/paper/2023/file/ac662d74829e4407ce1d126477f4a03a-Paper-Conference.pdf).

### Bài báo làm gì, triển khai những gì?

Fine-tune LLaMA 65B bằng 1.000 ví dụ được tuyển chọn, không dùng RLHF. Tác giả đánh giá câu trả lời bằng con người, khảo sát ảnh hưởng của chất lượng và độ đa dạng dữ liệu, đồng thời nghiên cứu khả năng chuyển sang hội thoại nhiều lượt.

Paper đề xuất *Superficial Alignment Hypothesis*: phần lớn kiến thức và năng lực đến từ pretraining; alignment chủ yếu dạy cách thể hiện chúng theo format và phong cách mong muốn.

### Điểm mạnh

Đặt một câu hỏi rõ về vai trò của SFT và dùng thiết lập tương đối đơn giản để khảo sát. Phân tích dữ liệu giúp người mới thấy vì sao số lượng mẫu chưa đủ để mô tả chất lượng tập train.

### Có thể học gì?

Phân biệt học cách trả lời với học thêm khả năng; quan sát chất lượng, độ đa dạng và số lượng ví dụ. Kết quả 1.000 mẫu trên 65B là kết quả trong thiết lập của bài, không phải công thức chung cho model 360M.

**Đọc ưu tiên:** giả thuyết ở đầu bài, cách tuyển chọn dữ liệu, đánh giá và ablation.

## 3. The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models

**Tác giả:** Shishir G. Patil và cộng sự. **Công bố:** ICML 2025.

**Nguồn:** [Paper và liên kết phần mềm](https://proceedings.mlr.press/v267/patil25a.html) · [Trang BFCL chính thức](https://gorilla.cs.berkeley.edu/leaderboard).

### Bài báo làm gì, triển khai những gì?

Xây benchmark đánh giá khả năng gọi function qua nhiều tình huống: lựa chọn hàm, gọi song song, tương tác tuần tự và biết không gọi khi không phù hợp. Bài trình bày phương pháp đánh giá dựa trên AST để so cấu trúc lời gọi thay vì chỉ so chuỗi văn bản, cùng các bài kiểm tra tương tác có trạng thái.

### Điểm mạnh

Phân rã tool calling thành nhiều tình huống cụ thể, có dữ liệu và công cụ đánh giá. Đây là tài liệu về cách đo năng lực, không chỉ về cách train model đạt điểm cao.

### Có thể học gì?

Hiểu các nhóm bài kiểm tra, cách chấm tên hàm và tham số, cách kiểm tra model từ chối lời gọi không phù hợp. Khi sử dụng code hoặc dữ liệu, ghi phiên bản BFCL vì benchmark tiếp tục phát triển.

**Đọc ưu tiên:** cấu trúc benchmark, AST evaluation và các ví dụ đúng/sai.

## 4. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets

**Tác giả:** Zuxin Liu và cộng sự, Salesforce AI Research. **Năm:** 2024.

**Nguồn:** [Paper](https://arxiv.org/abs/2406.18518) · [Dataset xLAM 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) · [Repo xLAM](https://github.com/SalesforceAIResearch/xLAM).

### Bài báo làm gì, triển khai những gì?

Xây pipeline sinh dữ liệu function calling từ 3.673 API thực thi được thuộc 21 nhóm. Dữ liệu được xác minh qua ba tầng: kiểm tra format, thực thi function và kiểm tra ngữ nghĩa. Nhóm công bố 60.000 mẫu và đánh giá các model được huấn luyện bằng dữ liệu đó trên BFCL.

### Điểm mạnh

Gắn việc tạo dữ liệu với kiểm chứng lời gọi thực tế. Người đọc có thể theo dõi quá trình từ API và schema đến câu hỏi, đáp án và kết quả train.

### Có thể học gì?

Hiểu nguồn dữ liệu xLAM đang dùng, lý do JSON hợp lệ chưa bảo đảm lời gọi đúng và cách tổ chức kiểm tra dữ liệu theo nhiều tầng. Phù hợp để đọc trước khi viết bộ chuyển đổi dữ liệu.

**Đọc ưu tiên:** pipeline sinh dữ liệu, ba tầng xác minh, ví dụ mẫu và đánh giá model.

## 5. Hammer: Robust Function-Calling for On-Device Language Models via Function Masking

**Tác giả:** Qiqiang Lin và cộng sự. **Thời điểm:** arXiv 2024; công bố tại ICLR 2025 với tiêu đề “Robust Function-Calling for On-Device Language Models via Function Masking”.

**Nguồn:** [Paper trên arXiv](https://arxiv.org/abs/2410.04587) · [Bản ICLR 2025](https://openreview.net/pdf?id=yVQcr4qjD6) · [HTML bản arXiv v2](https://arxiv.org/html/2410.04587v2).

### Bài báo làm gì, triển khai những gì?

Nghiên cứu việc model bị ảnh hưởng bởi tên function và tên tham số. Tác giả dùng *function masking* để giảm sự phụ thuộc vào tên, bổ sung dữ liệu nhận biết function không phù hợp và đánh giá trên nhiều benchmark. Các ablation khảo sát base model, tỉ lệ masking và tỉ lệ dữ liệu bổ sung.

### Điểm mạnh

Biến vấn đề “model có quá dựa vào tên không?” thành một can thiệp vào dữ liệu có thể kiểm tra. Nghiên cứu đồng thời xem xét khả năng gọi đúng và biết khi nào cần tránh gọi.

### Có thể học gì?

Hiểu sự khác nhau giữa tên, description và schema; cách khảo sát ảnh hưởng của đổi tên; cách tạo ví dụ không có function phù hợp. Liên quan trực tiếp tới bài kiểm tra rename và No-tool.

**Đọc ưu tiên:** phân tích tên ở mục 3, phương pháp ở mục 4 và ablation ở mục 5 của bản arXiv v2.

## 6. LoRA Learns Less and Forgets Less

**Tác giả:** Dan Biderman và cộng sự. **Công bố:** TMLR 2024.

**Nguồn:** [Paper](https://arxiv.org/abs/2405.09673) · [Bản HTML v2](https://arxiv.org/html/2405.09673v2).

### Bài báo làm gì, triển khai những gì?

So sánh LoRA và full fine-tune trên toán và lập trình, trong cả instruction fine-tuning và continued pretraining. Nghiên cứu đo năng lực ở miền đang học, suy giảm ngoài miền, ảnh hưởng của thời lượng train, rank, learning rate và phổ của thay đổi trọng số.

### Điểm mạnh

Xem xét đồng thời khả năng học và giữ năng lực cũ. Các đường đánh đổi cho thấy kết quả phụ thuộc miền và cấu hình; LoRA học ít hơn và quên ít hơn chưa tự nó chứng minh một phương pháp tốt hơn ở cùng mức năng lực.

### Có thể học gì?

Cách chọn benchmark đo quên, vẽ năng lực mới so với năng lực cũ và khảo sát siêu tham số riêng cho từng phương pháp. Phần SVD giúp hiểu cách phân tích cập nhật trọng số.

**Đọc ưu tiên:** mục 3.2–3.3, mục 4.3 về learning–forgetting tradeoff và phụ lục B về learning rate.

## 7. Tülu 3: Pushing Frontiers in Open Language Model Post-Training

**Tác giả:** Nathan Lambert và cộng sự, Ai2. **Thời điểm:** arXiv 2024; COLM 2025.

**Nguồn:** [Paper](https://arxiv.org/abs/2411.15124) · [HTML v5](https://arxiv.org/html/2411.15124v5) · [Trang dự án và các tài nguyên](https://allenai.org/tulu).

### Bài báo làm gì, triển khai những gì?

Trình bày quy trình post-training gồm SFT, DPO và RLVR. Nhóm công bố dữ liệu, code và công thức huấn luyện; mô tả tuyển chọn, khử nhiễm dữ liệu, đánh giá development và unseen, cùng các thí nghiệm về dữ liệu và thiết lập train.

### Điểm mạnh

Kết nối tổ chức dữ liệu, huấn luyện và đánh giá thành một quy trình có tài nguyên để tái lập. Paper cũng ghi nhận những vấn đề triển khai cụ thể, như tổng hợp loss khi gradient accumulation.

### Có thể học gì?

Cách tổ chức config, phân biệt tập dùng chọn thiết kế với tập đánh giá chưa xem, và liên hệ thay đổi dữ liệu với kỹ năng. Khi học riêng SFT, tập trung vào kết quả của giai đoạn SFT; điểm cuối sau DPO/RLVR phản ánh thêm các bước khác.

**Đọc ưu tiên:** mục 3 — Data, mục 4 — SFT và mục 7 — Evaluation Framework.

## 8. RL Fine-Tuning Heals OOD Forgetting in SFT

**Tác giả:** Hangzhan Jin và cộng sự. **Thời điểm:** arXiv 2025; bản v3 cập nhật tháng 5/2026.

**Nguồn:** [Paper và lịch sử phiên bản](https://arxiv.org/abs/2509.12235) · [HTML v3](https://arxiv.org/html/2509.12235v3).

### Bài báo làm gì, triển khai những gì?

Theo dõi năng lực ID và OOD qua các checkpoint SFT rồi RL. Nghiên cứu kết hợp bài toán có quy tắc kiểm soát được với đánh giá reasoning rộng hơn. Trong thiết lập khảo sát, OOD có thể đạt đỉnh sớm rồi giảm trong SFT. Tác giả dùng phân tích phổ và can thiệp trọng số để tìm hiểu quá trình quên và phục hồi.

### Điểm mạnh

Quan sát diễn biến trong quá trình train, thay vì chỉ so trước và sau. Kết hợp thay đổi hành vi với thí nghiệm trên trọng số, đồng thời phân biệt bằng chứng từ bài toán kiểm soát với benchmark công khai.

### Có thể học gì?

Cách đọc đường cong qua checkpoint, phân biệt tiến bộ trên dữ liệu quen với khả năng khái quát, và hiểu vì sao loss giảm chưa đủ mô tả năng lực. Kết quả trên reasoning là nguồn tham khảo phương pháp, chưa xác nhận cùng hiện tượng ở tool calling.

**Đọc ưu tiên:** mục 3 — Evaluation and Analysis; đọc mục 4 — SVD Analysis sau khi nắm pipeline hành vi.

## 9. The Llama 3 Herd of Models

**Nhóm tác giả:** Meta và cộng sự. **Năm:** 2024.

**Nguồn:** [Paper](https://arxiv.org/abs/2407.21783).

### Bài báo làm gì, triển khai những gì?

Mô tả quá trình phát triển họ Llama 3, gồm dữ liệu, pretraining, post-training và đánh giá. Phần post-training trình bày giao thức hội thoại, reward model, rejection sampling, SFT và DPO, cùng tuyển chọn dữ liệu cho các khả năng như reasoning, coding và tool use.

### Điểm mạnh

Cho thấy SFT nằm ở đâu trong quy trình phát triển một model lớn và liên hệ với các bước tạo dữ liệu, chọn câu trả lời và tối ưu theo preference.

### Có thể học gì?

Hiểu bức tranh tổng thể sau khi đã tự chạy SFT. Phân biệt phương pháp mô tả trong báo cáo với tài nguyên thực sự được phát hành; kết quả cuối của model qua nhiều bước không thể quy toàn bộ cho riêng SFT.

**Đọc ưu tiên:** mục 4 — Post-Training; đặc biệt mục 4.1.3 — Supervised Finetuning và mục 4.3.5 — Tool Use.

## Ghi chú khi đọc và tái sử dụng

- Ghi lại model, dữ liệu, mục tiêu và cách chấm của từng thí nghiệm trước khi so với kết quả của mình.
- Phân biệt kết quả tác giả quan sát trong một thiết lập với giả thuyết áp dụng rộng hơn.
- Khi lấy code, dữ liệu hoặc benchmark, lưu revision và phiên bản cụ thể. Link arXiv không có hậu tố phiên bản có thể chuyển tới bản mới hơn.
- Với bài có nhiều giai đoạn post-training, tìm kết quả ngay sau SFT nếu đang muốn hiểu riêng tác động của SFT.
