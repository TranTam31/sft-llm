# Báo cáo tập train

Sinh tự động bởi `python -m src.data.train_sets`. Đừng sửa tay; chạy lại để cập nhật.

Seed 0. D_chat 10000 mẫu, D_tool 10000 mẫu (PLAN 5.1; số chính thức chốt sau pilot, pilot lấy một phần đầu của mỗi file vì thứ tự đã xáo). Không prompt nào lặp lại prompt train khác hay 2654 prompt của dev, test và P: lượt user đầu trùng exact hoặc gần trùng (MinHash, cùng tham số với bước tách) thì bị bỏ qua.

## Tỉ lệ D_tool

Đã chốt (2026-10-24): 50% gọi tool xLAM (single / select / parallel chia gần đều), 20% Multi-turn, 10% Missing-info, 20% No-tool. Missing-info chỉ lấy từ Glaive, giữ hội thoại có lời gọi sau cần một tham số bắt buộc mà lượt user đầu chưa nêu; pool thiếu thì phần thiếu chuyển sang ba nhóm gọi tool. Khác bộ eval (nửa Glaive, nửa câu hỏi xLAM viết lại): nguồn Missing-info của train và eval lệch nhau.

| Nhóm | Pool | Sau lọc | Cần | Lấy | Bỏ vì trùng |
|---|---|---|---|---|---|
| chat | 15885 | – | 10000 | 10000 | exact: 1529, near: 121 |
| single | 5724 | – | 1805 | 1805 | exact: 4, near: 10 |
| select | 9717 | – | 1804 | 1804 | exact: 11, near: 29 |
| parallel | 20361 | – | 1697 | 1697 | near: 6, exact: 1 |
| multi_turn | 20408 | – | 2000 | 2000 | near: 1656, exact: 1875 |
| missing_info | 5267 | 4661 | 1000 | 694 | exact: 2528, near: 1439 |

## No-tool (PLAN 4.3)

Nháp dựng như bộ eval: `plain` là prompt chat (ngoài D_chat) kèm tool ở các cụm xa chủ đề nhất; `related` kèm tool cùng cụm gần prompt nhất; `no_fitting_tool` là câu hỏi xLAM single (ngoài D_tool) chỉ kèm tool ở cụm xa, trả lời bằng câu cố định. Chỉ dùng tool seen có trong pool train. Câu trả lời của `plain` và `related` là lượt assistant đầu của SmolTalk. Lượt gán nhãn giữ nháp là yêu cầu tự đủ ngữ cảnh, trả lời được bằng kiến thức chung (hoặc với 3b: thật sự cần dữ liệu hay hành động bên ngoài) và không tool nào giúp được.

Nháp: 2750; có nhãn: 2467; đạt: 2183; nhãn không khớp nháp nào: 0.

| Loại | Cần | Lấy | Bỏ vì quá dài |
|---|---|---|---|
| plain | 1334 | 1334 | 137 |
| related | 333 | 333 | 42 |
| no_fitting_tool | 333 | 333 | 5 |

| Nháp | Lấy | Cần | Bỏ vì trùng |
|---|---|---|---|
| related | 416 | 416 | near: 21, exact: 1 |
| plain | 1668 | 1668 | exact: 1343, near: 99 |
| no_fitting_tool | 666 | 666 | near: 16, exact: 2 |

## Kích thước

| File | Nhóm | Mẫu | Độ dài trung vị | Token assistant |
|---|---|---|---|---|
| d_chat | chat | 10000 | 389 | 2303786 |
| d_tool | missing_info | 694 | 488 | 97905 |
| d_tool | multi_turn | 2000 | 478 | 264958 |
| d_tool | no_tool | 2000 | 596 | 279505 |
| d_tool | parallel | 1697 | 508 | 160890 |
| d_tool | select | 1804 | 596 | 83290 |
| d_tool | single | 1805 | 303 | 89116 |

## Ghi chú

- Missing-info: Glaive chỉ có 694 câu mở đầu khác nhau thỏa luật lọc (pool 5267, phần lớn lặp lại), nên 306 mẫu thiếu chuyển sang ba nhóm gọi tool; nhóm gọi tool chiếm khoảng 53% D_tool.
- Token assistant: D_chat khoảng 2.30 triệu, D_tool khoảng 0.97 triệu. Khi trộn (E2), khoảng 70% token có loss là chat.
- Rà ngẫu nhiên 200 mẫu No-tool của D_tool (PLAN 4.3, subagent độc lập, 2026-10-24): 0/200 sai nhãn (không gọi tool là đúng ở mọi mẫu, cả 28 mẫu no_fitting_tool đều cần dữ liệu ngoài). 15/200 có câu trả lời SmolTalk kém: 9 sai hẳn (toán, code, kiến thức), 4 vi phạm ràng buộc của smol-summarize (quá 3 câu, đại từ ngôi 2/3), 2 ở ranh giới. Đây là chất lượng của nguồn SmolTalk, cũng có trong D_chat vốn không lọc, không phải lỗi bộ lọc No-tool.

## Hash

| File | SHA-256 |
|---|---|
| `d_chat.jsonl` | `58674077f13a7a743f17ad5e7bb7380cc02e0831c95979868ee8bdc21e86087d` |
| `d_tool.jsonl` | `5467d8789bbe5c2c44634561e3e13be013be2ed537c10f30c633fe3dce0dea69` |
| `drafts/no_tool.jsonl` | `456716a9ed1aa7b96e78ed16ac2347e26ab2c5b6b8354664ce6cb03dc627ba3a` |
