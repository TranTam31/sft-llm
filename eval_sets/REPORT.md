# Báo cáo bộ eval

Sinh tự động bởi `python -m src.eval.build_sets`. Đừng sửa tay; chạy lại để cập nhật.

Seed 0. Mỗi nhóm có 100 mẫu test và 33 mẫu dev (PLAN 3.2); bộ C có 150 mẫu test. Không có prompt nào lặp lại giữa các bộ: mẫu trùng exact hoặc gần trùng (MinHash, cùng tham số với bước tách) với mẫu đã chọn thì bị bỏ qua.

## Số mẫu

| Bộ | Nhóm | Có sẵn | test | dev | Bỏ vì trùng |
|---|---|---|---|---|---|
| t_seen | missing_info | 600 | 100 | 33 | exact: 355, near: 111 |
| t_seen | multi_turn | 600 | 100 | 33 | near: 105, exact: 194 |
| t_seen | parallel | 600 | 100 | 33 | – |
| t_seen | select | 600 | 100 | 33 | – |
| t_seen | single | 600 | 100 | 33 | – |
| t_unseen | missing_info | 1210 | 100 | 33 | exact: 958, near: 175 |
| t_unseen | multi_turn | 5603 | 100 | 33 | near: 878, exact: 1335 |
| t_unseen | parallel | 2570 | 100 | 33 | – |
| t_unseen | select | 1047 | 100 | 33 | near: 3, exact: 1 |
| t_unseen | single | 1567 | 100 | 33 | near: 1 |
| t_rename_nonsense | parallel | – | 100 | 33 | – |
| t_rename_nonsense | select | – | 100 | 33 | – |
| t_rename_nonsense | single | – | 100 | 33 | – |
| c | chat | 1000 | 150 | 50 | – |

## Lọc T-unseen (PLAN 4.2 ý 6)

93 tool unseen có cosine ≥ 0.85 tới một tool seen. Mọi mẫu `unseen_eval` có cung cấp một trong các tool này bị bỏ:

| Nhóm | Mẫu bị bỏ |
|---|---|
| clarify_candidate | 8 |
| multi_turn | 249 |
| no_call | 86 |
| parallel | 127 |
| select | 39 |
| single | 61 |

Cụm của các mẫu T-unseen còn lại (dev + test, một mẫu tính cho mọi cụm của nó): cụm 1: 1, cụm 2: 13, cụm 4: 58, cụm 7: 23, cụm 14: 69, cụm 16: 1, cụm 23: 47, cụm 29: 359, cụm 34: 3, cụm 35: 25, cụm 36: 85, cụm 46: 67, cụm 47: 2, cụm 49: 3.

Tool được gọi nhiều nhất trong T-unseen test: `validate_email` (16), `is_power` (16), `draw_cards` (14), `find_majority_element` (12), `bacterial_growth` (11), `chi_square_independence_test` (11), `calculate_standard_deviation` (11), `is_valid_email` (9), `is_power_of_two` (9), `numerical_derivative` (8), `get_range` (8), `find_next_greater_element` (8), `binary_addition` (8), `sort_numbers` (8), `is_anagram_phrase` (8).

## T-rename mức tên vô nghĩa (PLAN 4.4)

Tool đổi thành `fn_xxx`, tham số cấp ngoài cùng thành `p1, p2…` theo thứ tự trong schema; description giữ nguyên. Tên dạng định danh (có `_`, chữ số hoặc camelCase) xuất hiện trong prompt cũng được thay.

Tham số trong đáp án mà schema không khai báo, nên giữ tên cũ: 0.

## Missing-info và Multi-turn (nhãn do session gán)

Nhãn nằm ở `eval_sets/labels/` (dataset trên Hub). Missing-info lấy cùng số mẫu Glaive ở hai phía, phần còn lại là câu hỏi xLAM viết lại cho thiếu một tham số bắt buộc, để seen và unseen có cùng tỉ lệ nguồn; test và dev giữ tỉ lệ đó. Multi-turn nhắm cùng một tỉ lệ ở hai phía: lấy số mẫu cần đọc kết quả tool bằng mức phía nghèo hơn có, phần còn lại là mẫu tự tính được (đáp án suy ra được từ câu hỏi, đánh dấu `meta.derivable` để tính điểm phụ trên phần còn lại); phía thiếu mẫu tự tính được thì bù bằng mẫu cần đọc kết quả tool. Đáp án đúng của Missing-info là không gọi tool.

| Bộ | Nhóm | Có nhãn | Dùng | Glaive | Viết lại | Tự tính được |
|---|---|---|---|---|---|---|
| t_seen | missing_info | 189 | 133 | 60 | 73 | – |
| t_seen | multi_turn | 249 | 133 | – | – | 88 |
| t_unseen | missing_info | 134 | 133 | 60 | 73 | – |
| t_unseen | multi_turn | 267 | 133 | – | – | 99 |

Nhãn không khớp nháp nào: 0.

## Nháp chờ gán nhãn

Chưa phải bộ eval: cần một lượt gán nhãn rồi mới chia dev/test (PLAN 3.2).

| File | Số mẫu | Nháp |
|---|---|---|
| `drafts/t_seen_missing_info.jsonl` | 134 | `missing_param_candidates` |
| `drafts/t_seen_missing_info_rewrite.jsonl` | 160 | `removable` (viết lại câu hỏi) |
| `drafts/t_seen_multi_turn.jsonl` | 301 | `required_candidates` |
| `drafts/t_unseen_missing_info.jsonl` | 77 | `missing_param_candidates` |
| `drafts/t_unseen_missing_info_rewrite.jsonl` | 100 | `removable` (viết lại câu hỏi) |
| `drafts/t_unseen_multi_turn.jsonl` | 529 | `required_candidates` |

## Hash

| File | SHA-256 |
|---|---|
| `dev/c.jsonl` | `8b29f970f9a5c05802c85c769c7df9585f1fdef9e0651e6726e50db6bf1751cf` |
| `dev/t_rename_nonsense.jsonl` | `7d2f6263ca7ec1b19d6d253ac5b3bc365d72686fb6da6a136c5cc565a46a3d5c` |
| `dev/t_seen.jsonl` | `9c30d2bcb94e86d7e259c5e6cd08e6f8620314597521692c9d1110e208b68770` |
| `dev/t_unseen.jsonl` | `f3564f488b54fa606949971a8429b47a4ea66e4458a7dc93fbf3dd2892183c90` |
| `drafts/t_seen_missing_info.jsonl` | `ec2ed01a5b96cdfe7f7ddaf614c3d3c7505dd43821163909dda80663e44a6c1e` |
| `drafts/t_seen_missing_info_rewrite.jsonl` | `d8ee12d696d73b899f30194fe244a424fe222f8f0b8cc6d38fc703a5f5744d39` |
| `drafts/t_seen_multi_turn.jsonl` | `379c2b5f6a4c8771530d98423590a5375280ac684e7d289496dfec64111bb47b` |
| `drafts/t_unseen_missing_info.jsonl` | `e0a78bf169efdcdbb070def2ee0749ab89597f6d2a510d1afcb50b7587f08d42` |
| `drafts/t_unseen_missing_info_rewrite.jsonl` | `9092289f10b91f64262b92239859abcc7a5fe0f1a8bbc8ce1b6653ca1a0d63da` |
| `drafts/t_unseen_multi_turn.jsonl` | `54352350fb926746ae01c520166c565b0e26f2cf33625f54111da96879ade8c3` |
| `test/c.jsonl` | `82f99eb714d59864c789835aa046f6048d7d5cd603a1280be502d57670651be3` |
| `test/t_rename_nonsense.jsonl` | `3dc7e341e92e55b26ee7648fb1781d5f4e66adc8f0573becc86b96e0881b0903` |
| `test/t_seen.jsonl` | `ca35145fc5b73fbf207b53b2b253205c82b243dc7697c6f82eb36d6cb2c73bd7` |
| `test/t_unseen.jsonl` | `594d095724574b4c30c9848a1963ebd2ba5815b475365884be538af9558fb8a4` |
