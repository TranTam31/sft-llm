# Báo cáo bộ eval

Sinh tự động bởi `python -m src.eval.build_sets`. Đừng sửa tay; chạy lại để cập nhật.

Seed 0. Mỗi nhóm có 100 mẫu test và 33 mẫu dev (PLAN 3.2); bộ C có 150 mẫu test. Không có prompt nào lặp lại giữa các bộ: mẫu trùng exact hoặc gần trùng (MinHash, cùng tham số với bước tách) với mẫu đã chọn thì bị bỏ qua.

## Số mẫu

| Bộ | Nhóm | Có sẵn | test | dev | Bỏ vì trùng |
|---|---|---|---|---|---|
| t_seen | missing_info | 600 | 100 | 33 | exact: 355, near: 111 |
| t_seen | multi_turn | 600 | 100 | 33 | near: 105, exact: 194 |
| t_seen | no_tool | – | 100 | 33 | – |
| t_seen | parallel | 600 | 100 | 33 | – |
| t_seen | select | 600 | 100 | 33 | – |
| t_seen | single | 600 | 100 | 33 | – |
| t_unseen | missing_info | 1210 | 100 | 33 | exact: 958, near: 175 |
| t_unseen | multi_turn | 5603 | 100 | 33 | near: 878, exact: 1335 |
| t_unseen | no_tool | – | 100 | 33 | – |
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

Cụm của các mẫu T-unseen còn lại (dev + test, một mẫu tính cho mọi cụm của nó): cụm 1: 1, cụm 2: 13, cụm 4: 85, cụm 7: 27, cụm 14: 108, cụm 16: 4, cụm 22: 1, cụm 23: 70, cụm 26: 1, cụm 28: 2, cụm 29: 264, cụm 33: 4, cụm 34: 4, cụm 35: 25, cụm 36: 102, cụm 37: 1, cụm 38: 4, cụm 42: 2, cụm 46: 67, cụm 47: 3, cụm 48: 1, cụm 49: 5.

Tool được gọi nhiều nhất trong T-unseen test: `validate_email` (19), `emailvalidation` (12), `is_valid_palindrome` (12), `is_power` (10), `is_valid_email` (10), `draw_cards` (9), `calculate_standard_deviation` (9), `is_anagram_phrase` (9), `is_rotation` (9), `runner_up` (9), `find_majority_element` (8), `email_verifier` (8), `bacterial_growth` (7), `find_next_greater_element` (7), `chi_square_independence_test` (7).

## Cụm của tool được gọi

T-unseen đặt trần 40% mẫu mỗi cụm cho từng nhóm, tính riêng test và dev: pool unseen phần lớn là một cụm tính toán (cụm 29). Mẫu đã chọn được giữ nếu còn trong trần, phần thiếu rút thêm từ pool sau mọi bước khác, nên nháp và các bộ khác không đổi. Một mẫu tính cho mọi cụm của các tool nó gọi (tool nhiễu không tính); dev + test.

| Bộ | Nhóm | Ba cụm nhiều nhất (số mẫu) | Cụm lớn nhất / số mẫu |
|---|---|---|---|
| t_seen | parallel | cụm 49: 28, cụm 37: 20, cụm 26: 17 | 21% |
| t_seen | select | cụm 37: 15, cụm 38: 13, cụm 49: 12 | 11% |
| t_seen | single | cụm 28: 12, cụm 47: 12, cụm 38: 11 | 9% |
| t_unseen | parallel | cụm 29: 53, cụm 14: 34, cụm 23: 28 | 40% |
| t_unseen | select | cụm 29: 53, cụm 14: 23, cụm 36: 22 | 40% |
| t_unseen | single | cụm 29: 53, cụm 36: 31, cụm 14: 20 | 40% |

| Nhóm | Mẫu cũ giữ lại | Rút thêm | Có sẵn trong pool theo cụm (ba cụm lớn nhất) |
|---|---|---|---|
| parallel | 105 | 28 | cụm 29: 1635, cụm 14: 383, cụm 23: 277 (pool có 2570 mẫu) |
| select | 98 | 35 | cụm 29: 658, cụm 36: 124, cụm 14: 115 (pool có 1047 mẫu) |
| single | 91 | 42 | cụm 29: 1028, cụm 36: 209, cụm 14: 132 (pool có 1567 mẫu) |

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

No-tool (PLAN 4.3): prompt chat giữ riêng mà lượt gán nhãn giữ lại (`no_tool_prompts`), cùng câu hỏi xLAM cần tool. `plain` ghép tool ở các cụm xa chủ đề nhất; `related` ghép tool cùng cụm với prompt (có vẻ liên quan nhưng không cần dùng); `no_fitting_tool` là câu hỏi cần tool nhưng chỉ có tool ở cụm khác. Lượt gán nhãn kiểm từng nháp rằng không tool nào giúp được (`no_tool`). Hai phía có cùng số mẫu mỗi loại; test và dev giữ tỉ lệ đó. Đáp án đúng là không gọi tool.

| Bộ | Có nhãn | Dùng | plain | related | no_fitting_tool |
|---|---|---|---|---|---|
| t_seen | 207 | 133 | 89 | 22 | 22 |
| t_unseen | 195 | 133 | 89 | 22 | 22 |

Nhãn không khớp nháp nào: 0.

## Nháp chờ gán nhãn

Chưa phải bộ eval: cần một lượt gán nhãn rồi mới chia dev/test (PLAN 3.2).

| File | Số mẫu | Nháp |
|---|---|---|
| `drafts/t_seen_missing_info.jsonl` | 134 | `missing_param_candidates` |
| `drafts/t_seen_missing_info_rewrite.jsonl` | 160 | `removable` (viết lại câu hỏi) |
| `drafts/t_seen_multi_turn.jsonl` | 301 | `required_candidates` |
| `drafts/t_seen_no_tool.jsonl` | 288 | `kind` (rà: không tool nào giúp được) |
| `drafts/t_unseen_missing_info.jsonl` | 77 | `missing_param_candidates` |
| `drafts/t_unseen_missing_info_rewrite.jsonl` | 100 | `removable` (viết lại câu hỏi) |
| `drafts/t_unseen_multi_turn.jsonl` | 529 | `required_candidates` |
| `drafts/t_unseen_no_tool.jsonl` | 288 | `kind` (rà: không tool nào giúp được) |

## Hash

| File | SHA-256 |
|---|---|
| `dev/c.jsonl` | `8b29f970f9a5c05802c85c769c7df9585f1fdef9e0651e6726e50db6bf1751cf` |
| `dev/t_rename_nonsense.jsonl` | `7d2f6263ca7ec1b19d6d253ac5b3bc365d72686fb6da6a136c5cc565a46a3d5c` |
| `dev/t_seen.jsonl` | `41fd2763f7f9a34189788a0e590df04284c81373b7cf564f6c13fcbf5d73b7e2` |
| `dev/t_unseen.jsonl` | `6cb5e3f086da57bc104164979ba49d2190b19915d2c36ce9d15b753abdcc9309` |
| `drafts/t_seen_missing_info.jsonl` | `ec2ed01a5b96cdfe7f7ddaf614c3d3c7505dd43821163909dda80663e44a6c1e` |
| `drafts/t_seen_missing_info_rewrite.jsonl` | `d8ee12d696d73b899f30194fe244a424fe222f8f0b8cc6d38fc703a5f5744d39` |
| `drafts/t_seen_multi_turn.jsonl` | `379c2b5f6a4c8771530d98423590a5375280ac684e7d289496dfec64111bb47b` |
| `drafts/t_seen_no_tool.jsonl` | `71cbee680ce06a61d3482a3fec48f89aaa1591245241807335f3b9b156f70916` |
| `drafts/t_unseen_missing_info.jsonl` | `e0a78bf169efdcdbb070def2ee0749ab89597f6d2a510d1afcb50b7587f08d42` |
| `drafts/t_unseen_missing_info_rewrite.jsonl` | `9092289f10b91f64262b92239859abcc7a5fe0f1a8bbc8ce1b6653ca1a0d63da` |
| `drafts/t_unseen_multi_turn.jsonl` | `54352350fb926746ae01c520166c565b0e26f2cf33625f54111da96879ade8c3` |
| `drafts/t_unseen_no_tool.jsonl` | `a117c68afc5ec67eb7d3d31eccdb590f5587fbb0919ebdc2752eac10c2db01ae` |
| `test/c.jsonl` | `82f99eb714d59864c789835aa046f6048d7d5cd603a1280be502d57670651be3` |
| `test/t_rename_nonsense.jsonl` | `3dc7e341e92e55b26ee7648fb1781d5f4e66adc8f0573becc86b96e0881b0903` |
| `test/t_seen.jsonl` | `60516ada5b16f65708083af416929dff29eedf680ae195a6e034bb66a0b9e820` |
| `test/t_unseen.jsonl` | `6e4823fe89181e1af0d05325bf4bca22f1605a701b6ed98c8ac3dfd6bd559630` |
