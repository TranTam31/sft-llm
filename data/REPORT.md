# Báo cáo dữ liệu

Sinh tự động bởi `python -m src.data.build`. Đừng sửa tay; chạy lại để cập nhật.

Seed 0, `max_seq_len` 1024 (PLAN 2.5). 

## Số mẫu theo nguồn

| Nguồn | Dòng đọc | Giữ | Quá dài | Tỉ lệ quá dài | Lỗi chuyển đổi | Nhóm | Độ dài trung vị / p95 |
|---|---|---|---|---|---|---|---|
| glaive | 112960 | 99673 | 12919 | 11.5% | bad_call: 155, call_tool_not_offered: 25, invalid_turns: 24, response_without_call: 164 | call_no_response: 371, chat_no_tools: 21715, clarify_candidate: 15483, multi_turn: 47124, no_call: 14980 | 463 / 895 |
| smoltalk-everyday-conversations | 2000 | 2000 | 0 | 0.0% | – | chat: 2000 | 289 / 345 |
| smoltalk-smol-magpie-ultra | 18442 | 2000 | 16440 | 89.2% ⚠ | non_english: 2 | chat: 2000 | 868 / 1010 |
| smoltalk-smol-constraints | 1505 | 1500 | 5 | 0.3% | – | chat: 1500 | 275 / 548 |
| smoltalk-smol-rewrite | 1500 | 1500 | 0 | 0.0% | – | chat: 1500 | 412 / 658 |
| smoltalk-smol-summarize | 1870 | 1500 | 370 | 19.8% ⚠ | – | chat: 1500 | 343 / 874 |
| smoltalk-explore-instruct-rewriting | 1002 | 1000 | 0 | 0.0% | non_english: 2 | chat: 1000 | 180 / 243 |
| smoltalk-openhermes-100k | 3211 | 3000 | 157 | 5.0% | non_english: 54 | chat: 3000 | 412 / 825 |
| smoltalk-metamathqa-50k | 1512 | 1500 | 9 | 0.6% | non_english: 3 | chat: 1500 | 315 / 555 |
| smoltalk-numina-cot-100k | 1701 | 1500 | 194 | 11.5% | non_english: 7 | chat: 1500 | 524 / 933 |
| smoltalk-self-oss-instruct | 1509 | 1500 | 9 | 0.6% | – | chat: 1500 | 406 / 709 |
| xlam | 60000 | 56604 | 3396 | 5.7% | – | parallel: 29954, select: 18627, single: 8023 | 515 / 892 |

⚠ đánh dấu nguồn mất hơn 15% số mẫu vì độ dài: xem lại giới hạn trước khi train (PLAN 2.5). Hiện có: smoltalk-smol-magpie-ultra, smoltalk-smol-summarize.

Câu dẫn trước lời gọi bị bỏ (Glaive): {'glaive': 8215}.

System prompt riêng được gộp vào lượt user đầu (SmolTalk): {'smoltalk-smol-rewrite': 1500, 'smoltalk-smol-summarize': 1500, 'smoltalk-explore-instruct-rewriting': 525, 'smoltalk-openhermes-100k': 1184}.

## Ghi chú

- Giữ `max_seq_len` 1024 dù smol-magpie-ultra mất khoảng 89% vì độ dài: dữ liệu tool có p95 dưới 1024. Phần magpie còn lại lệch về hội thoại ngắn, nên chỉ lấy 2000 mẫu để giới hạn ảnh hưởng của độ lệch này lên D_chat.
- System prompt riêng của SmolTalk (smol-rewrite, smol-summarize…) được gộp vào đầu lượt user đầu tiên; system prompt cố định của PLAN 2.3 giữ nguyên.

## Subset SmolTalk không dùng

- `apigen-80k`: function-calling data from the APIGen pipeline that also produced xLAM
- `systemchats-30k`: teaches following custom system prompts, which the fixed system prompt (PLAN 2.3) rules out
- `longalign`: long-context data, mostly over max_seq_len

## Revision và hash

- `Salesforce/xlam-function-calling-60k` @ `26d14ebfe18b1f7b524bd39b404b50af5dc97866`
- `glaiveai/glaive-function-calling-v2` @ `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac`
- `HuggingFaceTB/smoltalk` @ `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`

| File | SHA-256 |
|---|---|
| `glaive.jsonl` | `b91f2f58aba56684274b880be6cf790f646901946af12057f750b0f777e8d60a` |
| `smoltalk-everyday-conversations.jsonl` | `de5c0b726da1015ce1e2ef961ed74ba3de8bc72035e87fc331d28b1e92b98ae9` |
| `smoltalk-smol-magpie-ultra.jsonl` | `c1559a1c3c8908e2e95243c9870308ea1c3713de75895d553e7a0cbefedd70cb` |
| `smoltalk-smol-constraints.jsonl` | `06d5474de2248f0d2a90b2bd5dca5750a47e5ac2b1f7fa9ac16a680656bb11cc` |
| `smoltalk-smol-rewrite.jsonl` | `df9ab62db0afacda84786632dc240457ba72e870473929af0a3cfde73c3e6c2e` |
| `smoltalk-smol-summarize.jsonl` | `01d03d2b6de0a4ce15e8c355610d291576e011611cdccc1a5903846ac57e3aef` |
| `smoltalk-explore-instruct-rewriting.jsonl` | `8b1872c4a11bfb494a88d4b2fd9e0b77ee5c4bf62050f7b39d977492ad4b9b72` |
| `smoltalk-openhermes-100k.jsonl` | `3536c58e9a04fc151ff6c6c7898f7cab6afb470912460e596af06cfe243e20b0` |
| `smoltalk-metamathqa-50k.jsonl` | `ea9a289c1d74e8e0c0609b58a9902b35dcf1359338c50ad85c8bd43d32f0effb` |
| `smoltalk-numina-cot-100k.jsonl` | `10bc8a140ed76ca143742d33c262d0cb021a2494021b24cf782e07feaca84299` |
| `smoltalk-self-oss-instruct.jsonl` | `d7663c6c1ed1813883624d16103d171bc884024279e123a386c879c7a3b7216d` |
| `xlam.jsonl` | `85a51f946219a1431031e252c6ecf7e11da9079a6658f28fd154d79c05b6ae3b` |
