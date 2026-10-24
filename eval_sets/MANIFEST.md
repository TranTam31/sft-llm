# Manifest bộ eval đã đóng băng

Sinh bởi `python -m src.eval.manifest --write`. Đừng sửa tay. Mọi lần chạy eval để báo cáo phải qua `python -m src.eval.manifest` trước. Dev/test nằm trên dataset Hub private (thư mục `eval_sets/`); `p.jsonl` nằm trong repo. Đổi một file sau khi đóng băng là đổi giao thức: ghi lý do vào `EXPERIMENTS.md` (PLAN 3.1).

Bản trên Hub: dataset `trantam31/sft-capability-data`, commit `642177462ebe8942ef129b071a9e9d927a004121` (2026-10-24), thư mục `eval_sets/`.

| File | SHA-256 |
|---|---|
| `dev/c.jsonl` | `8b29f970f9a5c05802c85c769c7df9585f1fdef9e0651e6726e50db6bf1751cf` |
| `dev/t_rename_nonsense.jsonl` | `7d2f6263ca7ec1b19d6d253ac5b3bc365d72686fb6da6a136c5cc565a46a3d5c` |
| `dev/t_rename_synonym.jsonl` | `23594a3627a495207b398dc848b1794d19e8f81cbc08475ea87bd8c6a31941f3` |
| `dev/t_seen.jsonl` | `41fd2763f7f9a34189788a0e590df04284c81373b7cf564f6c13fcbf5d73b7e2` |
| `dev/t_unseen.jsonl` | `6cb5e3f086da57bc104164979ba49d2190b19915d2c36ce9d15b753abdcc9309` |
| `p.jsonl` | `54db40ea3e75f7fc17cf0038474b8e6b605d587e6b9c03880edd2982180e392e` |
| `test/c.jsonl` | `82f99eb714d59864c789835aa046f6048d7d5cd603a1280be502d57670651be3` |
| `test/t_rename_nonsense.jsonl` | `3dc7e341e92e55b26ee7648fb1781d5f4e66adc8f0573becc86b96e0881b0903` |
| `test/t_rename_synonym.jsonl` | `a6d4b62e3cea59cb93c52ba7665da65bb0fb995587ce16da8501c89575359d26` |
| `test/t_seen.jsonl` | `60516ada5b16f65708083af416929dff29eedf680ae195a6e034bb66a0b9e820` |
| `test/t_unseen.jsonl` | `6e4823fe89181e1af0d05325bf4bca22f1605a701b6ed98c8ac3dfd6bd559630` |
