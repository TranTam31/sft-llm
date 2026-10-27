# CLAUDE.md

- Repo là nguồn sự thật: logic ở `src/`, notebook sinh bằng `python scripts/build_playground_notebook.py`. Colab chỉ là môi trường chạy; kết quả trong runtime phải tải về hoặc đẩy lên Hub/Drive.
- Test local: `.venv/Scripts/python -m unittest discover -s tests` (venv có torch CPU; máy chỉ đủ RAM cho model test cực nhỏ, model thật chạy trên Colab). Test dùng helper của test khác thì `from test_xxx import …`, không `from tests.test_xxx`: `tests/` không có `__init__.py`, trên Colab import theo package hỏng.
- Commit theo skill `commit`. Riêng dự án này: message không ghi phase (không `feat(phase1)`); thay đổi phạm vi rộng thì bỏ hẳn phần ngoặc (`feat: …`).
- Dữ liệu eval (`eval_sets/{test,dev,drafts,labels}/`) bị git-ignore vì rút từ xLAM (gated); bản gốc nằm trên dataset Hub private, thư mục `eval_sets/`. Chỉ `eval_sets/REPORT.md`, `MANIFEST.md` (hash đã đóng băng) và `p.jsonl` (bộ P tự viết, sinh từ `src/eval/probe_set.py`) được commit. Dev/test đã đóng băng: trước khi chạy eval để báo cáo, kiểm bằng `python -m src.eval.manifest`.
- Hub: dung lượng private chỉ có 100 GB. Repo model của run (snapshot, checkpoint) có thể public; output eval (`outputs.jsonl` chứa nguyên văn dev/test) và dataset luôn private, ở repo `<user>/smollm2-360m-results`. Trước khi chuyển một repo sang public, kiểm không còn `outputs.jsonl` cả trong lịch sử (squash).
- `peft` báo "incompatible version of torchao" trên Colab: Colab cài sẵn torchao 0.10; cell cài thư viện của notebook gỡ nó.

## Kết nối Colab qua MCP

Dùng `scripts/colab_mcp_bridge.py` (chạy ngoài sandbox) và Chrome qua Claude in Chrome, đăng nhập tài khoản Google có quyền trên notebook. Browser pane của Claude desktop chặn kết nối tới localhost nên không dùng được.

1. `python scripts/colab_mcp_bridge.py serve` (chạy nền), rồi `python scripts/colab_mcp_bridge.py url` để lấy fragment `#mcpProxyToken=…`. Gọi `url` ngay sau `serve` có thể còn đọc token của lần trước: gọi lại tới khi token đổi.
2. Mở `https://colab.research.google.com/drive/<NOTEBOOK_ID>` + fragment trong Chrome, tải lại trang, bấm **Connect**.
3. Gọi tool, vd. `python scripts/colab_mcp_bridge.py call run_code_cell '{"cellId": "<id>"}'`; payload lớn (nội dung cell) truyền bằng `@file.json`. Giữ tab Chrome mở suốt phiên.

Link notebook: hai tài khoản Google, mỗi tài khoản ở một profile Chrome riêng. Claude in Chrome có thể nối nhiều browser: dùng `list_connected_browsers` / `select_browser`, rồi mở `myaccount.google.com` để kiểm đúng tài khoản.
- Tài khoản cũ (hết hạn mức GPU): `00`–`03`, `02_data` = `19in1KDoL08oFmwPOcY4adSSVG8U2jYAX`.
- `tranductamhoctap@gmail.com` (có GPU, đã có secret `HF_TOKEN`): `04_bfcl` = `1JAc-p2TQJRYZ3LhKXGYXUKeSbP5zDrVo`, `05_eval` = `1-HHo7thAK92t2VzyQu4c-xSFcNWJpWwF`, `06_pilot` = `1vI5uB-0tchFhrXjr-ig2f54_dkdHfvR9`.
- Notebook mới: trang chủ Colab → Upload notebook, đưa file `.ipynb` của repo vào ô file bằng tool `file_upload`. Không thấy notebook thì hỏi người dùng.

- Mỗi lần chuyển sang notebook khác hoặc trang bị đóng: tắt bridge và chạy `serve` lại; bridge cũ treo.
- Secret `HF_TOKEN` chỉ đọc được khi cell đăng nhập chạy trên giao diện Colab: bấm vào ô code của cell đó trong Chrome rồi Ctrl+Enter (người dùng đã cho phép). Trang notebook quá lớn cho `find`/`read_page`; dùng mục lục bên trái và screenshot.
- Cell đang chạy qua MCP không dừng được bằng nút trên cell: dùng Runtime → Interrupt execution.
- Cell chạy quá 600 giây (train, quét eval): `call run_code_cell` trả "No response within 600s" nhưng cell vẫn chạy tiếp trên Colab. Đừng gọi lại bridge khi cell đó chưa xong; theo dõi bằng screenshot (dòng "Executing" ở đáy trang) hoặc file run đẩy lên Hub. Chạy lệnh `call` ở chế độ nền. Gộp train → chấm → đẩy kết quả vào một cell để commit cuối trên Hub là dấu hiệu xong; `list_repo_commits` có thể trễ vài phút, nên tìm trong toàn bộ danh sách commit chứ đừng chỉ xem commit đầu. Repo model của run bị squash sau mỗi checkpoint nên không còn commit cũ: theo dõi train bằng `progress.json` của repo đó, theo dõi chấm bằng commit trên repo kết quả.
- Tải file từ runtime bằng cách print: output cỡ vài trăm KB làm bridge treo (mọi lệnh sau đều hết giờ). Nén gzip + base64 trước khi print, mỗi lần một file. Bridge đã treo thì chạy `serve` lại.
- Runtime bị ngắt khi để nghỉ lâu (hoặc tool MCP chỉ còn `open_colab_browser_connection`): tải lại trang, Connect, rồi chạy lại cell cài thư viện, cell nạp source và cell đăng nhập HF. Hết hạn mức GPU thì chọn "Connect without GPU": dựng dữ liệu và bộ eval chạy được trên CPU (chậm hơn), hash không đổi.
- Liệt kê tiến trình kèm dòng lệnh: PowerShell `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` (xem `CommandLine`, `ParentProcessId`); `ps -W` không có dòng lệnh. Tắt bridge của mình bằng `call quit` hoặc dừng tác vụ nền đã chạy `serve`. `call quit` có thể làm `serve` thoát mà chính lệnh `call` vẫn treo chờ trả lời: kiểm lại danh sách tiến trình rồi kill đúng tiến trình `call quit` của mình.
- Bản notebook trên Drive có thể cũ hơn repo (sửa qua MCP bị mất khi trang tải lại): chạy cell nạp source và so SHA-256 bundle nó in với số `build_playground_notebook.py` in ra; lệch thì `update_cell` bằng nội dung từ notebook trong repo.
- Đừng gọi `get_cells` trên notebook có bundle source: bundle (~550 KB, gồm cả `p.jsonl`) làm bridge treo. Cell id trên Drive là id lúc tải notebook lên; id trong repo là hash nội dung nên đổi mỗi khi cell đổi. Lấy id từ bản `.ipynb` đã tải lên (hoặc ghi lại lúc tải), tìm cell trong repo theo nội dung.
- Colab miễn phí chỉ cho một session GPU: báo "Too many sessions" thì Manage sessions, tắt session cũ (kết quả của nó phải tải về trước).
- Claude in Chrome báo "not connected" thì nhờ người dùng mở Chrome; đừng chuyển sang browser pane.
- Các tiến trình `colab-mcp.exe` chạy từ `~/.codex` là của Codex, không phải bridge của dự án: không kill.
- `serve` tự chạy lại chính nó bằng Python của Codex với `--venv ~/.codex/...` (tiến trình con, `ParentProcessId` là `serve` của mình): dòng lệnh đó **không** có nghĩa Codex đang chạy bridge. Codex cũng có thể tự chạy `colab_mcp_bridge.py serve`, dùng chung `results/colab_mcp/`; hai bridge cùng lúc thì giành request của nhau (lệnh treo, trang mất kết nối). Trước khi `serve`, liệt kê tiến trình `colab_mcp_bridge.py`: bản nào không phải con của `serve` mình vừa chạy mới là của Codex, khi đó hỏi người dùng. Khi tắt bridge, chỉ kill tiến trình của mình, không kill theo tên script.
- Giữ bridge, tab Chrome và runtime suốt giữa các bước, kể cả khi xong một bước; nối lại tốn thời gian của người dùng. Chỉ dọn khi bàn giao hoặc người dùng bảo.

## Dọn trước khi bàn giao

- Chỉ khi bàn giao hoặc người dùng bảo: dừng cell đang chạy trên Colab, tắt bridge (`call quit`, hoặc kill tiến trình `colab_mcp_bridge.py`), đóng tab Chrome đã mở.

## Mạch việc

| Tag | Mạch |
|---|---|
| [SFT] | Dự án theo dõi SFT trên SmolLM2 (PLAN.md) |
