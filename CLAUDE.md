# CLAUDE.md

- Repo là nguồn sự thật: logic ở `src/`, notebook sinh bằng `python scripts/build_playground_notebook.py`. Colab chỉ là môi trường chạy; kết quả trong runtime phải tải về hoặc đẩy lên Hub/Drive.
- Test local: `.venv/Scripts/python -m unittest discover -s tests` (venv có torch CPU; máy chỉ đủ RAM cho model test cực nhỏ, model thật chạy trên Colab).

## Kết nối Colab qua MCP

Dùng `scripts/colab_mcp_bridge.py` (chạy ngoài sandbox) và Chrome qua Claude in Chrome, đăng nhập tài khoản Google có quyền trên notebook. Browser pane của Claude desktop chặn kết nối tới localhost nên không dùng được.

1. `python scripts/colab_mcp_bridge.py serve` (chạy nền), rồi `python scripts/colab_mcp_bridge.py url` để lấy fragment `#mcpProxyToken=…`. Gọi `url` ngay sau `serve` có thể còn đọc token của lần trước: gọi lại tới khi token đổi.
2. Mở `https://colab.research.google.com/drive/<NOTEBOOK_ID>` + fragment trong Chrome, tải lại trang, bấm **Connect**.
3. Gọi tool: `python scripts/colab_mcp_bridge.py call get_cells "{}"`. Giữ tab Chrome mở suốt phiên.

Link notebook: tìm tên file (vd. `02_data`) trên Drive trong Chrome rồi mở; không thấy thì hỏi người dùng.

- Mỗi lần chuyển sang notebook khác hoặc trang bị đóng: tắt bridge và chạy `serve` lại; bridge cũ treo.
- Secret `HF_TOKEN` chỉ đọc được khi cell đăng nhập chạy trên giao diện Colab: bấm vào ô code của cell đó trong Chrome rồi Ctrl+Enter (người dùng đã cho phép). Trang notebook quá lớn cho `find`/`read_page`; dùng mục lục bên trái và screenshot.
- Cell đang chạy qua MCP không dừng được bằng nút trên cell: dùng Runtime → Interrupt execution.
- Tải file từ runtime bằng cách print: output cỡ vài trăm KB làm bridge treo (mọi lệnh sau đều hết giờ). Nén gzip + base64 trước khi print, mỗi lần một file. Bridge đã treo thì chạy `serve` lại.
- Bản notebook trên Drive có thể cũ hơn repo (sửa qua MCP bị mất khi trang tải lại): trước khi chạy, so `get_cells` với notebook trong repo và `update_cell` theo cell id. Cell nạp source in SHA-256 bundle; phải khớp số `build_playground_notebook.py` in ra.
- Claude in Chrome báo "not connected" thì nhờ người dùng mở Chrome; đừng chuyển sang browser pane.
- Các tiến trình `colab-mcp.exe` chạy từ `~/.codex` là của Codex, không phải bridge của dự án: không kill.
- Codex cũng có thể chạy chính `colab_mcp_bridge.py serve` (bằng Python của Codex, `--venv ~/.codex/...`), và nó dùng chung `results/colab_mcp/`. Hai bridge chạy cùng lúc thì giành request của nhau: lệnh treo, trang báo mất kết nối. Trước khi `serve`, liệt kê các tiến trình `colab_mcp_bridge.py`; nếu có bản của Codex thì hỏi người dùng. Khi tắt bridge, chỉ kill tiến trình của mình, không kill theo tên script.
- Giữ bridge, tab Chrome và runtime suốt giữa các bước, kể cả khi xong một bước; nối lại tốn thời gian của người dùng. Chỉ dọn khi bàn giao hoặc người dùng bảo.

## Dọn trước khi bàn giao

- Chỉ khi bàn giao hoặc người dùng bảo: dừng cell đang chạy trên Colab, tắt bridge (`call quit`, hoặc kill tiến trình `colab_mcp_bridge.py`), đóng tab Chrome đã mở.

## Mạch việc

| Tag | Mạch |
|---|---|
| [SFT] | Dự án theo dõi SFT trên SmolLM2 (PLAN.md) |
