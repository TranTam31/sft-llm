# CLAUDE.md

- Repo là nguồn sự thật: logic ở `src/`, notebook sinh bằng `python scripts/build_playground_notebook.py`. Colab chỉ là môi trường chạy; kết quả trong runtime phải tải về hoặc đẩy lên Hub/Drive.
- Test local: `.venv/Scripts/python -m unittest discover -s tests` (venv có torch CPU; máy chỉ đủ RAM cho model test cực nhỏ, model thật chạy trên Colab).

## Kết nối Colab qua MCP

Dùng `scripts/colab_mcp_bridge.py` (chạy ngoài sandbox) và Chrome qua Claude in Chrome, đăng nhập tài khoản Google có quyền trên notebook. Browser pane của Claude desktop chặn kết nối tới localhost nên không dùng được.

1. `python scripts/colab_mcp_bridge.py serve` (chạy nền), rồi `python scripts/colab_mcp_bridge.py url` để lấy fragment `#mcpProxyToken=…`.
2. Mở `https://colab.research.google.com/drive/<NOTEBOOK_ID>` + fragment trong Chrome, tải lại trang, bấm **Connect**.
3. Gọi tool: `python scripts/colab_mcp_bridge.py call get_cells "{}"`. Giữ tab Chrome mở suốt phiên.

Link notebook: tìm tên file (vd. `02_data`) trên Drive trong Chrome rồi mở; không thấy thì hỏi người dùng.

- Mỗi lần chuyển sang notebook khác hoặc trang bị đóng: tắt bridge và chạy `serve` lại; bridge cũ treo.
- Secret `HF_TOKEN` chỉ đọc được khi cell đăng nhập chạy trên giao diện Colab: bấm vào ô code của cell đó trong Chrome rồi Ctrl+Enter (người dùng đã cho phép). Trang notebook quá lớn cho `find`/`read_page`; dùng mục lục bên trái và screenshot.
- Cell đang chạy qua MCP không dừng được bằng nút trên cell: dùng Runtime → Interrupt execution.
- Giữ bridge, tab Chrome và runtime suốt giữa các bước, kể cả khi xong một bước; nối lại tốn thời gian của người dùng. Chỉ dọn khi bàn giao hoặc người dùng bảo.

## Dọn trước khi bàn giao

- Chỉ khi bàn giao hoặc người dùng bảo: dừng cell đang chạy trên Colab, tắt bridge (`call quit`, hoặc kill tiến trình `colab_mcp_bridge.py`), đóng tab Chrome đã mở.

## Mạch việc

| Tag | Mạch |
|---|---|
| [SFT] | Dự án theo dõi SFT trên SmolLM2 (PLAN.md) |
