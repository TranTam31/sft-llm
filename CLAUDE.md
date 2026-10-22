# CLAUDE.md

- Repo là nguồn sự thật: logic ở `src/`, notebook sinh bằng `python scripts/build_playground_notebook.py`. Colab chỉ là môi trường chạy; kết quả trong runtime phải tải về hoặc đẩy lên Hub/Drive.

## Kết nối Colab qua MCP

Dùng `scripts/colab_mcp_bridge.py` (chạy ngoài sandbox) và Chrome qua Claude in Chrome, đăng nhập tài khoản Google có quyền trên notebook. Browser pane của Claude desktop chặn kết nối tới localhost nên không dùng được.

1. `python scripts/colab_mcp_bridge.py serve` (chạy nền), rồi `python scripts/colab_mcp_bridge.py url` để lấy fragment `#mcpProxyToken=…`.
2. Mở `https://colab.research.google.com/drive/<NOTEBOOK_ID>` + fragment trong Chrome, tải lại trang, bấm **Connect**.
3. Gọi tool: `python scripts/colab_mcp_bridge.py call get_cells "{}"`. Giữ tab Chrome mở suốt phiên.

Hỏi người dùng link notebook nếu chưa có trong phiên.
