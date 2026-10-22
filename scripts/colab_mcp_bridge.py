"""File-driven bridge to the Colab MCP server, for agents without native MCP.

serve: start colab-mcp, capture the browser connection URL instead of opening
       a browser, then answer requests dropped as files.
call:  send one request and print the JSON result.

    python scripts/colab_mcp_bridge.py serve
    python scripts/colab_mcp_bridge.py url
    python scripts/colab_mcp_bridge.py call list_tools
    python scripts/colab_mcp_bridge.py call get_cells "{}"

Runtime files (token, requests, results, logs) live in results/colab_mcp/,
which Git ignores. See CLAUDE.md for the connection procedure.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "results" / "colab_mcp"
DEFAULT_VENV = Path.home() / ".codex" / "mcp" / "colab-mcp" / ".venv"
# Colab must connect within this window; once colab-mcp's wait times out, its
# proxy start task is cancelled and the server never accepts a browser again.
CONNECT_TIMEOUT = 1800.0
BOOTSTRAP = r'''
import json, site, sys, webbrowser
from pathlib import Path
target, site_dir, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
del sys.argv[1:]
site.addsitedir(site_dir)
def capture(url, *args, **kwargs):
    Path(target).write_text(json.dumps({"url": url}), encoding="utf-8")
    return True
webbrowser.open_new = capture
import colab_mcp.session
colab_mcp.session.UI_CONNECTION_TIMEOUT = timeout
from colab_mcp import main
main()
'''


def base_python(venv: Path) -> Path:
    """The venv's base interpreter; uv under an MSIX app keeps it in the package's LocalCache."""
    home = next(line.split("=", 1)[1].strip() for line in (venv / "pyvenv.cfg").read_text().splitlines()
                if line.startswith("home"))
    candidates = [Path(home)]
    roaming = Path(os.environ["APPDATA"])
    if Path(home).is_relative_to(roaming):
        packages = Path(os.environ["LOCALAPPDATA"]) / "Packages"
        candidates += [package / "LocalCache" / "Roaming" / Path(home).relative_to(roaming)
                       for package in packages.glob("*")]
    for candidate in candidates:
        if (candidate / "python.exe").exists():
            return candidate / "python.exe"
    raise FileNotFoundError(f"Base interpreter for {venv} not found; checked {len(candidates)} locations.")


def log(value):
    with (WORK / "status.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": time.strftime("%H:%M:%S"), **value}, ensure_ascii=False) + "\n")


async def serve_loop(venv: Path):
    from datetime import timedelta
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    site_dir = str(venv / "Lib" / "site-packages")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", BOOTSTRAP, str(WORK / "url.json"), site_dir, str(CONNECT_TIMEOUT)],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    with (WORK / "server_stderr.log").open("a", encoding="utf-8") as errors:
        async with stdio_client(params, errlog=errors) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                log({"status": "waiting_for_browser"})
                result = await session.call_tool("open_colab_browser_connection", {},
                                                 read_timeout_seconds=timedelta(seconds=CONNECT_TIMEOUT + 60))
                connected = bool((result.structuredContent or {}).get("result"))
                log({"status": "connected" if connected else "connect_timeout"})
                if not connected:
                    return
                await handle_requests(session)


async def handle_requests(session):
    import asyncio

    requests, results = WORK / "req", WORK / "res"
    while True:
        for request_file in sorted(requests.glob("*.json")):
            try:
                request = json.loads(request_file.read_text(encoding="utf-8"))
                request_file.unlink()
                if request["name"] == "quit":
                    log({"status": "quit"})
                    return
                if request["name"] == "list_tools":
                    payload = (await session.list_tools()).model_dump(mode="json")
                else:
                    payload = (await session.call_tool(request["name"], request.get("arguments", {}))).model_dump(mode="json")
            except Exception as error:
                payload = {"bridge_error": repr(error)}
            temporary = results / (request_file.stem + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            temporary.replace(results / request_file.name)
        await asyncio.sleep(0.3)


def serve(venv: Path):
    for folder in (WORK, WORK / "req", WORK / "res"):
        folder.mkdir(parents=True, exist_ok=True)
    try:
        import mcp  # noqa: F401
    except ImportError:
        import site
        site.addsitedir(str(venv / "Lib" / "site-packages"))
        try:
            import mcp  # noqa: F401
        except ImportError:
            # Wrong interpreter: re-run under the venv's base Python.
            python = base_python(venv)
            sys.exit(subprocess.call([str(python), "-u", __file__, "serve", "--venv", str(venv)]))
    import asyncio
    (WORK / "url.json").unlink(missing_ok=True)
    for stale in (WORK / "req").glob("*.json"):
        stale.unlink()
    asyncio.run(serve_loop(venv))


def show_url(timeout=30):
    deadline = time.time() + timeout
    while not (WORK / "url.json").exists():
        if time.time() > deadline:
            sys.exit("No connection URL yet; is `serve` running?")
        time.sleep(0.5)
    print("#" + json.loads((WORK / "url.json").read_text(encoding="utf-8"))["url"].split("#", 1)[1])


def call(name, arguments, timeout):
    request_id = f"{time.time_ns()}"
    request = WORK / "req" / f"{request_id}.json"
    response = WORK / "res" / f"{request_id}.json"
    request.write_text(json.dumps({"name": name, "arguments": json.loads(arguments)}), encoding="utf-8")
    deadline = time.time() + timeout
    while not response.exists():
        if time.time() > deadline:
            sys.exit(f"No response within {timeout}s; check results/colab_mcp/status.jsonl.")
        time.sleep(0.3)
    sys.stdout.reconfigure(encoding="utf-8")
    print(response.read_text(encoding="utf-8"))
    response.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    commands.add_parser("url")
    call_parser = commands.add_parser("call")
    call_parser.add_argument("name", help="Tool name, list_tools, or quit.")
    call_parser.add_argument("arguments", nargs="?", default="{}", help="Tool arguments as JSON.")
    call_parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.command == "serve":
        serve(args.venv)
    elif args.command == "url":
        show_url()
    else:
        call(args.name, args.arguments, args.timeout)


if __name__ == "__main__":
    main()
