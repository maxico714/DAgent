"""DAgent - single-file application.

Run this file (or the one-cell launcher in DAgent.ipynb) and the interface
appears. Everything below is generated from notebooks/DAgent-dev.ipynb by
tools/assemble_app.py, which is what the test suite exercises.

    python dagent_app.py
"""

# =============================================================================
# Imports and environment
# =============================================================================

# =============================================================================
# Notebook cell 4
# =============================================================================
import os
import re
import json
import time
import uuid
import html
import asyncio
import inspect
import traceback
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Any

try:                     # optional outside Jupyter/Colab
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:      # plain python: no event loop to patch
    pass

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:      # only the web tools need it
    BeautifulSoup = None

try:
    from IPython.display import display, HTML, Javascript, update_display
except ImportError:      # plain python: rendering is a no-op, the engine still works
    def display(*args, **kwargs):
        return None

    def update_display(*args, **kwargs):
        return None

    class HTML(str):
        pass

    class Javascript(str):
        pass


IN_COLAB = False
try:
    from google.colab import drive, userdata
    from google.colab import output as colab_output
    IN_COLAB = True
except ImportError:
    colab_output = None

def get_secret(name: str) -> Optional[str]:
    """Read a secret from Colab's Secrets panel, falling back to an env var."""
    if IN_COLAB:
        try:
            value = userdata.get(name)
            if value:
                return value
        except Exception:
            pass
    return os.environ.get(name)

if IN_COLAB:
    drive.mount("/content/drive")
    WORKSPACE = "/content/drive/MyDrive/DAgent"
else:
    WORKSPACE = os.path.expanduser("~/DAgent")

for subdir in (
    "checkpoints", "memory", "logs",
    "training_data/raw", "training_data/successful",
    "training_data/failed", "training_data/preferences",
    "snapshots",
):
    os.makedirs(os.path.join(WORKSPACE, subdir), exist_ok=True)

print(f"Workspace: {WORKSPACE}")
print(f"In Colab:  {IN_COLAB}")
print("Environment ready")

# =============================================================================
# Notebook cell 6
# =============================================================================
@dataclass
class AgentConfig:
    # --- LLM: any id from PROVIDERS (see section 11) -------------------
    # openrouter | nvidia | openai | gemini | ollama | llamacpp | custom
    provider: str = "openrouter"
    model: str = ""                     # blank = the provider's default model
    max_tokens: int = 4096
    temperature: float = 0.4

    # --- Custom / self-hosted endpoint (used only when provider == "custom")
    base_url: str = ""                  # e.g. "https://my-host/v1"
    models_url: str = ""                # optional override for the /models URL
    key_env: str = "CUSTOM_API_KEY"     # env var / Colab secret holding the key
    provider_label: str = "Custom"

    # Execution
    max_steps_per_run: int = 20
    tool_timeout_seconds: int = 120

    # "safe" | "autopilot" | "dry_run"
    mode: str = "safe"

    # Output protection
    max_tool_output_chars: int = 20_000
    max_file_read_chars: int = 20_000

    # Workspace
    workspace: str = WORKSPACE
    agent_id: str = "kernel-agent-01"

CONFIG = AgentConfig()

print("Configuration loaded")
print(f"Provider: {CONFIG.provider}")
print(f"Model:    {CONFIG.model or '(provider default)'}")
print(f"Mode:     {CONFIG.mode}")
print(f"Steps:    {CONFIG.max_steps_per_run}")

# =============================================================================
# Notebook cell 8
# =============================================================================
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable
    permission: str = "read"
    destructive: bool = False


TOOL_REGISTRY: dict[str, Tool] = {}

PERMISSIONS = {
    "read":        {"label": "Read",           "requires_approval": False},
    "network":     {"label": "Network",        "requires_approval": False},
    "write":       {"label": "Write",          "requires_approval": True},
    "execute":     {"label": "Execute",        "requires_approval": True},
    "package":     {"label": "Package Install","requires_approval": True},
    "git":         {"label": "Git",            "requires_approval": True},
    "destructive": {"label": "Destructive",    "requires_approval": True},
}

PY_TYPE_TO_JSON = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object",
}


def tool(name: Optional[str] = None, description: Optional[str] = None,
         permission: str = "read", destructive: bool = False):
    """Register a Python function as an agent tool."""
    if permission not in PERMISSIONS:
        raise ValueError(f"Unknown permission '{permission}'. Available: {', '.join(PERMISSIONS)}")

    def decorator(fn):
        tool_name = name or fn.__name__
        doc = description or (fn.__doc__ or "").strip() or f"Tool {tool_name}"
        sig = inspect.signature(fn)

        properties, required = {}, []
        for pname, p in sig.parameters.items():
            ptype = p.annotation if p.annotation is not inspect.Parameter.empty else str
            json_type = PY_TYPE_TO_JSON.get(ptype, "string")
            properties[pname] = {"type": json_type}
            if p.default is inspect.Parameter.empty:
                required.append(pname)

        schema = {"type": "object", "properties": properties, "required": required}

        TOOL_REGISTRY[tool_name] = Tool(
            name=tool_name, description=doc, parameters=schema,
            fn=fn, permission=permission, destructive=destructive,
        )
        fn._tool_name = tool_name
        return fn

    return decorator


def tools_for_openai() -> list:
    """Convert the tool registry into OpenAI function-calling schema."""
    out = []
    for t in TOOL_REGISTRY.values():
        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        })
    return out


def validate_tool_call(tool_name: str, arguments: Any) -> tuple[bool, str]:
    t = TOOL_REGISTRY.get(tool_name)
    if t is None:
        return False, f"Unknown tool: '{tool_name}'"
    if not isinstance(arguments, dict):
        return False, f"Tool '{tool_name}' arguments must be a JSON object, got {type(arguments).__name__}."

    props = t.parameters.get("properties", {})
    required = t.parameters.get("required", [])

    missing = [n for n in required if n not in arguments]
    if missing:
        return False, f"Missing required argument(s) for '{tool_name}': {', '.join(missing)}"

    unexpected = [n for n in arguments if n not in props]
    if unexpected:
        return False, f"Unexpected argument(s) for '{tool_name}': {', '.join(unexpected)}"

    return True, ""


print("Tool registry ready")

# =============================================================================
# Notebook cell 10
# =============================================================================
class ApprovalStore:
    """Single source of truth for pending/decided approval requests."""

    def __init__(self):
        self.pending: dict[str, dict] = {}

    def new_request(self, tool_name: str, details: str) -> str:
        request_id = str(uuid.uuid4())
        self.pending[request_id] = {
            "status": "pending",
            "tool": tool_name,
            "details": details,
            "message": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return request_id

    def resolve(self, request_id: str, decision: str, message: str = "") -> bool:
        if request_id not in self.pending:
            return False
        self.pending[request_id]["status"] = decision
        self.pending[request_id]["message"] = message
        return True

    def get(self, request_id: str) -> Optional[dict]:
        return self.pending.get(request_id)

    def clear_all_pending(self):
        for req in self.pending.values():
            if req["status"] == "pending":
                req["status"] = "denied"
                req["message"] = "cleared"


APPROVALS = ApprovalStore()


def _approval_decision_callback(request_id: str, decision: str, message: str = ""):
    """Invoked from JS via google.colab.kernel.invokeFunction when a button is clicked."""
    ok = APPROVALS.resolve(request_id, decision, message)
    return {"ok": ok}


if IN_COLAB:
    colab_output.register_callback("dagent.approve", _approval_decision_callback)


def render_approval_request(request_id: str, tool_name: str, details: str):
    """Render the approve/deny/modify card. Safe to call repeatedly."""
    safe_details = html.escape(details)
    safe_tool = html.escape(tool_name)
    div_id = f"ka-approval-{request_id}"

    if IN_COLAB:
        card_html = f"""
<div id="{div_id}" style="font-family: -apple-system, Segoe UI, Roboto, sans-serif;
     border: 1px solid #e2a63b; background: #fff8ec; border-radius: 10px;
     padding: 14px 16px; margin: 8px 0; max-width: 720px;">
  <div style="font-weight: 600; font-size: 14px; color: #7a4b00; margin-bottom: 6px;">
    &#9888;&#65039; Approval required &mdash; <code>{safe_tool}</code>
  </div>
  <pre style="background:#1e1e1e; color:#d4d4d4; padding:10px; border-radius:6px;
       max-height:220px; overflow:auto; font-size:12px; white-space:pre-wrap;
       word-break:break-word;">{safe_details}</pre>
  <textarea id="{div_id}-note" placeholder="Optional note (used for Deny reason or Modify instruction)"
       style="width:100%; height:54px; margin-top:6px; box-sizing:border-box;
       font-size:12px; padding:6px; border-radius:6px; border:1px solid #ccc;"></textarea>
  <div style="margin-top:8px; display:flex; gap:8px;">
    <button id="{div_id}-approve" style="background:#1a7f37; color:white; border:none;
        padding:7px 14px; border-radius:6px; cursor:pointer; font-size:13px;">&#10003; Approve</button>
    <button id="{div_id}-modify" style="background:#8a6d00; color:white; border:none;
        padding:7px 14px; border-radius:6px; cursor:pointer; font-size:13px;">&#9998; Modify</button>
    <button id="{div_id}-deny" style="background:#cf222e; color:white; border:none;
        padding:7px 14px; border-radius:6px; cursor:pointer; font-size:13px;">&#10007; Deny</button>
  </div>
  <div id="{div_id}-status" style="margin-top:6px; font-size:12px; color:#555;"></div>
</div>
<script>
(function() {{
  const reqId = {json.dumps(request_id)};
  const root = document.getElementById({json.dumps(div_id)});
  function decide(decision) {{
    const note = document.getElementById({json.dumps(div_id)} + "-note").value;
    root.querySelectorAll("button").forEach(b => b.disabled = true);
    document.getElementById({json.dumps(div_id)} + "-status").innerText =
        "Sending " + decision + "...";
    google.colab.kernel.invokeFunction(
      "dagent.approve", [reqId, decision, note], {{}}
    ).then(function() {{
      document.getElementById({json.dumps(div_id)} + "-status").innerText =
          "\u2713 " + decision + " sent.";
      root.style.opacity = "0.55";
    }});
  }}
  document.getElementById({json.dumps(div_id)} + "-approve").onclick = () => decide("approved");
  document.getElementById({json.dumps(div_id)} + "-modify").onclick = () => decide("modified");
  document.getElementById({json.dumps(div_id)} + "-deny").onclick = () => decide("denied");
}})();
</script>
"""
        display(HTML(card_html))
    else:
        # Non-Colab fallback: block on real input() so the notebook still works in
        # plain Jupyter/local execution.
        print(f"\nApproval required for '{tool_name}':\n{details}")
        answer = input("Approve? [y]es / [n]o / [m]odify: ").strip().lower()
        note = ""
        if answer.startswith("m"):
            note = input("Instruction for the agent: ")
            APPROVALS.resolve(request_id, "modified", note)
        elif answer.startswith("y"):
            APPROVALS.resolve(request_id, "approved", "")
        else:
            note = input("Reason (optional): ")
            APPROVALS.resolve(request_id, "denied", note)


async def wait_for_approval(request_id: str, poll_interval: float = 0.3,
                             timeout: Optional[float] = 600) -> dict:
    waited = 0.0
    while True:
        req = APPROVALS.get(request_id)
        if req is None:
            return {"status": "error", "message": "request vanished"}
        if req["status"] != "pending":
            return req
        await asyncio.sleep(poll_interval)
        waited += poll_interval
        if timeout and waited > timeout:
            APPROVALS.resolve(request_id, "timeout", "no response")
            return {"status": "timeout", "message": "Approval timed out"}


print("Approval system ready (IN_COLAB =", IN_COLAB, ")")

# =============================================================================
# Notebook cell 13
# =============================================================================
def _safe_workspace_path(path: str) -> Path:
    workspace = Path(CONFIG.workspace).resolve()
    target = (workspace / path).resolve()
    try:
        target.relative_to(workspace)
    except ValueError:
        raise ValueError("Path escapes the agent workspace")
    return target


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[OUTPUT TRUNCATED: showing first {limit:,} characters]"


@tool(permission="read", description="List files and directories inside the agent workspace. Path is relative to the workspace root.")
def list_files(path: str = ".") -> str:
    target = _safe_workspace_path(path)
    if not target.exists():
        return f"ERROR: {path} not found"
    if not target.is_dir():
        return f"ERROR: {path} is not a directory"
    entries = []
    for item in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        entries.append(f"[DIR]  {item.name}/" if item.is_dir() else f"[FILE] {item.name}")
    return "\n".join(entries) if entries else "(empty directory)"


@tool(permission="read", description="Read a UTF-8 text file from the agent workspace.")
def read_file(path: str) -> str:
    target = _safe_workspace_path(path)
    if not target.exists():
        return f"ERROR: {path} not found"
    if not target.is_file():
        return f"ERROR: {path} is not a file"
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(CONFIG.max_file_read_chars)
    except Exception:
        return f"ERROR reading {path}:\n{traceback.format_exc()}"
    return content


@tool(permission="read", description="Search for a text substring across files in the workspace. Returns matching file paths and line snippets.")
def search_files(query: str, path: str = ".") -> str:
    target = _safe_workspace_path(path)
    if not target.exists():
        return f"ERROR: {path} not found"
    matches = []
    for f in target.rglob("*"):
        if f.is_file() and "snapshots" not in f.parts:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if query.lower() in line.lower():
                    rel = f.relative_to(Path(CONFIG.workspace))
                    matches.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(matches) >= 50:
                        break
        if len(matches) >= 50:
            break
    return _truncate("\n".join(matches) if matches else "No matches found.", CONFIG.max_tool_output_chars)


@tool(permission="read", description="Get metadata (size, type, modified time) about a file or directory.")
def get_file_info(path: str) -> str:
    target = _safe_workspace_path(path)
    if not target.exists():
        return f"ERROR: {path} not found"
    stat = target.stat()
    return json.dumps({
        "path": str(path),
        "type": "directory" if target.is_dir() else "file",
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }, indent=2)


def _snapshot_path_for(target: Path) -> Path:
    root = Path(CONFIG.workspace, "snapshots", datetime.now(timezone.utc).strftime("%Y%m%d"))
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%H%M%S_%f")
    return root / f"{ts}__{target.name or 'root'}"


def create_file_snapshot(path: str) -> Optional[str]:
    target = _safe_workspace_path(path)
    if not target.exists() or not target.is_file():
        return None
    snap = _snapshot_path_for(target)
    shutil.copy2(target, snap)
    return str(snap)


@tool(permission="write", description="Write UTF-8 text content to a file inside the agent workspace. Parent directories are created automatically. Existing files are snapshotted before being overwritten.")
def write_file(path: str, content: str) -> str:
    target = _safe_workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot = create_file_snapshot(path) if target.exists() else None
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    result = f"Successfully wrote {len(content):,} characters to {path}."
    if snapshot:
        result += f" (previous version snapshotted to {snapshot})"
    return result


@tool(permission="destructive", description="Delete a file inside the agent workspace. A snapshot is taken first so it can be restored with undo_file.")
def delete_file(path: str) -> str:
    target = _safe_workspace_path(path)
    if not target.exists():
        return f"ERROR: {path} not found"
    if not target.is_file():
        return f"ERROR: {path} is not a file"
    snapshot = create_file_snapshot(path)
    target.unlink()
    return f"Deleted {path}. Snapshot saved to {snapshot}."


def _find_snapshots_for_file(path: str) -> list[Path]:
    target = _safe_workspace_path(path)
    root = Path(CONFIG.workspace, "snapshots")
    if not root.exists():
        return []
    matches = [p for p in root.rglob(f"*__{target.name}") if p.is_file()]
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)


@tool(permission="write", description="Undo the most recent change to a workspace file by restoring its latest snapshot.")
def undo_file(path: str) -> str:
    target = _safe_workspace_path(path)
    snaps = _find_snapshots_for_file(path)
    if not snaps:
        return f"ERROR: No snapshot found for '{path}'."
    latest = snaps[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    current_backup = create_file_snapshot(path) if target.exists() else None
    shutil.copy2(latest, target)
    result = f"Restored '{path}' from snapshot {latest}."
    if current_backup:
        result += f" Previous current version backed up to {current_backup}."
    return result


print("Filesystem tools registered")

# =============================================================================
# Notebook cell 17
# =============================================================================
def _clean_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


@tool(permission="network", description="Search the public web for a query and return a ranked list of titles, URLs and snippets.")
def web_search(query: str, max_results: int = 8) -> str:
    max_results = max(1, min(max_results, 15))
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            timeout=15,
        )
    except Exception:
        return f"SEARCH_ERROR: {traceback.format_exc()}"

    if resp.status_code != 200:
        return f"SEARCH_ERROR: HTTP {resp.status_code} from search engine."

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for r in soup.select(".result"):
        link = r.select_one(".result__a")
        snippet = r.select_one(".result__snippet")
        if not link:
            continue
        url = link.get("href", "")
        title = _clean_text(link.get_text())
        snip = _clean_text(snippet.get_text()) if snippet else ""
        if url and title:
            results.append(f"- {title}\n  {url}\n  {snip}")
        if len(results) >= max_results:
            break

    if not results:
        return "No results found (the search engine may be rate-limiting; try again or rephrase)."
    return "\n\n".join(results)


@tool(permission="network", description="Fetch a webpage by URL and extract its readable text. Use after web_search to read a source in full.")
def fetch_url(url: str) -> str:
    if not url.strip():
        return "ERROR: URL cannot be empty"
    if not (url.startswith("http://") or url.startswith("https://")):
        return "ERROR: URL must start with http:// or https://"
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                     "Accept-Language": "en-US,en;q=0.9"},
            timeout=20, allow_redirects=True,
        )
    except Exception:
        return f"FETCH_ERROR: {traceback.format_exc()}"

    if resp.status_code != 200:
        return f"FETCH_ERROR: HTTP {resp.status_code}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe", "nav", "footer", "header", "form"]):
        tag.decompose()
    title_tag = soup.find("title")
    title = _clean_text(title_tag.get_text()) if title_tag else ""
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = _clean_text(main.get_text(" ", strip=True))
    return _truncate(f"TITLE: {title}\n\n{text}", 30_000)


print("Web tools registered")

# =============================================================================
# Notebook cell 21
# =============================================================================
@tool(permission="execute", description="Execute a shell command inside the Colab environment, cwd = agent workspace. Returns stdout/stderr.")
def shell(command: str) -> str:
    if not command.strip():
        return "ERROR: shell command cannot be empty"
    try:
        completed = subprocess.run(
            command, shell=True, cwd=CONFIG.workspace,
            capture_output=True, text=True, timeout=CONFIG.tool_timeout_seconds,
        )
        out = f"Exit code: {completed.returncode}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        return _truncate(out, CONFIG.max_tool_output_chars)
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {CONFIG.tool_timeout_seconds}s."
    except Exception:
        return f"ERROR executing shell command:\n{traceback.format_exc()}"


@tool(permission="package", description="Install one or more Python packages with pip, e.g. 'pandas' or 'requests==2.32.0'.")
def install_package(packages: str) -> str:
    if not packages.strip():
        return "ERROR: package specification cannot be empty"
    try:
        completed = subprocess.run(
            ["pip", "install", "-q", *packages.split()],
            capture_output=True, text=True, timeout=CONFIG.tool_timeout_seconds,
        )
        out = f"Exit code: {completed.returncode}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        return _truncate(out, CONFIG.max_tool_output_chars)
    except Exception:
        return f"ERROR installing package(s):\n{traceback.format_exc()}"


print("Shell/package tools registered")

# =============================================================================
# Notebook cell 23
# =============================================================================
def _git_command(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=CONFIG.workspace,
            capture_output=True, text=True, timeout=CONFIG.tool_timeout_seconds,
        )
        out = f"Exit code: {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        return _truncate(out, CONFIG.max_tool_output_chars)
    except Exception:
        return f"ERROR executing git command:\n{traceback.format_exc()}"


@tool(permission="git", description="Show current Git working-tree status.")
def git_status() -> str:
    return _git_command(["status", "--short", "--branch"])


@tool(permission="git", description="Show the Git diff for current working-tree changes.")
def git_diff() -> str:
    return _git_command(["diff", "--no-ext-diff"])


@tool(permission="git", description="Show recent Git commit history.")
def git_log(limit: int = 10) -> str:
    limit = max(1, min(limit, 50))
    return _git_command(["log", f"-{limit}", "--oneline", "--decorate"])


@tool(permission="git", description="Stage all tracked changes and create a Git commit with the given message.")
def git_commit(message: str) -> str:
    if not message.strip():
        return "ERROR: commit message cannot be empty"
    _git_command(["add", "-u"])
    return _git_command(["commit", "-m", message])


print("Git tools registered:", [t for t in TOOL_REGISTRY if t.startswith('git_')])

# =============================================================================
# Notebook cell 25
# =============================================================================
async def execute_tool_call(tool_name: str, raw_arguments: Any) -> tuple[str, dict]:
    # 1. parse
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return f"TOOL_ARGUMENT_ERROR: invalid JSON arguments: {exc}", {"ok": False, "tool": tool_name}
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        return f"TOOL_ARGUMENT_ERROR: unsupported argument type {type(raw_arguments).__name__}", {"ok": False, "tool": tool_name}

    # 2. lookup
    t = TOOL_REGISTRY.get(tool_name)
    if t is None:
        return f"TOOL_ERROR: Unknown tool '{tool_name}'", {"ok": False, "tool": tool_name}

    # 3. validate
    valid, err = validate_tool_call(tool_name, arguments)
    if not valid:
        return f"TOOL_ARGUMENT_ERROR: {err}", {"ok": False, "tool": tool_name, "arguments": arguments}

    requires_approval = PERMISSIONS[t.permission]["requires_approval"]

    # 4. dry run
    if CONFIG.mode == "dry_run":
        return (f"DRY_RUN: tool was not executed.\nTool: {tool_name}\n"
                f"Arguments:\n{json.dumps(arguments, indent=2, ensure_ascii=False)}",
                {"ok": True, "dry_run": True, "tool": tool_name})

    # 5. approval gate
    #here
    if CONFIG.mode == "safe" and requires_approval:
      details = json.dumps(arguments, indent=2, ensure_ascii=False)
      request_id = APPROVALS.new_request(tool_name, details)
      render_approval_request(request_id, tool_name, details)

      return (
        f"APPROVAL_REQUIRED: waiting for user decision on '{tool_name}'.",
        {
            "ok": False,
            "tool": tool_name,
            "approval_required": True,
            "request_id": request_id,
        },
    )

    # 6. execute
    try:
        result = t.fn(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return _truncate(str(result), CONFIG.max_tool_output_chars), {"ok": True, "tool": tool_name}
    except Exception:
        return f"TOOL_EXECUTION_ERROR:\n{traceback.format_exc()}", {"ok": False, "tool": tool_name}


print("execute_tool_call ready — registered tools:", list(TOOL_REGISTRY.keys()))

# =============================================================================
# Notebook cell 29
# =============================================================================
"""DAgent provider layer.

One adapter for every OpenAI-compatible endpoint, plus live model discovery.
Replaces the duplicated NVIDIAAdapter / OpenRouterAdapter pair: adding a
provider now means adding one ProviderSpec row, not a new class.
"""
from dataclasses import dataclass, field
from typing import Optional, Any

import requests

try:
    from openai import OpenAI
except ImportError:      # only cloud providers need it; local ones work without it
    OpenAI = None


class ProviderError(RuntimeError):
    """Actionable provider failure. Message is safe to show in the UI."""


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    base_url: str
    key_env: Optional[str]
    default_model: str
    list_url: str
    list_shape: str = "openai"      # openai | gemini | ollama
    list_auth: str = "bearer"       # bearer | query | none
    extra_headers: dict = field(default_factory=dict)
    local: bool = False
    supports_tools: bool = True
    note: str = ""


# list_url is absolute so a provider can expose models somewhere other than
# {base_url}/models (Gemini and Ollama both do).
PROVIDERS = {
    "openrouter": ProviderSpec(
        id="openrouter",
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        default_model="openrouter/free",
        list_url="https://openrouter.ai/api/v1/models",
        list_auth="none",
        extra_headers={"HTTP-Referer": "https://colab.research.google.com", "X-Title": "DAgent"},
        note="Model list is public; chat needs the key.",
    ),
    "nvidia": ProviderSpec(
        id="nvidia",
        label="NVIDIA NIM",
        base_url="https://integrate.api.nvidia.com/v1",
        key_env="NVIDIA_API_KEY",
        default_model="deepseek-ai/deepseek-v4-pro-0813",
        list_url="https://integrate.api.nvidia.com/v1/models",
        note="NVIDIA-hosted models.",
    ),
    "openai": ProviderSpec(
        id="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        list_url="https://api.openai.com/v1/models",
    ),
    "gemini": ProviderSpec(
        id="gemini",
        label="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env="GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
        list_url="https://generativelanguage.googleapis.com/v1beta/models",
        list_shape="gemini",
        list_auth="query",
        note="Google exposes an OpenAI-compatible surface at /v1beta/openai/.",
    ),
    "ollama": ProviderSpec(
        id="ollama",
        label="Ollama (local)",
        base_url="http://127.0.0.1:11434/v1",
        key_env=None,
        default_model="llama3.2:3b",
        list_url="http://127.0.0.1:11434/api/tags",
        list_shape="ollama",
        list_auth="none",
        local=True,
        supports_tools=False,
        note="Must be running on the machine that hosts the notebook.",
    ),
    "llamacpp": ProviderSpec(
        id="llamacpp",
        label="llama.cpp server (local)",
        base_url="http://127.0.0.1:8080/v1",
        key_env=None,
        default_model="local-model",
        list_url="http://127.0.0.1:8080/v1/models",
        list_auth="none",
        local=True,
        supports_tools=False,
        note="Start llama-server with a GGUF file first.",
    ),
    "custom": ProviderSpec(
        id="custom",
        label="Custom OpenAI-compatible",
        base_url="",
        key_env="CUSTOM_API_KEY",
        default_model="",
        list_url="",
        note="Any endpoint that speaks /v1/chat/completions.",
    ),
}


def resolve_spec(provider_id: str, config: Any = None) -> ProviderSpec:
    """Return the spec for a provider id, filling in a custom base_url."""
    key = (provider_id or "").strip().lower()
    spec = PROVIDERS.get(key)
    if spec is None:
        supported = ", ".join(sorted(PROVIDERS))
        raise ProviderError(
            "Unknown provider '%s'. Supported: %s" % (provider_id, supported)
        )
    if key == "custom":
        base = (getattr(config, "base_url", "") or "").strip().rstrip("/")
        if not base:
            raise ProviderError(
                "Provider 'custom' needs CONFIG.base_url, e.g. https://my-host/v1"
            )
        list_url = (getattr(config, "models_url", "") or "").strip() or (base + "/models")
        key_env = getattr(config, "key_env", None) or "CUSTOM_API_KEY"
        spec = ProviderSpec(
            id="custom",
            label=getattr(config, "provider_label", None) or "Custom",
            base_url=base,
            key_env=key_env,
            default_model=getattr(config, "model", "") or "",
            list_url=list_url,
            list_auth="bearer",
            note="User-supplied endpoint.",
        )
    return spec


def api_key_for(spec: ProviderSpec) -> Optional[str]:
    """Read the key from Colab Secrets or the environment. Never logged."""
    if not spec.key_env:
        return None
    value = get_secret(spec.key_env)
    if not value:
        where = "Colab Secrets (key icon -> Secrets -> %s)" % spec.key_env
        raise ProviderError(
            "%s is not set. Add it in %s, or export it as an environment variable."
            % (spec.key_env, where)
        )
    return value


def _headers(spec: ProviderSpec, key: Optional[str]) -> dict:
    headers = dict(spec.extra_headers)
    if spec.list_auth == "bearer" and key:
        headers["Authorization"] = "Bearer " + key
    return headers


def _parse_models(spec: ProviderSpec, payload: Any) -> list:
    """Normalise the three response shapes into [{id, label, context}]."""
    rows = []
    if spec.list_shape == "gemini":
        for item in (payload or {}).get("models", []) or []:
            name = item.get("name", "")
            model_id = name.split("/", 1)[-1] if name else ""
            if not model_id:
                continue
            methods = item.get("supportedGenerationMethods") or []
            if methods and "generateContent" not in methods:
                continue
            rows.append({
                "id": model_id,
                "label": item.get("displayName") or model_id,
                "context": item.get("inputTokenLimit"),
            })
    elif spec.list_shape == "ollama":
        for item in (payload or {}).get("models", []) or []:
            model_id = item.get("name") or item.get("model") or ""
            if not model_id:
                continue
            rows.append({"id": model_id, "label": model_id, "context": None})
    else:
        for item in (payload or {}).get("data", []) or []:
            model_id = item.get("id") or ""
            if not model_id:
                continue
            rows.append({
                "id": model_id,
                "label": item.get("name") or model_id,
                "context": item.get("context_length") or item.get("context_window"),
            })
    rows.sort(key=lambda r: r["id"])
    return rows


def list_models(provider_id: str, config: Any = None, timeout: int = 15) -> list:
    """Fetch the provider's real model list. Raises ProviderError on failure."""
    spec = resolve_spec(provider_id, config)
    key = api_key_for(spec)
    url = spec.list_url
    params = {}
    if spec.list_auth == "query":
        if not key:
            raise ProviderError("%s needs a key to list models." % spec.label)
        params["key"] = key
    try:
        response = requests.get(
            url, headers=_headers(spec, key), params=params, timeout=timeout
        )
    except requests.exceptions.RequestException as exc:
        hint = " Is the local server running?" if spec.local else ""
        raise ProviderError("%s is unreachable: %s.%s" % (spec.label, exc.__class__.__name__, hint))
    if response.status_code == 401:
        raise ProviderError("%s rejected the key (401). Check %s." % (spec.label, spec.key_env))
    if response.status_code == 403:
        raise ProviderError("%s returned 403. The key may lack access to this endpoint." % spec.label)
    if response.status_code == 404:
        raise ProviderError(
            "%s has no model-list endpoint at %s. Set CONFIG.model manually." % (spec.label, url)
        )
    if response.status_code >= 400:
        raise ProviderError("%s model list failed: HTTP %d" % (spec.label, response.status_code))
    try:
        payload = response.json()
    except ValueError:
        raise ProviderError("%s returned a non-JSON model list." % spec.label)
    rows = _parse_models(spec, payload)
    if not rows:
        raise ProviderError(
            "%s returned 0 models. Check the account, or set CONFIG.model manually." % spec.label
        )
    return rows


class NormalizedResponse:
    def __init__(self, text: str, tool_calls: list, usage: dict, raw: Any = None):
        self.text = text or ""
        self.tool_calls = tool_calls or []
        self.usage = usage or {"input_tokens": 0, "output_tokens": 0}
        self.raw = raw


class OpenAICompatibleAdapter:
    """The only adapter. Works for every spec above."""

    def __init__(self, spec: ProviderSpec, model: str, max_tokens: int, temperature: float):
        self.spec = spec
        self.model = model or spec.default_model
        if not self.model:
            raise ProviderError("%s needs CONFIG.model." % spec.label)
        key = api_key_for(spec) or "not-needed"
        headers = dict(spec.extra_headers)
        if OpenAI is None:
            raise ProviderError(
                "The 'openai' package is required for %s. "
                "Run:  %%pip install openai" % spec.label
            )
        self.client = OpenAI(base_url=spec.base_url, api_key=key, default_headers=headers)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, messages: list, tools: Optional[list] = None) -> NormalizedResponse:
        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if tools and self.spec.supports_tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        completion = self.client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        msg = choice.message

        tool_calls = []
        for call in (getattr(msg, "tool_calls", None) or []):
            tool_calls.append({
                "id": call.id,
                "name": call.function.name,
                "arguments": call.function.arguments,
            })

        usage = {}
        if getattr(completion, "usage", None):
            usage = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
            }
        return NormalizedResponse(text=msg.content or "", tool_calls=tool_calls, usage=usage, raw=completion)


def build_llm(config: Any):
    """Build the adapter for CONFIG.provider. Same call site as before."""
    spec = resolve_spec(config.provider, config)
    return OpenAICompatibleAdapter(spec, config.model, config.max_tokens, config.temperature)


def provider_table() -> list:
    """Rows for the settings UI: what exists, and what it needs."""
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "needs_key": bool(spec.key_env),
            "key_env": spec.key_env,
            "local": spec.local,
            "default_model": spec.default_model,
            "note": spec.note,
        }
        for spec in sorted(PROVIDERS.values(), key=lambda s: s.label.lower())
    ]


print("Provider layer ready: %d providers (%s)" % (len(PROVIDERS), ", ".join(sorted(PROVIDERS))))

# =============================================================================
# Notebook cell 33
# =============================================================================
SYSTEM_PROMPT = """You are DAgent, an autonomous agent running inside a Google Colab
environment.

You have tools for: inspecting/reading/writing files, searching the web, fetching
webpages, executing shell commands, installing Python packages, and Git operations.

Rules:
1. Use tools whenever they help complete the task.
2. Never claim an action was completed unless a tool actually reported success.
3. Inspect the environment before assuming what files or code exist.
4. After making changes, verify the result when practical.
5. If a tool call is denied or modified by the user, adapt your plan accordingly instead
   of repeating the same call.
6. Give a concise final answer only once the task is actually done.
"""


class DAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = None  # built on first run(): constructing an agent must
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def run(self, task: str):
        if self.llm is None:
            try:
                self.llm = build_llm(self.config)
            except Exception as exc:
                yield {"type": "error",
                       "message": f"LLM setup failed: {type(exc).__name__}: {exc}"}
                return
        self.messages.append({"role": "user", "content": task})
        available_tools = tools_for_openai()

        for step in range(self.config.max_steps_per_run):
            try:
                response = self.llm.chat(messages=self.messages, tools=available_tools)
            except Exception as exc:
                yield {"type": "error", "message": f"LLM call failed: {type(exc).__name__}: {exc}"}
                return

            if response.text:
                yield {"type": "assistant_text", "content": response.text}

            if not response.tool_calls:
                assistant_msg = {"role": "assistant", "content": response.text or ""}
                self.messages.append(assistant_msg)
                yield {"type": "final", "content": response.text or "(no output)"}
                return

            # Record the assistant turn with its tool calls (OpenAI wants this shape)
            self.messages.append({
                "role": "assistant",
                "content": response.text or None,
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in response.tool_calls
                ],
            })

            for call in response.tool_calls:
                yield {"type": "tool_call", "tool": call["name"], "arguments": call["arguments"]}

                result_text, metadata = await execute_tool_call(call["name"], call["arguments"])

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result_text,
                })

                yield {"type": "tool_result", "tool": call["name"], "result": result_text, "metadata": metadata}

        yield {"type": "final", "content": f"Stopped after reaching max_steps_per_run={self.config.max_steps_per_run}."}


print("DAgent ready")

# =============================================================================
# Notebook cell 37
# =============================================================================
CHAT_AGENT = DAgent(CONFIG)

_PROVIDER_ROWS = provider_table()
_PROVIDERS_JSON = json.dumps(_PROVIDER_ROWS)
_PROVIDER_OPTIONS = "\n".join(
    '<option value="%s"%s>%s%s</option>' % (
        html.escape(p["id"]),
        " selected" if p["id"] == CONFIG.provider else "",
        html.escape(p["label"]),
        "" if not p["needs_key"] else " &middot; key: " + html.escape(p["key_env"] or ""),
    )
    for p in _PROVIDER_ROWS
)


def _render_bubble(role: str, text: str):
    colors = {
        "user":      ("#0a58ff", "#eef4ff"),
        "assistant": ("#1a7f37", "#eefbf1"),
        "tool":      ("#6e6e6e", "#f4f4f4"),
        "error":     ("#cf222e", "#fff0f0"),
    }
    border, bg = colors.get(role, ("#888", "#f4f4f4"))
    label = {"user": "You", "assistant": "DAgent", "tool": "Tool", "error": "Error"}.get(role, role)
    safe = html.escape(text).replace("\n", "<br>")
    display(HTML(f"""
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif; max-width:720px;
     margin:6px 0; padding:10px 14px; border-radius:10px; border-left:4px solid {border};
     background:{bg}; font-size:13px; line-height:1.5;">
  <div style="font-weight:600; color:{border}; margin-bottom:4px;">{label}</div>
  <div>{safe}</div>
</div>
"""))


async def _run_chat_turn(message: str, mode: str, max_steps: str,
                         provider: str = "", model: str = ""):
    if provider:
        CONFIG.provider = provider
    if model:
        CONFIG.model = model
    CONFIG.mode = mode if mode in {"safe", "autopilot", "dry_run"} else CONFIG.mode
    try:
        CONFIG.max_steps_per_run = max(1, int(max_steps))
    except (TypeError, ValueError):
        pass

    # Rebuild the adapter so a provider/model change takes effect immediately.
    try:
        CHAT_AGENT.llm = build_llm(CONFIG)
    except ProviderError as exc:
        _render_bubble("error", str(exc))
        return

    _render_bubble("user", message)

    events = []
    async for event in CHAT_AGENT.run(message):
        events.append(event)
        if event["type"] == "assistant_text":
            _render_bubble("assistant", event["content"])
        elif event["type"] == "tool_call":
            _render_bubble("tool", f"calling {event['tool']}({event['arguments']})")
        elif event["type"] == "tool_result":
            _render_bubble("tool", f"{event['tool']} -> {event['result'][:800]}")
        elif event["type"] == "final":
            _render_bubble("assistant", event["content"])
        elif event["type"] == "error":
            _render_bubble("error", event["message"])

    # Section 17 records the turn once that cell has been run. If it has not,
    # nothing breaks and nothing is written.
    _capture = globals().get("capture_turn")
    if _capture is not None:
        try:
            _capture(message, events)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            print("training capture skipped: %s: %s" % (type(exc).__name__, exc))


def _chat_send_callback(message: str, mode: str, max_steps: str,
                        provider: str = "", model: str = ""):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_run_chat_turn(message, mode, max_steps, provider, model))
    return {"ok": True}


def _chat_models_callback(provider: str):
    """Live model list for the UI. Never raises: the UI shows `error` verbatim."""
    try:
        rows = list_models(provider, CONFIG, timeout=8)
    except ProviderError as exc:
        return {"ok": False, "models": [], "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        return {"ok": False, "models": [], "error": "%s: %s" % (type(exc).__name__, exc)}
    return {"ok": True, "models": rows, "error": ""}


if IN_COLAB:
    colab_output.register_callback("dagent.send_message", _chat_send_callback)
    colab_output.register_callback("dagent.list_models", _chat_models_callback)


_CHAT_TEMPLATE = """
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif; max-width:780px;
     border:1px solid #ddd; border-radius:12px; padding:16px; background:#fafafa;">
  <div style="font-weight:700; font-size:15px; margin-bottom:12px;">&#129302; DAgent Chat</div>

  <div style="display:flex; gap:10px; margin-bottom:8px; flex-wrap:wrap;">
    <label style="font-size:12px; color:#444; flex:1 1 220px;">Provider
      <select id="ka-provider" style="display:block; width:100%; margin-top:3px; padding:5px 6px;
        border-radius:6px; border:1px solid #ccc;">__PROVIDER_OPTIONS__</select>
    </label>
    <label style="font-size:12px; color:#444; flex:1 1 220px;">Model
      <select id="ka-model" style="display:block; width:100%; margin-top:3px; padding:5px 6px;
        border-radius:6px; border:1px solid #ccc;"><option value="">Loading&hellip;</option></select>
    </label>
    <button id="ka-refresh" style="align-self:flex-end; margin-bottom:1px; background:#fff;
      border:1px solid #ccc; border-radius:6px; padding:5px 10px; font-size:12px; cursor:pointer;">
      Refresh models</button>
  </div>

  <div style="display:flex; gap:10px; margin-bottom:8px; flex-wrap:wrap;">
    <label style="font-size:12px; color:#444; flex:1 1 220px;">Mode
      <select id="ka-mode" style="display:block; width:100%; margin-top:3px; padding:5px 6px;
        border-radius:6px; border:1px solid #ccc;">
        <option value="safe"__SEL_SAFE__>safe (approval required)</option>
        <option value="autopilot"__SEL_AUTO__>autopilot</option>
        <option value="dry_run"__SEL_DRY__>dry_run</option>
      </select>
    </label>
    <label style="font-size:12px; color:#444; flex:0 0 110px;">Max steps
      <input id="ka-steps" type="number" min="1" max="100" value="__MAX_STEPS__"
        style="display:block; width:100%; margin-top:3px; padding:5px 6px; border-radius:6px;
        border:1px solid #ccc;">
    </label>
    <label style="font-size:12px; color:#444; flex:1 1 220px;">Override model id (optional)
      <input id="ka-manual" type="text" placeholder="e.g. openai/gpt-4o-mini"
        style="display:block; width:100%; margin-top:3px; padding:5px 6px; border-radius:6px;
        border:1px solid #ccc;">
    </label>
  </div>

  <div id="ka-hint" style="font-size:12px; color:#6e6e6e; margin-bottom:6px; min-height:16px;"></div>

  <textarea id="ka-input" placeholder="Ask DAgent to do something..."
    style="width:100%; height:70px; box-sizing:border-box; padding:8px; font-size:13px;
    border-radius:8px; border:1px solid #ccc; resize:vertical;"></textarea>

  <div style="margin-top:8px; display:flex; justify-content:space-between; align-items:center;">
    <span id="ka-send-status" style="font-size:12px; color:#888;"></span>
    <button id="ka-send" style="background:#0a58ff; color:white; border:none; padding:8px 18px;
      border-radius:8px; cursor:pointer; font-size:13px; font-weight:600;">Send &#9658;</button>
  </div>
</div>
<script>
(function () {
  var PROVIDERS = __PROVIDERS_JSON__;
  var provSel = document.getElementById("ka-provider");
  var modelSel = document.getElementById("ka-model");
  var manual = document.getElementById("ka-manual");
  var refresh = document.getElementById("ka-refresh");
  var hint = document.getElementById("ka-hint");
  var modeSel = document.getElementById("ka-mode");
  var stepsInput = document.getElementById("ka-steps");
  var input = document.getElementById("ka-input");
  var btn = document.getElementById("ka-send");
  var status = document.getElementById("ka-send-status");

  function setHint(text, isError) {
    hint.textContent = text || "";
    hint.style.color = isError ? "#cf222e" : "#6e6e6e";
  }

  function fillModels(models, error) {
    modelSel.innerHTML = "";
    if (!models || !models.length) {
      var opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(no models listed)";
      modelSel.appendChild(opt);
      if (error) setHint(error, true);
      return;
    }
    models.forEach(function (m) {
      var opt = document.createElement("option");
      opt.value = m.id;
      var label = m.label || m.id;
      if (m.context) label += "  \\u00b7  " + Math.round(m.context / 1000) + "k ctx";
      opt.textContent = label;
      modelSel.appendChild(opt);
    });
    setHint(models.length + " model(s) available", false);
  }

  function loadModels() {
    var p = provSel.value;
    modelSel.innerHTML = "";
    var loading = document.createElement("option");
    loading.value = "";
    loading.textContent = "Loading\\u2026";
    modelSel.appendChild(loading);
    setHint("Fetching " + p + " model list\\u2026", false);
    google.colab.kernel.invokeFunction("dagent.list_models", [p], {})
      .then(function (res) {
        var data = (res && res.data) ? res.data : res;
        fillModels(data && data.models, data && data.error);
      })
      .catch(function (err) {
        fillModels([], "Could not list models: " + err);
      });
  }

  provSel.onchange = loadModels;
  refresh.onclick = loadModels;

  function send() {
    var msg = input.value.trim();
    if (!msg) return;
    var model = manual.value.trim() || modelSel.value;
    btn.disabled = true;
    status.textContent = "Working\\u2026";
    google.colab.kernel.invokeFunction(
      "dagent.send_message",
      [msg, modeSel.value, stepsInput.value, provSel.value, model],
      {}
    ).then(function () {
      status.textContent = "Done. Scroll down for the reply.";
      btn.disabled = false;
      input.value = "";
    }).catch(function () {
      status.textContent = "Error \\u2014 see cell output.";
      btn.disabled = false;
    });
  }

  btn.onclick = send;
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  loadModels();
})();
</script>
"""

_chat_ui = (
    _CHAT_TEMPLATE
    .replace("__PROVIDER_OPTIONS__", _PROVIDER_OPTIONS)
    .replace("__PROVIDERS_JSON__", _PROVIDERS_JSON)
    .replace("__SEL_SAFE__", " selected" if CONFIG.mode == "safe" else "")
    .replace("__SEL_AUTO__", " selected" if CONFIG.mode == "autopilot" else "")
    .replace("__SEL_DRY__", " selected" if CONFIG.mode == "dry_run" else "")
    .replace("__MAX_STEPS__", str(CONFIG.max_steps_per_run))
)

def render_chat():
    """Show the chat box. Safe to call more than once."""
    display(HTML(_chat_ui))
    return _chat_ui

# =============================================================================
# Notebook cell 41
# =============================================================================
# 14 - Presentation mode: hide the code cells, keep the app
# Toggle with the "Show code" button in the bottom-right corner.
_PRESENTATION_JS = r"""(function () {
  if (window.__dagentPresentation) { return; }

  // Colab has shipped several editor engines over time. Rather than guess one
  // selector, hide whatever editor element actually sits inside a code cell.
  var EDITOR_SELECTOR = '.input_area, .inputarea, .CodeMirror, .monaco-editor';
  var state = { hidden: false, observer: null };

  function apply(hide) {
    var cells = document.querySelectorAll('.cell.code');
    for (var i = 0; i < cells.length; i++) {
      var cell = cells[i];
      var editor = cell.querySelector(EDITOR_SELECTOR);
      if (editor) { editor.style.display = hide ? 'none' : ''; }
      var gutter = cell.querySelector('.gutter');
      if (gutter) { gutter.style.display = hide ? 'none' : ''; }
    }
    document.body.classList.toggle('dagent-hide-code', hide);
  }

  function setHidden(hide) {
    state.hidden = hide;
    apply(hide);
    var btn = document.getElementById('dagent-code-toggle');
    if (btn) { btn.textContent = hide ? 'Show code' : 'Hide code'; }
  }

  function debounce(fn, ms) {
    var t = null;
    return function () {
      if (t) { clearTimeout(t); }
      t = setTimeout(fn, ms);
    };
  }

  // Colab re-renders cells after execution, which drops the inline styles.
  var reapply = debounce(function () { apply(state.hidden); }, 150);
  state.observer = new MutationObserver(function () {
    if (state.hidden) { reapply(); }
  });
  state.observer.observe(document.body, { childList: true, subtree: true });

  function makeButton() {
    if (document.getElementById('dagent-code-toggle')) { return; }
    var b = document.createElement('button');
    b.id = 'dagent-code-toggle';
    b.textContent = 'Show code';
    b.style.cssText = 'position:fixed;right:18px;bottom:18px;z-index:99999;' +
      'background:#111827;color:#fff;border:none;border-radius:999px;' +
      'padding:8px 14px;font:600 12px -apple-system,Segoe UI,Roboto,sans-serif;' +
      'cursor:pointer;opacity:.72;box-shadow:0 2px 8px rgba(0,0,0,.25);';
    b.onclick = function () { setHidden(!state.hidden); };
    document.body.appendChild(b);
  }

  window.__dagentPresentation = {
    get hidden() { return state.hidden; },
    show: function () { setHidden(false); },
    hide: function () { setHidden(true); }
  };

  makeButton();
  setHidden(true);
})();"""

def render_presentation():
    """Hide notebook code in Colab and leave a reversible Show code toggle."""
    if IN_COLAB:
        display(Javascript(_PRESENTATION_JS))
    return _PRESENTATION_JS

# =============================================================================
# Notebook cell 42
# =============================================================================
# 15 - Local models: pick the download method first, then the model.
#
# Nothing here downloads on import. `local_plan()` returns a plan, and
# `download_gguf()` executes it with resume + progress. Model names, file
# sizes and gating flags below were read from the Hugging Face API; Ollama
# tags are fetched live from ollama.com so they cannot go stale.


class LocalModelError(RuntimeError):
    """Actionable local-model failure. Safe to show in the UI."""


# The first screen of the local-model UI. Order matters: least setup first.
LOCAL_METHODS = {
    "ollama": {
        "id": "ollama",
        "label": "Ollama",
        "summary": "Ollama downloads the model and serves an OpenAI-compatible API. One command.",
        "needs": "Ollama installed on the machine running this notebook",
        "setup": [
            "Install Ollama: https://ollama.com/download",
            "Start it: ollama serve",
            "Then pull the model and point DAgent at http://127.0.0.1:11434/v1",
        ],
        "api_base": "http://127.0.0.1:11434/v1",
        "provider": "ollama",
        "serves_api": True,
    },
    "gguf": {
        "id": "gguf",
        "label": "GGUF file (llama.cpp)",
        "summary": "Download one quantised .gguf file here, then serve it with llama-server.",
        "needs": "llama.cpp built locally so `llama-server` is on PATH",
        "setup": [
            "Build llama.cpp: https://github.com/ggml-org/llama.cpp",
            "Serve the downloaded file: llama-server -m <file> --port 8080",
            "Point DAgent at http://127.0.0.1:8080/v1",
        ],
        "api_base": "http://127.0.0.1:8080/v1",
        "provider": "llamacpp",
        "serves_api": True,
    },
    "hf": {
        "id": "hf",
        "label": "Hugging Face snapshot",
        "summary": "Full-precision weights for transformers, vLLM or fine-tuning. Largest download.",
        "needs": "transformers/torch or vLLM; a token for gated repositories",
        "setup": [
            "pip install -U huggingface_hub",
            "huggingface-cli login   (only for gated repositories)",
            "Serve with vLLM: vllm serve <repo> --port 8000",
        ],
        "api_base": "",
        "provider": None,
        "serves_api": False,
    },
}

# Every row was checked against the Hugging Face API: the GGUF repository
# exists, the exact file exists, `bytes` is its real size, and `gated` is the
# upstream repository's gating flag (gated repos need a token for `hf`).
LOCAL_CATALOG = [
    {
        "id": "qwen2.5-1.5b",
        "label": "Qwen2.5 1.5B Instruct",
        "params": "1.5B",
        "license": "apache-2.0",
        "note": "Smallest useful general model; comfortable on CPU.",
        "gguf": {"repo": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
                 "file": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf", "bytes": 986048768},
        "ollama": {"library": "qwen2.5", "tag": "1.5b"},
        "hf": {"repo": "Qwen/Qwen2.5-1.5B-Instruct", "gated": False},
    },
    {
        "id": "llama3.2-1b",
        "label": "Llama 3.2 1B Instruct",
        "params": "1B",
        "license": "llama3.2",
        "note": "Fastest option; good for routing and short answers.",
        "gguf": {"repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
                 "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "bytes": 807694464},
        "ollama": {"library": "llama3.2", "tag": "1b"},
        "hf": {"repo": "meta-llama/Llama-3.2-1B-Instruct", "gated": True},
    },
    {
        "id": "gemma2-2b",
        "label": "Gemma 2 2B Instruct",
        "params": "2B",
        "license": "gemma",
        "note": "Strong small model; Gemma licence terms apply.",
        "gguf": {"repo": "bartowski/gemma-2-2b-it-GGUF",
                 "file": "gemma-2-2b-it-Q4_K_M.gguf", "bytes": 1708582752},
        "ollama": {"library": "gemma2", "tag": "2b"},
        "hf": {"repo": "google/gemma-2-2b-it", "gated": True},
    },
    {
        "id": "qwen2.5-3b",
        "label": "Qwen2.5 3B Instruct",
        "params": "3B",
        "license": "other",
        "note": "Best quality/VRAM balance at 3B. Upstream licence is 'other' - read it before shipping.",
        "gguf": {"repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
                 "file": "Qwen2.5-3B-Instruct-Q4_K_M.gguf", "bytes": 1929903264},
        "ollama": {"library": "qwen2.5", "tag": "3b"},
        "hf": {"repo": "Qwen/Qwen2.5-3B-Instruct", "gated": False},
    },
    {
        "id": "llama3.2-3b",
        "label": "Llama 3.2 3B Instruct",
        "params": "3B",
        "license": "llama3.2",
        "note": "Solid general model with tool-calling support.",
        "gguf": {"repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
                 "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "bytes": 2019377696},
        "ollama": {"library": "llama3.2", "tag": "3b"},
        "hf": {"repo": "meta-llama/Llama-3.2-3B-Instruct", "gated": True},
    },
    {
        "id": "qwen2.5-coder-7b",
        "label": "Qwen2.5 Coder 7B Instruct",
        "params": "7B",
        "license": "apache-2.0",
        "note": "Code-focused; needs about 5 GB of free RAM or VRAM.",
        "gguf": {"repo": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
                 "file": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf", "bytes": 4683074336},
        "ollama": {"library": "qwen2.5-coder", "tag": "7b"},
        "hf": {"repo": "Qwen/Qwen2.5-Coder-7B-Instruct", "gated": False},
    },
    {
        "id": "qwen2.5-7b",
        "label": "Qwen2.5 7B Instruct",
        "params": "7B",
        "license": "apache-2.0",
        "note": "Best general quality here; needs about 5 GB of free RAM or VRAM.",
        "gguf": {"repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
                 "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "bytes": 4683074240},
        "ollama": {"library": "qwen2.5", "tag": "7b"},
        "hf": {"repo": "Qwen/Qwen2.5-7B-Instruct", "gated": False},
    },
]

# Ollama library names whose tags are fetched live. All six were confirmed to
# resolve on ollama.com; tags themselves are never hardcoded.
OLLAMA_LIBRARY = ("qwen2.5", "llama3.2", "gemma2", "qwen2.5-coder", "smollm2", "phi3.5")


def human_bytes(n):
    """Format a byte count for display. None stays None."""
    if n is None:
        return "unknown"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return ("%d %s" % (value, unit)) if unit == "B" else ("%.1f %s" % (value, unit))
        value /= 1024
    return "%.1f TB" % value


def local_methods():
    """Rows for the first screen. No network."""
    return [
        {
            "id": m["id"],
            "label": m["label"],
            "summary": m["summary"],
            "needs": m["needs"],
            "setup": list(m["setup"]),
            "serves_api": m["serves_api"],
        }
        for m in LOCAL_METHODS.values()
    ]


def local_entry(model_id):
    """Look up one catalogue row, or raise with the valid ids."""
    for entry in LOCAL_CATALOG:
        if entry["id"] == model_id:
            return entry
    known = ", ".join(e["id"] for e in LOCAL_CATALOG)
    raise LocalModelError("Unknown local model '%s'. Known: %s" % (model_id, known))


def local_models(method_id):
    """Second screen: the catalogue, annotated for the chosen method."""
    method = LOCAL_METHODS.get((method_id or "").strip().lower())
    if method is None:
        known = ", ".join(LOCAL_METHODS)
        raise LocalModelError("Unknown download method '%s'. Choose one of: %s" % (method_id, known))

    rows = []
    for entry in LOCAL_CATALOG:
        if method_id == "ollama":
            tag = entry["ollama"]
            rows.append({
                "id": entry["id"],
                "label": entry["label"],
                "params": entry["params"],
                "license": entry["license"],
                "note": entry["note"],
                "ref": "%s:%s" % (tag["library"], tag["tag"]),
                "size_bytes": None,
                "size_text": "chosen by Ollama",
                "warning": "",
            })
        elif method_id == "gguf":
            gguf = entry["gguf"]
            rows.append({
                "id": entry["id"],
                "label": entry["label"],
                "params": entry["params"],
                "license": entry["license"],
                "note": entry["note"],
                "ref": gguf["file"],
                "size_bytes": gguf["bytes"],
                "size_text": human_bytes(gguf["bytes"]),
                "warning": "licence 'other' - review before redistribution"
                if entry["license"] == "other" else "",
            })
        else:
            hf = entry["hf"]
            rows.append({
                "id": entry["id"],
                "label": entry["label"],
                "params": entry["params"],
                "license": entry["license"],
                "note": entry["note"],
                "ref": hf["repo"],
                "size_bytes": None,
                "size_text": "full precision, several GB",
                "warning": "gated: needs a Hugging Face token and licence acceptance"
                if hf["gated"] else "",
            })
    return rows


def fetch_ollama_tags(library, timeout=20):
    """Read the real tag list for an Ollama library from ollama.com.

    Returns plain size tags (e.g. ['1.5b', '3b']) plus 'latest'. Raises
    LocalModelError when the page cannot be read, so the caller can fall back
    to a manual tag instead of inventing one.
    """
    library = (library or "").strip()
    if library not in OLLAMA_LIBRARY:
        raise LocalModelError(
            "Ollama library '%s' is not in OLLAMA_LIBRARY. Known: %s"
            % (library, ", ".join(OLLAMA_LIBRARY))
        )
    url = "https://ollama.com/library/%s/tags" % library
    try:
        response = requests.get(url, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise LocalModelError("Could not reach %s: %s" % (url, exc.__class__.__name__))
    if response.status_code != 200:
        raise LocalModelError("%s returned HTTP %d" % (url, response.status_code))

    pattern = r'href="/library/%s:([^"/]+)"' % re.escape(library)
    seen, ordered = set(), []
    for tag in re.findall(pattern, response.text):
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)

    # Keep the tags a normal user wants: a size, optionally -instruct, or latest.
    simple = [t for t in ordered if t == "latest" or re.fullmatch(r"\d+(\.\d+)?b(-instruct)?", t)]
    if not simple:
        raise LocalModelError("%s listed no simple size tags; use a manual tag." % url)
    return simple


def local_plan(model_id, method_id, dest_dir=None):
    """Describe exactly what would happen, without doing any of it."""
    method_id = (method_id or "").strip().lower()
    method = LOCAL_METHODS.get(method_id)
    if method is None:
        raise LocalModelError(
            "Unknown download method '%s'. Choose one of: %s" % (method_id, ", ".join(LOCAL_METHODS))
        )
    entry = local_entry(model_id)
    dest_root = dest_dir or os.path.join(WORKSPACE, "models")

    if method_id == "ollama":
        ref = "%s:%s" % (entry["ollama"]["library"], entry["ollama"]["tag"])
        return {
            "method": "ollama",
            "model_id": model_id,
            "label": entry["label"],
            "ref": ref,
            "command": "ollama pull %s" % ref,
            "api_base": method["api_base"],
            "provider": method["provider"],
            "serves_api": True,
            "bytes": None,
            "dest": None,
            "downloadable": False,
            "note": "Run the command on the machine hosting Ollama; DAgent only needs the API base.",
        }

    if method_id == "gguf":
        gguf = entry["gguf"]
        dest = os.path.join(dest_root, gguf["file"])
        return {
            "method": "gguf",
            "model_id": model_id,
            "label": entry["label"],
            "ref": gguf["file"],
            "url": "https://huggingface.co/%s/resolve/main/%s" % (gguf["repo"], gguf["file"]),
            "filename": gguf["file"],
            "dest": dest,
            "bytes": gguf["bytes"],
            "size_text": human_bytes(gguf["bytes"]),
            "command": 'llama-server -m "%s" --port 8080' % dest,
            "api_base": method["api_base"],
            "provider": method["provider"],
            "serves_api": True,
            "downloadable": True,
            "license": entry["license"],
            "note": "After llama-server starts, refresh the model list in the chat box.",
        }

    hf = entry["hf"]
    dest = os.path.join(dest_root, "hf", hf["repo"].split("/")[-1])
    command = "huggingface-cli download %s --local-dir \"%s\"" % (hf["repo"], dest)
    if hf["gated"]:
        command += "  # gated: run `huggingface-cli login` first"
    return {
        "method": "hf",
        "model_id": model_id,
        "label": entry["label"],
        "ref": hf["repo"],
        "command": command,
        "dest": dest,
        "bytes": None,
        "gated": hf["gated"],
        "api_base": method["api_base"],
        "provider": None,
        "serves_api": False,
        "downloadable": False,
        "license": entry["license"],
        "note": "This is a weights download for training or vLLM, not an OpenAI-compatible endpoint.",
    }


def download_gguf(plan, progress=None, session=None, chunk_size=1 << 20, timeout=120):
    """Download a GGUF plan with resume, progress and size verification.

    - writes to `<dest>.part` and renames only when complete, so a killed
      download never looks finished
    - resumes with a Range request when `.part` already exists; if the server
      ignores the range (200 instead of 206) it restarts cleanly instead of
      appending and corrupting the file
    - verifies the final size against the catalogue when one is known
    """
    if plan.get("method") != "gguf":
        raise LocalModelError("download_gguf() only handles gguf plans, got %r" % plan.get("method"))
    url = plan.get("url") or ""
    if not url.startswith(("https://", "http://")):
        raise LocalModelError("Refusing to download from %r: only http(s) URLs are allowed." % url)

    dest = plan["dest"]
    expected = plan.get("bytes")
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    part = dest + ".part"

    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if expected is None or size == expected:
            return {"status": "already", "bytes": size, "path": dest}
        raise LocalModelError(
            "%s exists but is %s, expected %s. Delete it and retry."
            % (dest, human_bytes(size), human_bytes(expected))
        )

    offset = os.path.getsize(part) if os.path.exists(part) else 0
    if expected is not None and offset > expected:
        os.remove(part)
        offset = 0

    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = "bytes=%d-" % offset

    http = session or requests.Session()
    try:
        response = http.get(url, headers=headers, stream=True, timeout=(15, timeout))
    except requests.exceptions.RequestException as exc:
        raise LocalModelError("Download failed: %s: %s" % (exc.__class__.__name__, exc))

    status = response.status_code
    if status == 416:
        # Nothing left to send: the part file is already complete.
        if expected is not None and offset == expected:
            os.replace(part, dest)
            return {"status": "resumed", "bytes": offset, "path": dest}
        raise LocalModelError(
            "Server rejected the resume range at %s. Delete %s and retry."
            % (human_bytes(offset), part)
        )
    if status >= 400:
        raise LocalModelError("Download failed: HTTP %d from %s" % (status, url))

    if status == 206:
        mode = "ab"
        length = response.headers.get("Content-Length")
        total = (offset + int(length)) if length is not None else expected
        resumed = bool(offset)
    else:
        # Server ignored the Range header: start over rather than append.
        mode = "wb"
        offset = 0
        length = response.headers.get("Content-Length")
        total = int(length) if length is not None else expected
        resumed = False

    done = offset
    try:
        with open(part, mode) as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    finally:
        response.close()

    if total is not None and done != total:
        raise LocalModelError(
            "Incomplete download: %s of %s. Run it again to resume."
            % (human_bytes(done), human_bytes(total))
        )
    if expected is not None and done != expected:
        raise LocalModelError(
            "Size mismatch: got %s, expected %s. Delete %s and retry."
            % (human_bytes(done), human_bytes(expected), part)
        )

    os.replace(part, dest)
    return {"status": "resumed" if resumed else "downloaded", "bytes": done, "path": dest}


def ollama_pull_command(model_id):
    """The exact `ollama pull` line for a catalogue entry."""
    entry = local_entry(model_id)
    return "ollama pull %s:%s" % (entry["ollama"]["library"], entry["ollama"]["tag"])


def llama_server_command(path, port=8080):
    """The exact `llama-server` line for a downloaded GGUF file."""
    return 'llama-server -m "%s" --port %d' % (path, port)


def apply_local_model(model_id, method_id):
    """Point CONFIG at the chosen local model. Does not start any server."""
    method_id = (method_id or "").strip().lower()
    plan = local_plan(model_id, method_id)

    if method_id == "ollama":
        CONFIG.provider = "ollama"
        CONFIG.model = plan["ref"]
        CONFIG.base_url = ""
        return {"provider": CONFIG.provider, "model": CONFIG.model,
                "message": "Set provider=ollama, model=%s. Make sure `ollama serve` is running." % plan["ref"]}

    if method_id == "gguf":
        CONFIG.provider = "llamacpp"
        CONFIG.model = ""
        CONFIG.base_url = ""
        return {"provider": CONFIG.provider, "model": "(discovered from llama-server)",
                "message": "Set provider=llamacpp. Start llama-server, then refresh the model list in the chat box."}

    raise LocalModelError(
        "Hugging Face snapshots are weights, not an API endpoint. Train or serve them with "
        "vLLM (`vllm serve %s`), then choose provider 'custom' with base_url http://127.0.0.1:8000/v1."
        % plan["ref"]
    )


print(
    "Local models ready: %d methods, %d catalogue entries (%s)"
    % (len(LOCAL_METHODS), len(LOCAL_CATALOG), ", ".join(LOCAL_METHODS))
)

# =============================================================================
# Notebook cell 43
# =============================================================================
# 16 - Local model picker: method first, then the model.
#
# The model dropdown stays disabled until a download method is chosen, because
# the same model has a different reference, size and command per method.

_LOCAL_STATE = {"method": "", "model": "", "plan": None, "last_progress": 0.0}

if IN_COLAB:
    _LOCAL_PROGRESS = display(
        HTML('<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
             'font-size:12px;color:#6e6e6e;">No download running.</div>'),
        display_id=True,
    )
else:
    _LOCAL_PROGRESS = None


def _progress_html(done, total):
    if total:
        pct = 100.0 * done / float(total)
        return ('<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;font-size:12px;'
                'color:#1a7f37;">Downloading: %s / %s (%.1f%%)</div>'
                % (human_bytes(done), human_bytes(total), pct))
    return ('<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;font-size:12px;'
            'color:#1a7f37;">Downloading: %s</div>' % human_bytes(done))


def _progress_cb(done, total):
    """Throttled progress: at most two UI updates per second."""
    now = time.time()
    finished = total is not None and done >= total
    if _LOCAL_PROGRESS is not None and (finished or now - _LOCAL_STATE["last_progress"] > 0.5):
        _LOCAL_PROGRESS.update(HTML(_progress_html(done, total)))
        _LOCAL_STATE["last_progress"] = now


def _local_methods_cb():
    return {"ok": True, "methods": local_methods(), "error": ""}


def _local_models_cb(method_id):
    try:
        return {"ok": True, "models": local_models(method_id), "error": ""}
    except LocalModelError as exc:
        return {"ok": False, "models": [], "error": str(exc)}


def _local_prepare_cb(model_id, method_id, dest_dir=""):
    try:
        plan = local_plan(model_id, method_id, (dest_dir or "").strip() or None)
    except LocalModelError as exc:
        return {"ok": False, "plan": None, "error": str(exc)}
    _LOCAL_STATE.update({"method": method_id, "model": model_id, "plan": plan})
    return {"ok": True, "plan": plan, "error": ""}


def _local_download_cb(model_id, method_id, dest_dir=""):
    try:
        plan = local_plan(model_id, method_id, (dest_dir or "").strip() or None)
        if not plan.get("downloadable"):
            return {"ok": False, "result": None,
                    "error": "Method '%s' does not download here. Run the command it shows." % method_id}
        result = download_gguf(plan, progress=_progress_cb)
    except LocalModelError as exc:
        return {"ok": False, "result": None, "error": str(exc)}
    if _LOCAL_PROGRESS is not None:
        _LOCAL_PROGRESS.update(HTML(
            '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;font-size:12px;'
            'color:#1a7f37;">%s: %s (%s)</div>'
            % (result["status"], result["path"], human_bytes(result["bytes"]))
        ))
    return {"ok": True, "result": result, "error": ""}


def _local_apply_cb(model_id, method_id):
    try:
        out = apply_local_model(model_id, method_id)
    except LocalModelError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "provider": out["provider"], "model": out["model"],
            "message": out["message"], "error": ""}


if IN_COLAB:
    colab_output.register_callback("dagent.local_methods", _local_methods_cb)
    colab_output.register_callback("dagent.local_models", _local_models_cb)
    colab_output.register_callback("dagent.local_prepare", _local_prepare_cb)
    colab_output.register_callback("dagent.local_download", _local_download_cb)
    colab_output.register_callback("dagent.local_apply", _local_apply_cb)


_LOCAL_UI = """
<div id="dl-root" style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:780px;
     border:1px solid #ddd;border-radius:12px;padding:16px;background:#fafafa;">
  <div style="font-weight:700;font-size:15px;margin-bottom:4px;">Local models</div>
  <div style="font-size:12px;color:#6e6e6e;margin-bottom:12px;">
    Pick how the model should arrive. Nothing is downloaded until you press the button.
  </div>

  <div style="font-size:12px;font-weight:600;color:#444;margin-bottom:6px;">1 &middot; Download method</div>
  <div id="dl-methods" style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px;"></div>
  <div id="dl-method-note" style="font-size:12px;color:#6e6e6e;margin-bottom:14px;min-height:32px;"></div>

  <div id="dl-step2" style="opacity:.45;">
    <div style="font-size:12px;font-weight:600;color:#444;margin-bottom:6px;">2 &middot; Model</div>
    <select id="dl-model" disabled style="width:100%;padding:6px;border-radius:6px;
      border:1px solid #ccc;margin-bottom:4px;"></select>
    <div id="dl-model-note" style="font-size:12px;color:#6e6e6e;margin-bottom:10px;min-height:16px;">
      Choose a download method first.
    </div>

    <div style="font-size:12px;font-weight:600;color:#444;margin-bottom:6px;">
      3 &middot; Destination folder <span style="font-weight:400;color:#888;">(blank = workspace/models)</span>
    </div>
    <input id="dl-dest" type="text" placeholder="leave blank for the default"
      style="width:100%;box-sizing:border-box;padding:6px;border-radius:6px;border:1px solid #ccc;
      margin-bottom:10px;font-size:12px;">

    <div id="dl-plan" style="display:none;font-size:12px;background:#fff;border:1px solid #e5e5e5;
      border-radius:8px;padding:10px;margin-bottom:10px;"></div>

    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <button id="dl-download" disabled style="background:#0a58ff;color:#fff;border:none;
        padding:8px 16px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;">
        Download</button>
      <button id="dl-apply" disabled style="background:#fff;color:#111827;border:1px solid #ccc;
        padding:8px 16px;border-radius:8px;font-size:13px;cursor:pointer;">
        Use this model</button>
    </div>
    <div id="dl-status" style="font-size:12px;color:#6e6e6e;margin-top:8px;min-height:16px;"></div>
  </div>
</div>
<script>
(function () {
  var methodsEl = document.getElementById("dl-methods");
  var methodNote = document.getElementById("dl-method-note");
  var step2 = document.getElementById("dl-step2");
  var modelSel = document.getElementById("dl-model");
  var modelNote = document.getElementById("dl-model-note");
  var destInput = document.getElementById("dl-dest");
  var planBox = document.getElementById("dl-plan");
  var downloadBtn = document.getElementById("dl-download");
  var applyBtn = document.getElementById("dl-apply");
  var statusEl = document.getElementById("dl-status");

  var method = "";
  var plan = null;

  function call(name, args) {
    return google.colab.kernel.invokeFunction(name, args, {})
      .then(function (res) { return (res && res.data) ? res.data : res; });
  }

  function setStatus(text, isError) {
    statusEl.textContent = text || "";
    statusEl.style.color = isError ? "#cf222e" : "#6e6e6e";
  }

  function setStep2Enabled(on) {
    step2.style.opacity = on ? "1" : ".45";
    modelSel.disabled = !on;
    downloadBtn.disabled = !on || !plan;
    applyBtn.disabled = !on;
  }

  function renderPlan() {
    planBox.innerHTML = "";
    if (!plan) { planBox.style.display = "none"; return; }
    planBox.style.display = "block";

    function row(label, value, mono) {
      if (!value) { return; }
      var div = document.createElement("div");
      div.style.marginBottom = "4px";
      var b = document.createElement("strong");
      b.textContent = label + ": ";
      div.appendChild(b);
      var span = document.createElement("span");
      span.textContent = value;
      if (mono) { span.style.fontFamily = "ui-monospace,Consolas,monospace"; }
      div.appendChild(span);
      planBox.appendChild(div);
    }

    row("Reference", plan.ref, true);
    row("Size", plan.size_text || "");
    row("Licence", plan.license || "");
    row("Command", plan.command, true);
    row("Note", plan.note || "");

    if (plan.downloadable) {
      downloadBtn.textContent = "Download " + (plan.size_text || "");
    } else {
      downloadBtn.textContent = "Run the command above";
    }
    downloadBtn.disabled = !plan.downloadable;
  }

  function loadModels() {
    modelSel.innerHTML = "";
    var loading = document.createElement("option");
    loading.value = "";
    loading.textContent = "Loading\\u2026";
    modelSel.appendChild(loading);
    setStatus("Fetching models for " + method + "\\u2026", false);
    call("dagent.local_models", [method]).then(function (data) {
      modelSel.innerHTML = "";
      plan = null;
      renderPlan();
      if (!data || !data.ok) {
        var bad = document.createElement("option");
        bad.value = "";
        bad.textContent = "(unavailable)";
        modelSel.appendChild(bad);
        modelNote.textContent = (data && data.error) || "Could not load models.";
        setStatus("", true);
        return;
      }
      data.models.forEach(function (m) {
        var opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.label + "  \\u00b7  " + m.params + "  \\u00b7  " + (m.size_text || "");
        modelSel.appendChild(opt);
      });
      modelNote.textContent = data.models.length + " model(s). Pick one to see the exact command.";
      setStatus("", false);
      loadPlan();
    }).catch(function (err) {
      modelNote.textContent = "Could not load models: " + err;
    });
  }

  function loadPlan() {
    var id = modelSel.value;
    if (!id) { return; }
    call("dagent.local_prepare", [id, method, destInput.value]).then(function (data) {
      if (!data || !data.ok) {
        plan = null;
        renderPlan();
        setStatus((data && data.error) || "Could not build a plan.", true);
        return;
      }
      plan = data.plan;
      renderPlan();
      setStatus(plan.downloadable ? "Ready to download." : "This method needs a command, not a download.", false);
      downloadBtn.disabled = !plan.downloadable;
    }).catch(function (err) { setStatus("Prepare failed: " + err, true); });
  }

  function chooseMethod(id, button) {
    method = id;
    var all = methodsEl.querySelectorAll("button");
    for (var i = 0; i < all.length; i++) {
      all[i].style.borderColor = all[i] === button ? "#0a58ff" : "#ccc";
      all[i].style.background = all[i] === button ? "#eef4ff" : "#fff";
      all[i].style.fontWeight = all[i] === button ? "600" : "400";
    }
    methodNote.textContent = button.dataset.note || "";
    setStep2Enabled(true);
    modelNote.textContent = "Pick a model.";
    loadModels();
  }

  downloadBtn.onclick = function () {
    if (!plan || !plan.downloadable) {
      setStatus("Run the command shown above; this method does not download here.", false);
      return;
    }
    downloadBtn.disabled = true;
    setStatus("Downloading\\u2026 progress is printed under this cell. Keep this tab open.", false);
    call("dagent.local_download", [modelSel.value, method, destInput.value]).then(function (data) {
      downloadBtn.disabled = false;
      if (!data || !data.ok) {
        setStatus((data && data.error) || "Download failed.", true);
        return;
      }
      setStatus(data.result.status + ": " + data.result.path, false);
    }).catch(function (err) {
      downloadBtn.disabled = false;
      setStatus("Download call failed: " + err, true);
    });
  };

  applyBtn.onclick = function () {
    if (!modelSel.value) { return; }
    call("dagent.local_apply", [modelSel.value, method]).then(function (data) {
      if (!data || !data.ok) {
        setStatus((data && data.error) || "Could not apply.", true);
        return;
      }
      setStatus(data.message + " Re-run section 13 to refresh the chat dropdowns.", false);
    }).catch(function (err) { setStatus("Apply failed: " + err, true); });
  };

  modelSel.onchange = loadPlan;
  destInput.onchange = loadPlan;

  setStep2Enabled(false);
  call("dagent.local_methods", []).then(function (data) {
    if (!data || !data.ok) { methodNote.textContent = "Could not load methods."; return; }
    data.methods.forEach(function (m) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = m.label;
      b.dataset.note = m.summary + "  Needs: " + m.needs;
      b.style.cssText = "background:#fff;border:1px solid #ccc;border-radius:8px;" +
        "padding:7px 12px;font-size:13px;cursor:pointer;";
      b.onclick = function () { chooseMethod(m.id, b); };
      methodsEl.appendChild(b);
    });
  });
})();
</script>
"""

def render_local_models():
    """Show the local model picker. Safe to call more than once."""
    display(HTML(_LOCAL_UI))
    return _LOCAL_UI

# =============================================================================
# Notebook cell 44
# =============================================================================
# 17 - Training data: capture every turn, redact secrets, export for fine-tuning.
#
# Nothing leaves the machine: records are appended as JSONL under
# <WORKSPACE>/training_data, which is Google Drive when Drive is mounted.

import glob
import hashlib
import random

TRAINING_DIR = os.path.join(WORKSPACE, "training_data")
TRAINING_ENABLED = True          # set False to stop recording
TRAINING_MIN_ANSWER_CHARS = 24   # shorter answers are usually noise

# Anything that looks like a credential is replaced before it can reach disk.
# Order matters: specific prefixes first, the generic `sk-` last, otherwise an
# OpenRouter key would be labelled as an OpenAI key.
_SECRET_PATTERNS = [
    (re.compile(r"sk-or-v1-[A-Za-z0-9_\-]{16,}"), "[REDACTED:openrouter-key]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[REDACTED:github-token]"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "[REDACTED:github-token]"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "[REDACTED:hf-token]"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), "[REDACTED:google-key]"),
    (re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}"), "[REDACTED:nvidia-key]"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "[REDACTED:openai-key]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)\b(api[_-]?key|token|secret)\b\s*[=:]\s*[\"']?[A-Za-z0-9._\-]{12,}"),
     r"\1=[REDACTED]"),
]


def redact(text):
    """Remove anything that looks like a credential. Safe on None."""
    if not text:
        return ""
    out = str(text)
    for pattern, replacement in _SECRET_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def _turn_id(task, answer):
    digest = hashlib.sha1(("%s\x00%s" % (task, answer)).encode("utf-8", "replace"))
    return "t-" + digest.hexdigest()[:12]


def _training_path(directory=None, when=None):
    directory = directory or TRAINING_DIR
    day = (when or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    return os.path.join(directory, "turns-%s.jsonl" % day)


def capture_turn(task, events, directory=None, extra=None):
    """Persist one completed turn. Returns the record, or None when disabled.

    `events` is the list of dicts yielded by DAgent.run().
    """
    if not TRAINING_ENABLED:
        return None
    if not isinstance(events, list):
        events = list(events)

    answer = ""
    for event in events:
        if event.get("type") == "final":
            answer = event.get("content") or ""
    tools = [e.get("tool", "") for e in events if e.get("type") == "tool_call"]
    errors = [e.get("message", "") for e in events if e.get("type") == "error"]

    record = {
        "v": 1,
        "id": _turn_id(task or "", answer),
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": getattr(CONFIG, "agent_id", ""),
        "task": redact(task),
        "answer": redact(answer),
        "steps": len(tools),
        "tools": tools,
        "provider": getattr(CONFIG, "provider", ""),
        "model": getattr(CONFIG, "model", ""),
        "mode": getattr(CONFIG, "mode", ""),
        "ok": not errors and bool(answer.strip()),
        "error": redact("; ".join(errors)),
        "n_events": len(events),
    }
    if extra:
        for key, value in extra.items():
            if key not in record:
                record[key] = redact(value) if isinstance(value, str) else value

    path = _training_path(directory)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return record


def list_turns(directory=None, path=None):
    """Read every recorded turn. Malformed lines are counted, never silently dropped."""
    paths = [path] if path else sorted(
        glob.glob(os.path.join(directory or TRAINING_DIR, "*.jsonl"))
    )
    records, bad = [], 0
    for file_path in paths:
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    bad += 1
    if bad:
        print("warning: skipped %d malformed line(s)" % bad)
    return records


def dataset_stats(records=None):
    """Counts for the UI: how much data exists and how good it is."""
    records = list_turns() if records is None else records
    if not records:
        return {"total": 0, "usable": 0, "with_tools": 0, "avg_answer_chars": 0,
                "providers": {}, "days": 0}
    usable = [r for r in records if r.get("ok") and len(r.get("answer", "")) >= TRAINING_MIN_ANSWER_CHARS]
    providers = {}
    for r in records:
        key = r.get("provider") or "unknown"
        providers[key] = providers.get(key, 0) + 1
    return {
        "total": len(records),
        "usable": len(usable),
        "with_tools": sum(1 for r in records if r.get("tools")),
        "avg_answer_chars": int(sum(len(r.get("answer", "")) for r in records) / len(records)),
        "providers": providers,
        "days": len({r.get("ts", "")[:10] for r in records}),
    }


def dedupe(records):
    """Keep the first occurrence of each (task, answer) pair."""
    seen, out = set(), []
    for record in records:
        key = (record.get("task", "").strip(), record.get("answer", "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def sft_example(record, system=None):
    """One OpenAI-style fine-tuning row."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": record.get("task", "")})
    messages.append({"role": "assistant", "content": record.get("answer", "")})
    return {"messages": messages}


def export_sft(dest=None, records=None, drop_errors=True, deduplicate=True,
               min_answer_chars=None, system=None):
    """Write a JSONL fine-tuning file and return where it went and how many rows."""
    records = list_turns() if records is None else records
    minimum = TRAINING_MIN_ANSWER_CHARS if min_answer_chars is None else min_answer_chars

    rows = []
    for record in records:
        if drop_errors and not record.get("ok"):
            continue
        if len(record.get("answer", "")) < minimum:
            continue
        if not record.get("task", "").strip():
            continue
        rows.append(record)
    if deduplicate:
        rows = dedupe(rows)

    dest = dest or os.path.join(TRAINING_DIR, "sft-%s.jsonl" % datetime.now(timezone.utc).strftime("%Y%m%d"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        for record in rows:
            fh.write(json.dumps(sft_example(record, system=system), ensure_ascii=False) + "\n")

    return {"path": dest, "rows": len(rows), "skipped": len(records) - len(rows)}


def split_dataset(records, val_ratio=0.1, seed=0):
    """Deterministic train/validation split."""
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1), got %r" % (val_ratio,))
    items = list(records)
    random.Random(seed).shuffle(items)
    cut = int(round(len(items) * val_ratio))
    return items[cut:], items[:cut]


print("Training data ready: %s (enabled=%s)" % (TRAINING_DIR, TRAINING_ENABLED))

# =============================================================================
# Notebook cell 45
# =============================================================================
# 18 - Personal knowledge: a small TF-IDF index over your own files.
#
# This is Tier 1 of the retraining tunnel. No GPU, no training: your documents
# are retrieved at question time and pasted into the prompt. It works the same
# on day one and after a thousand files.

import math

RAG_TEXT_EXTENSIONS = {
    ".md", ".txt", ".rst", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".csv", ".html", ".ipynb", ".sql", ".sh",
}
RAG_MAX_FILE_BYTES = 2 * 1024 * 1024
RAG_CHUNK_CHARS = 900
RAG_CHUNK_OVERLAP = 120


def tokenize(text):
    """Lowercase word tokens. No stemming: predictable beats clever here."""
    return [w for w in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(w) > 1]


def chunk_text(text, size=RAG_CHUNK_CHARS, overlap=RAG_CHUNK_OVERLAP):
    """Split on paragraph boundaries, then hard-wrap long paragraphs."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    text = (text or "").strip()
    if not text:
        return []

    paragraphs, current = [], ""
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if len(block) <= size:
            if current and len(current) + len(block) + 2 <= size:
                current = current + "\n\n" + block
            else:
                if current:
                    paragraphs.append(current)
                current = block
        else:
            if current:
                paragraphs.append(current)
                current = ""
            step = size - overlap
            for start in range(0, len(block), step):
                piece = block[start:start + size]
                if piece.strip():
                    paragraphs.append(piece)
                if start + size >= len(block):
                    break
    if current:
        paragraphs.append(current)
    return paragraphs


def _iter_documents(folder, extensions=None, max_bytes=RAG_MAX_FILE_BYTES):
    """Yield (path, text) for readable text files under `folder`."""
    allowed = extensions or RAG_TEXT_EXTENSIONS
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError("Folder not found: %s" % folder)
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in sorted(dirs) if not d.startswith(".")]
        for name in sorted(files):
            path = os.path.join(root, name)
            if os.path.splitext(name)[1].lower() not in allowed:
                continue
            try:
                if os.path.getsize(path) > max_bytes:
                    continue
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if text.strip():
                yield path, text


def build_index(folder, extensions=None):
    """Build a TF-IDF index. Returns a plain dict, so it can be saved as JSON."""
    chunks, df, total_tokens = [], {}, 0
    for path, text in _iter_documents(folder, extensions):
        rel = os.path.relpath(path, folder).replace("\\", "/")
        for position, piece in enumerate(chunk_text(text)):
            tokens = tokenize(piece)
            if not tokens:
                continue
            counts = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token in counts:
                df[token] = df.get(token, 0) + 1
            total_tokens += len(tokens)
            chunks.append({
                "id": "%s#%d" % (rel, position),
                "source": rel,
                "text": piece,
                "len": len(tokens),
                "tf": counts,
            })

    if not chunks:
        raise ValueError("No indexable text found under %s" % folder)

    n = len(chunks)
    idf = {token: math.log((n + 1.0) / (count + 1.0)) + 1.0 for token, count in df.items()}
    return {
        "version": 1,
        "root": os.path.abspath(folder),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_chunks": n,
        "n_tokens": total_tokens,
        "idf": idf,
        "chunks": chunks,
    }


def save_index(index, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False)
    return path


def load_index(path):
    with open(path, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    if not isinstance(index, dict) or "chunks" not in index:
        raise ValueError("%s is not a DAgent index" % path)
    return index


def _vectorize(tokens, idf):
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    if not counts:
        return {}
    length = float(len(tokens))
    return {
        token: (count / length) * idf.get(token, 1.0)
        for token, count in counts.items()
    }


def _cosine(a, b):
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search(index, query, k=4, min_score=0.01):
    """Top-k chunks for a query. Returns [] when nothing is relevant enough."""
    if not index or not index.get("chunks"):
        return []
    idf = index.get("idf", {})
    query_vector = _vectorize(tokenize(query), idf)
    if not query_vector:
        return []

    scored = []
    for chunk in index["chunks"]:
        score = _cosine(query_vector, _vectorize_from_counts(chunk, idf))
        if score >= min_score:
            scored.append((score, chunk))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [
        {"score": round(score, 4), "id": chunk["id"], "source": chunk["source"],
         "text": chunk["text"]}
        for score, chunk in scored[:max(1, int(k))]
    ]


def _vectorize_from_counts(chunk, idf):
    counts = chunk.get("tf", {})
    length = float(chunk.get("len") or sum(counts.values()) or 1)
    return {token: (count / length) * idf.get(token, 1.0) for token, count in counts.items()}


def context_block(hits, max_chars=4000, header="Relevant notes from the user's own files"):
    """Format hits for prompt injection. Empty string when there is nothing."""
    if not hits:
        return ""
    parts, used = [], 0
    for hit in hits:
        block = "[%s]\n%s" % (hit["source"], hit["text"].strip())
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 200:
                parts.append(block[:remaining])
            break
        parts.append(block)
        used += len(block) + 2
    if not parts:
        return ""
    return "%s:\n\n%s" % (header, "\n\n".join(parts))


print("Personal knowledge ready: %d extensions, chunk=%d chars" % (len(RAG_TEXT_EXTENSIONS), RAG_CHUNK_CHARS))

# =============================================================================
# Notebook cell 46
# =============================================================================
# 19 - Retraining tunnel: three tiers, only one of which needs a GPU.
#
# Tier 1 retrieval and tier 2 style prompting work today with no training.
# Tier 3 (LoRA) is generated here as config + script, and is explicitly
# marked unverified: it needs a GPU and a measured before/after comparison,
# which is what evaluate_answers()/compare_answers() are for.

TUNNEL_MIN_TURNS_TIER2 = 20
TUNNEL_MIN_TURNS_TIER3 = 200


def gpu_status():
    """Report what the runtime actually offers. Never assumes a GPU."""
    try:
        import torch
    except Exception:
        return {"torch": False, "cuda": False, "device": "none",
                "reason": "torch is not installed"}
    try:
        available = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if available else "cpu"
    except Exception as exc:
        return {"torch": True, "cuda": False, "device": "unknown",
                "reason": "%s: %s" % (type(exc).__name__, exc)}
    return {"torch": True, "cuda": available, "device": name,
            "reason": "" if available else "no CUDA device visible"}


def tunnel_plan(records=None):
    """What can run now, what is blocked, and by exactly what."""
    records = list_turns() if records is None else records
    stats = dataset_stats(records)
    gpu = gpu_status()

    tier1 = {
        "tier": 1,
        "id": "retrieval",
        "label": "Retrieval over your files",
        "needs": "a folder of notes, docs or code",
        "training": False,
        "available": True,
        "blocker": "",
        "how": "build_index(folder) then search(index, question) and pass context_block(hits) to the model.",
    }
    tier2 = {
        "tier": 2,
        "id": "style",
        "label": "Style pack (few-shot from your own turns)",
        "needs": ">= %d usable captured turns" % TUNNEL_MIN_TURNS_TIER2,
        "training": False,
        "available": stats["usable"] >= TUNNEL_MIN_TURNS_TIER2,
        "blocker": "" if stats["usable"] >= TUNNEL_MIN_TURNS_TIER2
        else "%d/%d usable turns captured so far" % (stats["usable"], TUNNEL_MIN_TURNS_TIER2),
        "how": "build_style_pack(records) then prepend it to the system prompt.",
    }
    tier3 = {
        "tier": 3,
        "id": "lora",
        "label": "LoRA fine-tune",
        "needs": ">= %d usable turns and a CUDA GPU" % TUNNEL_MIN_TURNS_TIER3,
        "training": True,
        "available": stats["usable"] >= TUNNEL_MIN_TURNS_TIER3 and gpu["cuda"],
        "blocker": ", ".join(filter(None, [
            "" if stats["usable"] >= TUNNEL_MIN_TURNS_TIER3
            else "%d/%d usable turns" % (stats["usable"], TUNNEL_MIN_TURNS_TIER3),
            "" if gpu["cuda"] else gpu["reason"],
        ])),
        "how": "export_sft() -> write_lora_config() -> lora_train_script(); compare before/after with compare_answers().",
    }
    return {"tiers": [tier1, tier2, tier3], "stats": stats, "gpu": gpu}


def build_style_pack(records, n=6, max_chars=3000, min_answer_chars=40):
    """Pick a few of your own best answers as few-shot examples (tier 2)."""
    candidates = [
        r for r in records
        if r.get("ok") and r.get("task", "").strip()
        and len(r.get("answer", "")) >= min_answer_chars
    ]
    candidates.sort(key=lambda r: (-len(r.get("answer", "")), r.get("id", "")))
    chosen = candidates[:max(1, int(n))]

    blocks, used = [], 0
    for record in chosen:
        task = record["task"].strip()
        answer = record["answer"].strip()
        budget = max(200, (max_chars - used) // max(1, len(chosen)))
        if len(answer) > budget:
            answer = answer[:budget].rstrip() + "..."
        block = "User: %s\nAssistant: %s" % (task, answer)
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)


def style_pack_prompt(pack):
    """Wrap a style pack as an addition to the system prompt."""
    if not pack.strip():
        return ""
    return (
        "Match the tone, length and structure of these earlier answers by the same "
        "assistant. Do not copy their content, only the style.\n\n" + pack.strip()
    )


def write_lora_config(path, r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=None):
    """Write a PEFT LoraConfig as JSON. Does not train anything."""
    config = {
        "r": int(r),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": list(target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    return config


LORA_SCRIPT = '''"""LoRA fine-tune on the exported SFT file.

UNVERIFIED: this script was generated, not executed. It needs a CUDA GPU and
current versions of transformers, peft, trl and datasets. Read it before you
run it, and compare results with compare_answers() before trusting them.

    python lora_train.py --train_file sft.jsonl --output_dir ./lora-out
"""
import argparse
import json
import pathlib

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--config", default="lora_config.json")
    ap.add_argument("--epochs", type=float, default=1.0)
    args = ap.parse_args()

    config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    dataset = load_dataset("json", data_files=args.train_file, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(output_dir=args.output_dir, num_train_epochs=args.epochs,
                       per_device_train_batch_size=1, gradient_accumulation_steps=8,
                       logging_steps=5, save_strategy="epoch"),
        peft_config=LoraConfig(**config),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("saved adapter to", args.output_dir)


if __name__ == "__main__":
    main()
'''


def lora_train_script(path):
    """Write the (unverified) training script and return its path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(LORA_SCRIPT)
    return path


def rouge_l(reference, candidate):
    """ROUGE-L F1 over word tokens. Pure Python, deterministic."""
    ref = tokenize(reference)
    cand = tokenize(candidate)
    if not ref or not cand:
        return 0.0
    # Longest common subsequence, rolling row to keep memory flat.
    previous = [0] * (len(cand) + 1)
    for token in ref:
        current = [0]
        for j, other in enumerate(cand, start=1):
            if token == other:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / float(len(cand))
    recall = lcs / float(len(ref))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_answers(pairs, predict, limit=None):
    """Score a predictor against held-out (task, answer) pairs.

    `predict` is any callable: task -> answer string.
    """
    items = list(pairs)[:limit] if limit else list(pairs)
    if not items:
        return {"n": 0, "mean_rouge_l": 0.0, "exact": 0, "worst": []}

    scores, worst = [], []
    for pair in items:
        task = pair.get("task", "")
        expected = pair.get("answer", "")
        try:
            got = predict(task)
        except Exception as exc:
            got = ""
            worst.append({"task": task[:120], "score": 0.0,
                          "error": "%s: %s" % (type(exc).__name__, exc)})
            scores.append(0.0)
            continue
        score = rouge_l(expected, got)
        scores.append(score)
        worst.append({"task": task[:120], "score": round(score, 3), "got": (got or "")[:200]})

    worst.sort(key=lambda row: row["score"])
    return {
        "n": len(scores),
        "mean_rouge_l": round(sum(scores) / len(scores), 4),
        "exact": sum(1 for s in scores if s >= 0.999),
        "worst": worst[:5],
    }


def compare_answers(pairs, predict_before, predict_after, limit=None):
    """Before/after on the same pairs. Positive delta means it improved."""
    before = evaluate_answers(pairs, predict_before, limit=limit)
    after = evaluate_answers(pairs, predict_after, limit=limit)
    return {
        "before": before,
        "after": after,
        "delta_rouge_l": round(after["mean_rouge_l"] - before["mean_rouge_l"], 4),
        "verdict": "better" if after["mean_rouge_l"] > before["mean_rouge_l"]
        else ("worse" if after["mean_rouge_l"] < before["mean_rouge_l"] else "unchanged"),
    }


def holdout_pairs(records, n=20, seed=0):
    """Deterministic held-out (task, answer) pairs for evaluation."""
    usable = [r for r in records if r.get("ok") and r.get("task") and r.get("answer")]
    items = list(usable)
    random.Random(seed).shuffle(items)
    return [{"task": r["task"], "answer": r["answer"]} for r in items[:max(1, int(n))]]


# tunnel_plan() is called by the UI, not at import time.


# =============================================================================
# Application entry point
# =============================================================================

def ensure_dependencies():
    """Install the runtime dependencies when running in Colab. No-op elsewhere."""
    if not IN_COLAB:
        return {"installed": False, "reason": "not running in Colab"}
    import importlib
    import subprocess
    import sys

    missing = []
    for module, package in (("openai", "openai"), ("requests", "requests"),
                            ("bs4", "beautifulsoup4"), ("nest_asyncio", "nest_asyncio")):
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if not missing:
        return {"installed": False, "reason": "already present", "missing": []}
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + missing,
                   check=False)
    return {"installed": True, "missing": missing}


def render_status():
    """A compact panel: how much data exists and what the tunnel can do now."""
    try:
        stats = dataset_stats()
        plan = tunnel_plan()
    except Exception as exc:  # noqa: BLE001 - shown, never swallowed
        display(HTML('<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
                     'font-size:12px;color:#cf222e;">Status unavailable: %s</div>'
                     % html.escape("%s: %s" % (type(exc).__name__, exc))))
        return None

    rows = [
        ("Captured turns", "%d (%d usable)" % (stats["total"], stats["usable"])),
        ("Turns with tools", str(stats["with_tools"])),
        ("Days recorded", str(stats["days"])),
    ]
    tier_rows = "".join(
        "<tr><td style='padding:2px 10px 2px 0;'>Tier %d &middot; %s</td>"
        "<td style='color:%s;'>%s</td></tr>"
        % (tier["tier"], html.escape(tier["label"]),
           "#1a7f37" if tier["available"] else "#9a6700",
           "ready" if tier["available"] else html.escape(tier["blocker"] or "blocked"))
        for tier in plan["tiers"]
    )
    body = "".join("<tr><td style='padding:2px 10px 2px 0;'>%s</td><td>%s</td></tr>"
                   % (html.escape(k), html.escape(v)) for k, v in rows)
    gpu = plan["gpu"]
    panel = (
        '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:780px;'
        'border:1px solid #ddd;border-radius:12px;padding:14px;background:#fafafa;margin-top:12px;">'
        '<div style="font-weight:700;font-size:14px;margin-bottom:8px;">Your data</div>'
        '<table style="font-size:12px;color:#333;border-collapse:collapse;">%s</table>'
        '<div style="font-weight:600;font-size:12px;margin:10px 0 4px;">Retraining tunnel</div>'
        '<table style="font-size:12px;color:#333;border-collapse:collapse;">%s</table>'
        '<div style="font-size:11px;color:#6e6e6e;margin-top:8px;">GPU: %s</div>'
        '<div style="font-size:11px;color:#6e6e6e;">Training data: %s</div>'
        '</div>'
        % (body, tier_rows, html.escape(gpu["device"]), html.escape(TRAINING_DIR))
    )
    display(HTML(panel))
    return panel


def render_header():
    provider = getattr(CONFIG, "provider", "")
    model = getattr(CONFIG, "model", "") or "(provider default)"
    display(HTML(
        '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:780px;'
        'margin:0 0 12px;">'
        '<div style="font-size:20px;font-weight:700;">DAgent</div>'
        '<div style="font-size:12px;color:#6e6e6e;">provider: %s &middot; model: %s &middot; '
        'mode: %s</div></div>'
        % (html.escape(provider), html.escape(model), html.escape(str(CONFIG.mode)))
    ))


def main():
    """Start the application. This is the only thing a user has to run."""
    deps = ensure_dependencies()
    if deps.get("installed"):
        print("installed:", ", ".join(deps["missing"]))
    render_presentation()
    render_header()
    render_chat()
    render_local_models()
    render_status()
    print("DAgent is ready. Use the boxes above; no other cell needs to run.")


if __name__ == "__main__":
    main()
