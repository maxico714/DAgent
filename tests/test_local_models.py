"""Offline checks for the notebook's local-model cell (section 15).

Extracts cell 42 and runs it against a real HTTP server on 127.0.0.1, so the
downloader is exercised end to end: fresh download, resume with a Range
request, a server that ignores Range, a 416 on an already-complete part file,
and a truncated response. No internet, no real model files.

Run:  python tests/test_local_models.py
"""
import http.server
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import threading
import types

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELL_INDEX = 42

NB = json.loads((ROOT / "notebooks" / "DAgent-dev.ipynb").read_text(encoding="utf-8"))
SRC = "".join(NB["cells"][CELL_INDEX]["source"])

PAYLOAD = bytes(range(256)) * 1024  # 262,144 deterministic bytes


class _Handler(http.server.BaseHTTPRequestHandler):
    payload = PAYLOAD
    honor_range = True
    truncate_to = None
    seen = []

    def do_GET(self):  # noqa: N802 - stdlib naming
        type(self).seen.append({"path": self.path, "range": self.headers.get("Range")})
        data = type(self).payload
        if type(self).truncate_to is not None:
            data = data[: type(self).truncate_to]
        rng = self.headers.get("Range")
        if type(self).honor_range and rng and rng.startswith("bytes="):
            start = int(rng.split("=", 1)[1].split("-", 1)[0])
            if start >= len(data):
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % len(data))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = data[start:]
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, len(data) - 1, len(data)))
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Server:
    def __init__(self, **attrs):
        handler = type("H", (_Handler,), dict(seen=[], **attrs))
        self.handler = handler
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return "http://127.0.0.1:%d/model.gguf" % self.port

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def load_cell(workspace):
    ns = {
        "__name__": "local_cell",
        "os": os,
        "re": re,
        "requests": requests,
        "Path": pathlib.Path,
        "WORKSPACE": workspace,
        "CONFIG": types.SimpleNamespace(provider="openrouter", model="x", base_url=""),
    }
    exec(compile(SRC, "DAgent-dev.ipynb:cell%d" % CELL_INDEX, "exec"), ns)
    return ns


TMP = tempfile.mkdtemp(prefix="dagent-local-")
L = load_cell(TMP)


def make_plan(server, expected=len(PAYLOAD), name="model.gguf"):
    dest = os.path.join(TMP, "models", name)
    return {
        "method": "gguf",
        "url": server.url,
        "dest": dest,
        "bytes": expected,
        "filename": name,
    }


def test_methods_are_method_first():
    rows = L["local_methods"]()
    assert [r["id"] for r in rows] == ["ollama", "gguf", "hf"], rows
    assert all(r["summary"] and r["needs"] and r["setup"] for r in rows)
    assert [r["serves_api"] for r in rows] == [True, True, False]


def test_catalog_is_internally_consistent():
    entries = L["LOCAL_CATALOG"]
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), ids
    assert len(entries) >= 5
    for e in entries:
        assert e["gguf"]["file"].endswith(".gguf"), e
        assert "/" in e["gguf"]["repo"], e
        assert e["gguf"]["bytes"] > 100 * 1024 * 1024, e
        assert e["license"], e
        assert e["ollama"]["library"] in L["OLLAMA_LIBRARY"], e
        assert e["ollama"]["tag"], e
        assert "/" in e["hf"]["repo"], e


def test_models_for_each_method_carry_a_real_reference():
    gguf_rows = L["local_models"]("gguf")
    assert [r["size_bytes"] for r in gguf_rows] == [e["gguf"]["bytes"] for e in L["LOCAL_CATALOG"]]
    assert all(r["ref"].endswith(".gguf") for r in gguf_rows)

    ollama_rows = L["local_models"]("ollama")
    assert all(":" in r["ref"] for r in ollama_rows)

    hf_rows = L["local_models"]("hf")
    assert all(r["ref"].count("/") == 1 for r in hf_rows)
    assert any("gated" in r["warning"] for r in hf_rows), hf_rows


def test_unknown_method_and_model_are_actionable():
    try:
        L["local_models"]("ftp")
    except L["LocalModelError"] as exc:
        assert "ollama" in str(exc) and "gguf" in str(exc)
    else:
        raise AssertionError("expected LocalModelError")

    try:
        L["local_entry"]("nope")
    except L["LocalModelError"] as exc:
        assert "qwen2.5-1.5b" in str(exc)
    else:
        raise AssertionError("expected LocalModelError")


def test_plan_points_at_huggingface_and_workspace():
    plan = L["local_plan"]("qwen2.5-1.5b", "gguf")
    assert plan["url"].startswith("https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/")
    assert plan["url"].endswith("Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
    assert plan["dest"].startswith(os.path.join(TMP, "models")), plan["dest"]
    assert plan["downloadable"] is True
    assert "llama-server" in plan["command"]

    ollama = L["local_plan"]("qwen2.5-3b", "ollama")
    assert ollama["command"] == "ollama pull qwen2.5:3b"
    assert ollama["downloadable"] is False

    hf = L["local_plan"]("llama3.2-1b", "hf")
    assert hf["gated"] is True and "huggingface-cli login" in hf["command"]
    assert hf["serves_api"] is False


def test_fresh_download_writes_exact_bytes():
    server = Server()
    try:
        plan = make_plan(server)
        samples = []
        result = L["download_gguf"](plan, progress=lambda done, total: samples.append((done, total)))
        assert result["status"] == "downloaded", result
        data = pathlib.Path(plan["dest"]).read_bytes()
        assert data == PAYLOAD
        assert not os.path.exists(plan["dest"] + ".part")
        assert samples and samples[-1][0] == len(PAYLOAD), samples[-3:]
        assert all(t == len(PAYLOAD) for _, t in samples), samples[:2]
        assert server.handler.seen[0]["range"] is None
    finally:
        server.close()


def test_resume_sends_range_and_keeps_the_prefix():
    server = Server()
    try:
        plan = make_plan(server, name="resume.gguf")
        part = plan["dest"] + ".part"
        pathlib.Path(part).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(part).write_bytes(PAYLOAD[:100000])

        result = L["download_gguf"](plan)
        assert result["status"] == "resumed", result
        assert pathlib.Path(plan["dest"]).read_bytes() == PAYLOAD
        assert server.handler.seen[0]["range"] == "bytes=100000-", server.handler.seen
    finally:
        server.close()


def test_server_ignoring_range_does_not_corrupt_the_file():
    server = Server(honor_range=False)
    try:
        plan = make_plan(server, name="norange.gguf")
        part = plan["dest"] + ".part"
        pathlib.Path(part).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(part).write_bytes(PAYLOAD[:100000])

        result = L["download_gguf"](plan)
        assert result["status"] == "downloaded", result
        assert pathlib.Path(plan["dest"]).read_bytes() == PAYLOAD
        assert result["bytes"] == len(PAYLOAD)
    finally:
        server.close()


def test_416_on_complete_part_finalises_it():
    server = Server()
    try:
        plan = make_plan(server, name="complete.gguf")
        part = plan["dest"] + ".part"
        pathlib.Path(part).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(part).write_bytes(PAYLOAD)

        result = L["download_gguf"](plan)
        assert result["status"] == "resumed", result
        assert pathlib.Path(plan["dest"]).read_bytes() == PAYLOAD
        assert server.handler.seen[0]["range"] == "bytes=%d-" % len(PAYLOAD)
    finally:
        server.close()


def test_truncated_response_raises_and_leaves_the_part_file():
    server = Server(truncate_to=50000)
    try:
        plan = make_plan(server, name="short.gguf")
        try:
            L["download_gguf"](plan)
        except L["LocalModelError"] as exc:
            assert "Incomplete" in str(exc) or "mismatch" in str(exc), str(exc)
        else:
            raise AssertionError("expected LocalModelError")
        assert not os.path.exists(plan["dest"])
        assert os.path.exists(plan["dest"] + ".part")
    finally:
        server.close()


def test_existing_complete_file_is_reused():
    server = Server()
    try:
        plan = make_plan(server, name="existing.gguf")
        pathlib.Path(plan["dest"]).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(plan["dest"]).write_bytes(PAYLOAD)
        result = L["download_gguf"](plan)
        assert result["status"] == "already", result
        assert server.handler.seen == [], "no request should be made"
    finally:
        server.close()


def test_non_http_scheme_is_refused():
    plan = {"method": "gguf", "url": "file:///etc/passwd", "dest": os.path.join(TMP, "x.gguf"), "bytes": None}
    try:
        L["download_gguf"](plan)
    except L["LocalModelError"] as exc:
        assert "http" in str(exc)
    else:
        raise AssertionError("expected LocalModelError")


def test_apply_ollama_sets_config_and_hf_is_refused():
    cfg = L["CONFIG"]
    out = L["apply_local_model"]("qwen2.5-3b", "ollama")
    assert cfg.provider == "ollama" and cfg.model == "qwen2.5:3b", out
    assert "ollama serve" in out["message"]

    out = L["apply_local_model"]("qwen2.5-1.5b", "gguf")
    assert cfg.provider == "llamacpp" and cfg.model == "", out
    assert "refresh" in out["message"].lower()

    try:
        L["apply_local_model"]("qwen2.5-1.5b", "hf")
    except L["LocalModelError"] as exc:
        assert "vllm" in str(exc).lower()
    else:
        raise AssertionError("expected LocalModelError")


def test_human_bytes():
    hb = L["human_bytes"]
    assert hb(None) == "unknown"
    assert hb(512) == "512 B"
    assert hb(986048768) == "940.4 MB", hb(986048768)
    assert hb(4683074336) == "4.4 GB", hb(4683074336)


def test_ollama_tags_rejects_unknown_library_without_network():
    try:
        L["fetch_ollama_tags"]("not-a-real-library")
    except L["LocalModelError"] as exc:
        assert "OLLAMA_LIBRARY" in str(exc)
    else:
        raise AssertionError("expected LocalModelError")


if __name__ == "__main__":
    failed = 0
    try:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                try:
                    fn()
                    print("PASS", name)
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print("FAIL", name, "::", exc)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s)" % failed)
    sys.exit(1 if failed else 0)
