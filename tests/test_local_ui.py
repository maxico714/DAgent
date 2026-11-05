"""Offline checks for the local-model picker cell (section 16).

Extracts cells 42 and 43, executes them with stubbed Colab globals, and checks
that the method-first gating and the callbacks behave. No network, no Colab.

Run:  python tests/test_local_ui.py
"""
import json
import os
import pathlib
import shutil
import sys
import tempfile
import types

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGIC_INDEX = 42
UI_INDEX = 43

NB = json.loads((ROOT / "DAgent.ipynb").read_text(encoding="utf-8"))
LOGIC_SRC = "".join(NB["cells"][LOGIC_INDEX]["source"])
UI_SRC = "".join(NB["cells"][UI_INDEX]["source"])

TMP = tempfile.mkdtemp(prefix="dagent-ui-")
REGISTERED = {}


class _Handle:
    def __init__(self):
        self.updates = []

    def update(self, obj):
        self.updates.append(obj)


def build(in_colab=False):
    ns = {
        "__name__": "local_ui",
        "os": os,
        "re": __import__("re"),
        "time": __import__("time"),
        "requests": requests,
        "Path": pathlib.Path,
        "WORKSPACE": TMP,
        "CONFIG": types.SimpleNamespace(provider="openrouter", model="x", base_url=""),
        "IN_COLAB": in_colab,
        "display": lambda obj, display_id=False: _Handle(),
        "HTML": lambda s: s,
        "update_display": lambda obj, display_id=None: None,
        "colab_output": types.SimpleNamespace(
            register_callback=lambda name, fn: REGISTERED.__setitem__(name, fn)
        ),
    }
    exec(compile(LOGIC_SRC, "cell42", "exec"), ns)
    exec(compile(UI_SRC, "cell43", "exec"), ns)
    return ns


U = build(in_colab=False)


def test_methods_callback_lists_three_methods():
    res = U["_local_methods_cb"]()
    assert res["ok"] is True
    assert [m["id"] for m in res["methods"]] == ["ollama", "gguf", "hf"], res


def test_models_callback_is_method_scoped():
    gguf = U["_local_models_cb"]("gguf")
    assert gguf["ok"] is True and len(gguf["models"]) == 7, gguf
    assert all(r["ref"].endswith(".gguf") for r in gguf["models"])

    hf = U["_local_models_cb"]("hf")
    assert all("/" in r["ref"] and not r["ref"].endswith(".gguf") for r in hf["models"])

    bad = U["_local_models_cb"]("ftp")
    assert bad["ok"] is False and "ollama" in bad["error"], bad


def test_prepare_returns_a_plan_and_records_state():
    res = U["_local_prepare_cb"]("qwen2.5-1.5b", "gguf", "")
    assert res["ok"] is True, res
    plan = res["plan"]
    assert plan["downloadable"] is True
    assert plan["url"].endswith(".gguf")
    assert plan["dest"].startswith(os.path.join(TMP, "models"))
    assert U["_LOCAL_STATE"]["model"] == "qwen2.5-1.5b"


def test_prepare_honours_a_custom_destination():
    dest = os.path.join(TMP, "elsewhere")
    res = U["_local_prepare_cb"]("qwen2.5-1.5b", "gguf", dest)
    assert res["plan"]["dest"].startswith(dest), res["plan"]["dest"]


def test_prepare_rejects_unknown_model():
    res = U["_local_prepare_cb"]("nope", "gguf", "")
    assert res["ok"] is False and "Known" in res["error"], res


def test_download_refuses_non_downloadable_methods():
    for method in ("ollama", "hf"):
        res = U["_local_download_cb"]("qwen2.5-1.5b", method, "")
        assert res["ok"] is False, res
        assert "command" in res["error"].lower(), res


def test_apply_callback_sets_provider():
    res = U["_local_apply_cb"]("qwen2.5-3b", "ollama")
    assert res["ok"] is True, res
    assert res["provider"] == "ollama" and res["model"] == "qwen2.5:3b"
    assert U["CONFIG"].provider == "ollama"


def test_apply_callback_reports_hf_refusal():
    res = U["_local_apply_cb"]("qwen2.5-1.5b", "hf")
    assert res["ok"] is False and "vllm" in res["error"].lower(), res


def test_ui_gates_the_model_picker_behind_the_method():
    ui = U["_LOCAL_UI"]
    assert 'id="dl-methods"' in ui
    assert 'id="dl-model"' in ui and "disabled" in ui
    assert "Choose a download method first." in ui
    # the three method buttons are rendered from the callback, not hardcoded
    for token in ("__", "{{", "}}"):
        assert token not in ui or token == "__", token
    assert "dagent.local_methods" in ui
    assert "dagent.local_models" in ui
    assert "dagent.local_prepare" in ui
    assert "dagent.local_download" in ui
    assert "dagent.local_apply" in ui


def test_ui_never_interpolates_values_with_innerHTML():
    ui = U["_LOCAL_UI"]
    assert "innerHTML = m" not in ui
    assert "textContent" in ui


def test_progress_html_formats_percentage():
    html = U["_progress_html"](512, 1024)
    assert "50.0%" in html and "512 B" in html and "1.0 KB" in html, html
    unknown = U["_progress_html"](512, None)
    assert "512 B" in unknown and "%" not in unknown


def test_callbacks_are_registered_in_colab_only():
    REGISTERED.clear()
    build(in_colab=False)
    assert REGISTERED == {}, REGISTERED

    build(in_colab=True)
    assert set(REGISTERED) == {
        "dagent.local_methods",
        "dagent.local_models",
        "dagent.local_prepare",
        "dagent.local_download",
        "dagent.local_apply",
    }, sorted(REGISTERED)


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
