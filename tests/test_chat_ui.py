"""Offline tests for the notebook chat cell (section 13).

Extracts cells 29 and 37 from notebooks/DAgent-dev.ipynb, executes them with stubbed globals,
and checks the provider/model UI wiring. No network, no keys, no Colab.
"""
import asyncio
import json
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]

if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, base_url=None, api_key=None, default_headers=None):
            self.base_url = base_url
            self.api_key = api_key
            self.default_headers = default_headers or {}

    fake_openai.OpenAI = _FakeOpenAI
    sys.modules["openai"] = fake_openai

nb = json.loads((ROOT / "notebooks" / "DAgent-dev.ipynb").read_text(encoding="utf-8"))
PROVIDER_SRC = "".join(nb["cells"][29]["source"])
CHAT_SRC = "".join(nb["cells"][37]["source"])

BUBBLES = []


class _StubAgent:
    def __init__(self, config):
        self.config = config
        self.llm = None

    async def run(self, message):
        for _ in ():
            yield _


def build_namespace(secret_lookup):
    ns = {
        "__name__": "chat_cell",
        "json": json,
        "html": __import__("html"),
        "asyncio": asyncio,
        "IN_COLAB": False,
        "colab_output": None,
        "display": lambda x: BUBBLES.append(x),
        "HTML": lambda s: s,
        "get_secret": secret_lookup,
    }
    exec(compile(PROVIDER_SRC, "cell29", "exec"), ns)

    cfg = types.SimpleNamespace(
        provider="openrouter", model="", max_tokens=1024, temperature=0.2,
        base_url="", models_url="", key_env=None, provider_label="Custom",
        mode="safe", max_steps_per_run=20, workspace=".", agent_id="test",
    )
    ns["CONFIG"] = cfg
    ns["AgentConfig"] = type("AgentConfig", (), {})
    ns["DAgent"] = _StubAgent
    exec(compile(CHAT_SRC, "cell37", "exec"), ns)
    return ns


def install_get(ns, mapping):
    class Resp:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload

        def json(self):
            return self._payload

    def get(url, headers=None, params=None, timeout=None):
        for prefix, resp in mapping.items():
            if url.startswith(prefix):
                return resp
        return Resp(404, {})

    ns["requests"].get = get


KEYS = {"OPENROUTER_API_KEY": "k", "OPENAI_API_KEY": "k", "NVIDIA_API_KEY": "k", "GEMINI_API_KEY": "k"}


def test_models_callback_ok():
    ns = build_namespace(lambda n: KEYS.get(n))
    install_get(ns, {"https://api.openai.com/v1/models": type("R", (), {"status_code": 200, "json": lambda s: {"data": [{"id": "gpt-4o"}]}})()})
    res = ns["_chat_models_callback"]("openai")
    assert res["ok"] is True and [m["id"] for m in res["models"]] == ["gpt-4o"], res


def test_models_callback_error_is_returned_not_raised():
    ns = build_namespace(lambda n: KEYS.get(n))
    res = ns["_chat_models_callback"]("bogus")
    assert res["ok"] is False and "Supported:" in res["error"], res


def test_ui_has_all_providers_and_no_tokens():
    ns = build_namespace(lambda n: KEYS.get(n))
    ui = ns["_chat_ui"]
    for pid in ("openrouter", "nvidia", "openai", "gemini", "ollama", "llamacpp", "custom"):
        assert 'value="%s"' % pid in ui, pid
    assert "ka-provider" in ui and "ka-model" in ui and "ka-manual" in ui
    assert "__PROVIDER" not in ui and "__SEL_" not in ui and "__MAX_STEPS__" not in ui


def test_send_applies_provider_and_model():
    ns = build_namespace(lambda n: KEYS.get(n))
    install_get(ns, {})
    asyncio.run(ns["_run_chat_turn"]("hi", "autopilot", "5", "openai", "gpt-4o-mini"))
    assert ns["CONFIG"].provider == "openai"
    assert ns["CONFIG"].model == "gpt-4o-mini"
    assert ns["CONFIG"].mode == "autopilot"
    assert ns["CONFIG"].max_steps_per_run == 5


def test_missing_key_renders_error_bubble_not_crash():
    ns = build_namespace(lambda n: None)
    BUBBLES.clear()
    asyncio.run(ns["_run_chat_turn"]("hi", "safe", "5", "nvidia", "x"))
    joined = " ".join(str(b) for b in BUBBLES)
    assert "NVIDIA_API_KEY" in joined, joined
    assert "error" in joined.lower()


def test_capture_turn_is_called_when_the_cell_exists():
    ns = build_namespace(lambda n: KEYS.get(n))
    install_get(ns, {})
    calls = []
    ns["capture_turn"] = lambda message, events: calls.append((message, list(events)))
    asyncio.run(ns["_run_chat_turn"]("hi", "safe", "5", "openai", "gpt-4o-mini"))
    assert calls == [("hi", [])], calls


def test_capture_failure_never_breaks_the_chat_turn():
    ns = build_namespace(lambda n: KEYS.get(n))
    install_get(ns, {})

    def boom(message, events):
        raise RuntimeError("disk full")

    ns["capture_turn"] = boom
    # Must not raise: a capture failure is reported, not fatal.
    asyncio.run(ns["_run_chat_turn"]("hi", "safe", "5", "openai", "gpt-4o-mini"))


def test_chat_works_without_the_capture_cell():
    ns = build_namespace(lambda n: KEYS.get(n))
    install_get(ns, {})
    assert "capture_turn" not in ns
    asyncio.run(ns["_run_chat_turn"]("hi", "safe", "5", "openai", "gpt-4o-mini"))


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, "::", exc)
    print("\n%d failure(s)" % failed)
    raise SystemExit(1 if failed else 0)
