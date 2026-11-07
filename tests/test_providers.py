"""Offline regression tests for the notebook's provider cell (section 11).

The notebook is the product, so the test extracts cell 29 from notebooks/DAgent-dev.ipynb and
executes it with stubbed globals. No network, no real API keys.

Run:  python -m pytest tests -q      (or: python tests/test_providers.py)
"""
import json
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELL_INDEX = 29

# --- stub the openai package -------------------------------------------------
if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, base_url=None, api_key=None, default_headers=None):
            self.base_url = base_url
            self.api_key = api_key
            self.default_headers = default_headers or {}

    fake_openai.OpenAI = _FakeOpenAI
    sys.modules["openai"] = fake_openai


def load_cell():
    nb = json.loads((ROOT / "notebooks" / "DAgent-dev.ipynb").read_text(encoding="utf-8"))
    src = "".join(nb["cells"][CELL_INDEX]["source"])
    ns = {"__name__": "provider_cell"}
    exec(compile(src, "DAgent-dev.ipynb:cell%d" % CELL_INDEX, "exec"), ns)
    return ns


P = load_cell()


class Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def install_get(mapping):
    calls = []

    def get(url, headers=None, params=None, timeout=None):
        calls.append({"url": url, "headers": headers or {}, "params": params or {}})
        for prefix, resp in mapping.items():
            if url.startswith(prefix):
                return resp
        return Resp(404, {})

    P["requests"].get = get
    return calls


P["get_secret"] = lambda name: {
    "OPENROUTER_API_KEY": "k",
    "NVIDIA_API_KEY": "k",
    "OPENAI_API_KEY": "k",
    "GEMINI_API_KEY": "k",
}.get(name)


def test_openai_shape():
    install_get({"https://api.openai.com/v1/models": Resp(200, {"data": [{"id": "b"}, {"id": "a"}]})})
    ids = [r["id"] for r in P["list_models"]("openai")]
    assert ids == ["a", "b"], ids


def test_gemini_shape_strips_prefix_and_filters_embeddings():
    install_get({"https://generativelanguage.googleapis.com/v1beta/models": Resp(200, {"models": [
        {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash",
         "inputTokenLimit": 1048576, "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
    ]})})
    rows = P["list_models"]("gemini")
    assert [r["id"] for r in rows] == ["gemini-2.5-flash"], rows
    assert rows[0]["context"] == 1048576


def test_gemini_key_goes_in_query_not_headers():
    calls = install_get({"https://generativelanguage.googleapis.com/v1beta/models": Resp(200, {"models": [
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
    ]})})
    P["list_models"]("gemini")
    call = calls[-1]
    assert call["params"].get("key") == "k"
    assert "Authorization" not in call["headers"]


def test_ollama_shape():
    install_get({"http://127.0.0.1:11434/api/tags": Resp(200, {"models": [{"name": "llama3.2:3b"}]})})
    assert [r["id"] for r in P["list_models"]("ollama")] == ["llama3.2:3b"]


def test_unknown_provider_is_actionable():
    try:
        P["list_models"]("nope")
    except P["ProviderError"] as exc:
        assert "Supported:" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_401_is_named():
    install_get({"https://integrate.api.nvidia.com/v1/models": Resp(401, None)})
    try:
        P["list_models"]("nvidia")
    except P["ProviderError"] as exc:
        assert "401" in str(exc) and "NVIDIA_API_KEY" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_offline_local_hint():
    def boom(url, headers=None, params=None, timeout=None):
        raise P["requests"].exceptions.ConnectionError("refused")

    P["requests"].get = boom
    try:
        P["list_models"]("ollama")
    except P["ProviderError"] as exc:
        assert "running" in str(exc).lower()
    else:
        raise AssertionError("expected ProviderError")


def test_build_llm_defaults_and_base_url():
    install_get({})
    cfg = types.SimpleNamespace(provider="gemini", model="", max_tokens=8, temperature=0,
                                base_url="", models_url="", key_env=None)
    llm = P["build_llm"](cfg)
    assert llm.model == "gemini-2.5-flash"
    assert llm.client.base_url.startswith("https://generativelanguage")


def test_custom_requires_base_url():
    cfg = types.SimpleNamespace(provider="custom", model="m", base_url="", models_url="",
                                key_env="K", provider_label="X")
    try:
        P["build_llm"](cfg)
    except P["ProviderError"] as exc:
        assert "base_url" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_custom_base_url_normalised():
    cfg = types.SimpleNamespace(provider="custom", model="m", base_url="https://h/v1/",
                                models_url="", key_env="K", provider_label="X")
    spec = P["resolve_spec"]("custom", cfg)
    assert spec.base_url == "https://h/v1"
    assert spec.list_url == "https://h/v1/models"


def test_provider_table_marks_local():
    table = P["provider_table"]()
    assert len(table) == 7
    assert {r["id"] for r in table if r["local"]} == {"ollama", "llamacpp"}


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
