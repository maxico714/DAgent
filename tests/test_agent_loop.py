"""Offline tests for the agent loop (section 12).

Extracts cells 29 and 33 from notebooks/DAgent-dev.ipynb and drives the loop with a scripted
fake LLM and a fake tool executor. No network, no keys, no Colab.

This is the core of the product: everything else is plumbing around it.

Run:  python tests/test_agent_loop.py
"""
import asyncio
import json
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROVIDER_INDEX = 29
LOOP_INDEX = 33

if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, base_url=None, api_key=None, default_headers=None):
            self.base_url = base_url
            self.api_key = api_key
            self.default_headers = default_headers or {}

    fake_openai.OpenAI = _FakeOpenAI
    sys.modules["openai"] = fake_openai

NB = json.loads((ROOT / "notebooks" / "DAgent-dev.ipynb").read_text(encoding="utf-8"))
PROVIDER_SRC = "".join(NB["cells"][PROVIDER_INDEX]["source"])
LOOP_SRC = "".join(NB["cells"][LOOP_INDEX]["source"])

TOOL_CALLS = []


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self.responses:
            raise AssertionError("fake LLM ran out of scripted responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


async def fake_execute(tool_name, raw_arguments):
    TOOL_CALLS.append({"tool": tool_name, "arguments": raw_arguments})
    return ("RESULT<%s>" % tool_name), {"ok": True, "tool": tool_name}


def build(responses, max_steps=5, mode="autopilot"):
    ns = {
        "__name__": "loop_cell",
        "json": json,
        "asyncio": asyncio,
        "types": types,
    }
    exec(compile(PROVIDER_SRC, "cell29", "exec"), ns)
    NormalizedResponse = ns["NormalizedResponse"]

    llm = FakeLLM(responses)
    cfg = types.SimpleNamespace(
        provider="openrouter", model="m", max_tokens=256, temperature=0,
        base_url="", models_url="", key_env=None, provider_label="X",
        mode=mode, max_steps_per_run=max_steps, tool_timeout_seconds=10,
        workspace=".", agent_id="t", max_tool_output_chars=1000, max_file_read_chars=1000,
    )
    ns.update({
        "AgentConfig": type("AgentConfig", (), {}),
        "CONFIG": cfg,
        "build_llm": lambda config: llm,
        "tools_for_openai": lambda: [{"type": "function", "function": {"name": "list_files"}}],
        "execute_tool_call": fake_execute,
    })
    exec(compile(LOOP_SRC, "cell33", "exec"), ns)
    return ns, llm, NormalizedResponse


def text_response(NormalizedResponse, text):
    return NormalizedResponse(text=text, tool_calls=[], usage={})


def tool_response(NormalizedResponse, name, arguments='{"path": "."}', call_id="c1", text=""):
    return NormalizedResponse(
        text=text,
        tool_calls=[{"id": call_id, "name": name, "arguments": arguments}],
        usage={},
    )


def collect(agent, task):
    async def runner():
        return [event async for event in agent.run(task)]

    return asyncio.run(runner())


def test_plain_answer():
    ns, llm, NR = build([None])
    llm.responses = [NR(text="hello", tool_calls=[], usage={})]
    agent = ns["DAgent"](ns["CONFIG"])
    events = collect(agent, "hi")

    assert [e["type"] for e in events] == ["assistant_text", "final"], events
    assert events[1]["content"] == "hello"
    assert agent.messages[0]["role"] == "system"
    assert agent.messages[1] == {"role": "user", "content": "hi"}
    assert agent.messages[-1] == {"role": "assistant", "content": "hello"}
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"], "tools must be offered to the model"


def test_tool_call_then_answer():
    TOOL_CALLS.clear()
    ns, llm, NR = build([None, None])
    llm.responses = [
        NR(text="", tool_calls=[{"id": "c1", "name": "list_files", "arguments": '{"path": "."}'}], usage={}),
        NR(text="done", tool_calls=[], usage={}),
    ]
    agent = ns["DAgent"](ns["CONFIG"])
    events = collect(agent, "list the files")

    kinds = [e["type"] for e in events]
    assert kinds == ["tool_call", "tool_result", "assistant_text", "final"], kinds
    assert events[0]["tool"] == "list_files"
    assert events[1]["result"] == "RESULT<list_files>"
    assert events[-1]["content"] == "done"

    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"], roles

    assistant_turn = agent.messages[2]
    assert assistant_turn["tool_calls"][0]["function"]["name"] == "list_files"
    assert assistant_turn["tool_calls"][0]["id"] == "c1"
    assert agent.messages[3]["tool_call_id"] == "c1"
    assert agent.messages[3]["content"] == "RESULT<list_files>"


def test_arguments_are_forwarded_as_the_raw_json_string():
    TOOL_CALLS.clear()
    raw = '{"path": "a b", "n": 2}'
    ns, llm, NR = build([None, None])
    llm.responses = [
        NR(text="", tool_calls=[{"id": "c1", "name": "read_file", "arguments": raw}], usage={}),
        NR(text="ok", tool_calls=[], usage={}),
    ]
    collect(ns["DAgent"](ns["CONFIG"]), "go")

    assert TOOL_CALLS[0]["arguments"] == raw, TOOL_CALLS
    assert isinstance(TOOL_CALLS[0]["arguments"], str)


def test_multiple_tool_calls_in_one_turn_keep_their_order():
    TOOL_CALLS.clear()
    ns, llm, NR = build([None, None])
    llm.responses = [
        NR(text="", tool_calls=[
            {"id": "a", "name": "list_files", "arguments": "{}"},
            {"id": "b", "name": "get_file_info", "arguments": "{}"},
        ], usage={}),
        NR(text="done", tool_calls=[], usage={}),
    ]
    agent = ns["DAgent"](ns["CONFIG"])
    events = collect(agent, "two tools")

    tool_events = [e for e in events if e["type"] == "tool_result"]
    assert [e["tool"] for e in tool_events] == ["list_files", "get_file_info"]
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b"]
    assert len(TOOL_CALLS) == 2


def test_text_and_tool_calls_in_one_response_emit_both():
    ns, llm, NR = build([None, None])
    llm.responses = [
        NR(text="thinking out loud", tool_calls=[{"id": "c1", "name": "list_files", "arguments": "{}"}], usage={}),
        NR(text="final", tool_calls=[], usage={}),
    ]
    events = collect(ns["DAgent"](ns["CONFIG"]), "go")
    assert [e["type"] for e in events][:3] == ["assistant_text", "tool_call", "tool_result"], events


def test_llm_error_yields_error_and_stops_without_raising():
    ns, llm, NR = build([None])
    llm.responses = [RuntimeError("provider exploded")]
    agent = ns["DAgent"](ns["CONFIG"])
    events = collect(agent, "go")

    assert len(events) == 1 and events[0]["type"] == "error", events
    assert "RuntimeError" in events[0]["message"] and "provider exploded" in events[0]["message"]
    assert len(llm.calls) == 1


def test_max_steps_is_enforced_exactly():
    ns, llm, NR = build([None, None, None], max_steps=3)
    llm.responses = [
        NR(text="", tool_calls=[{"id": "c%d" % i, "name": "list_files", "arguments": "{}"}], usage={})
        for i in range(3)
    ]
    agent = ns["DAgent"](ns["CONFIG"])
    events = collect(agent, "loop forever")

    assert len(llm.calls) == 3, len(llm.calls)
    assert events[-1]["type"] == "final"
    assert "max_steps_per_run=3" in events[-1]["content"], events[-1]


def test_history_accumulates_across_turns():
    ns, llm, NR = build([None, None])
    llm.responses = [
        NR(text="first", tool_calls=[], usage={}),
        NR(text="second", tool_calls=[], usage={}),
    ]
    agent = ns["DAgent"](ns["CONFIG"])
    collect(agent, "one")
    collect(agent, "two")

    users = [m["content"] for m in agent.messages if m["role"] == "user"]
    assert users == ["one", "two"], users
    assert len(llm.calls[1]["messages"]) == len(llm.calls[0]["messages"]) + 2


def test_empty_text_response_becomes_a_placeholder_final():
    ns, llm, NR = build([None])
    llm.responses = [NR(text="", tool_calls=[], usage={})]
    events = collect(ns["DAgent"](ns["CONFIG"]), "go")
    assert events[-1]["type"] == "final"
    assert events[-1]["content"] == "(no output)"


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
    sys.exit(1 if failed else 0)
