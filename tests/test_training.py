"""Offline checks for sections 17-19: training data, retrieval, retraining tunnel.

Extracts cells 44, 45 and 46 from DAgent.ipynb, executes them in one namespace
against a temporary workspace, and checks the behaviour. No network, no GPU,
no Colab.

Run:  python tests/test_training.py
"""
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import types
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_CELL, RAG_CELL, TUNNEL_CELL = 44, 45, 46

NB = json.loads((ROOT / "DAgent.ipynb").read_text(encoding="utf-8"))
SOURCES = {i: "".join(NB["cells"][i]["source"]) for i in (DATA_CELL, RAG_CELL, TUNNEL_CELL)}

TMP = tempfile.mkdtemp(prefix="dagent-training-")
WORKSPACE = os.path.join(TMP, "workspace")
os.makedirs(WORKSPACE, exist_ok=True)


def load():
    ns = {
        "__name__": "training_cells",
        "os": os,
        "re": re,
        "json": json,
        "glob": __import__("glob"),
        "hashlib": __import__("hashlib"),
        "random": __import__("random"),
        "math": __import__("math"),
        "datetime": datetime,
        "timezone": timezone,
        "WORKSPACE": WORKSPACE,
        "CONFIG": types.SimpleNamespace(agent_id="test-agent", provider="openrouter",
                                        model="m", mode="autopilot"),
    }
    for index in (DATA_CELL, RAG_CELL, TUNNEL_CELL):
        exec(compile(SOURCES[index], "cell%d" % index, "exec"), ns)
    return ns


T = load()
TRAINING_DIR = os.path.join(WORKSPACE, "training_data")


def reset_training_dir():
    shutil.rmtree(TRAINING_DIR, ignore_errors=True)


def turn_events(answer="A useful answer that is long enough to be kept.", error=None):
    events = [{"type": "tool_call", "tool": "list_files", "arguments": "{}"},
              {"type": "tool_result", "tool": "list_files", "result": "ok"}]
    if error:
        events.append({"type": "error", "message": error})
    events.append({"type": "final", "content": answer})
    return events


def seed_turns(count=25, answer=None):
    reset_training_dir()
    for i in range(count):
        T["capture_turn"]("question %d" % i, turn_events(answer or ("answer %d " % i) * 6))
    return T["list_turns"]()


# ---------------------------------------------------------------- redaction

def test_redacts_known_credential_shapes():
    cases = {
        "sk-abcdefghijklmnopqrstuvwx": "[REDACTED:openai-key]",
        "sk-or-v1-abcdefghijklmnopqrstuvwx": "[REDACTED:openrouter-key]",
        "ghp_abcdefghijklmnopqrstuvwxyz": "[REDACTED:github-token]",
        "hf_abcdefghijklmnopqrstuvwxyz": "[REDACTED:hf-token]",
        "AIza" + "a" * 33: "[REDACTED:google-key]",
        "nvapi-" + "a" * 24: "[REDACTED:nvidia-key]",
    }
    for secret, token in cases.items():
        out = T["redact"]("my key is %s ok" % secret)
        assert token in out, (secret, out)
        assert secret not in out, (secret, out)


def test_redacts_bearer_and_assignment_forms():
    assert "Bearer [REDACTED]" in T["redact"]("Authorization: Bearer abcdefghijklmnop1234")
    out = T["redact"]('api_key = "abcdefghijklmnop1234"')
    assert "abcdefghijklmnop1234" not in out and "[REDACTED]" in out, out


def test_redaction_leaves_normal_text_alone():
    text = "The function returns a list of files and the test passes in 4 seconds."
    assert T["redact"](text) == text
    assert T["redact"](None) == ""


# ---------------------------------------------------------------- capture

def test_capture_writes_one_jsonl_record():
    reset_training_dir()
    record = T["capture_turn"]("list the files", turn_events())
    assert record["task"] == "list the files"
    assert record["steps"] == 1 and record["tools"] == ["list_files"]
    assert record["ok"] is True and record["error"] == ""
    assert record["id"].startswith("t-") and len(record["id"]) == 14
    assert record["provider"] == "openrouter"

    files = list(pathlib.Path(TRAINING_DIR).glob("turns-*.jsonl"))
    assert len(files) == 1, files
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["task"] == "list the files"


def test_capture_redacts_before_writing():
    reset_training_dir()
    T["capture_turn"]("use sk-abcdefghijklmnopqrstuvwx", turn_events("done with sk-abcdefghijklmnopqrstuvwx"))
    raw = list(pathlib.Path(TRAINING_DIR).glob("turns-*.jsonl"))[0].read_text(encoding="utf-8")
    assert "sk-abcdefghijklmnopqrstuvwx" not in raw
    assert "[REDACTED:openai-key]" in raw


def test_capture_can_be_disabled():
    reset_training_dir()
    T["TRAINING_ENABLED"] = False
    try:
        assert T["capture_turn"]("x", turn_events()) is None
        assert T["list_turns"]() == []
    finally:
        T["TRAINING_ENABLED"] = True


def test_error_turns_are_marked_not_ok():
    reset_training_dir()
    record = T["capture_turn"]("boom", turn_events(error="RuntimeError: nope"))
    assert record["ok"] is False
    assert "nope" in record["error"]


def test_malformed_lines_are_skipped_not_fatal():
    reset_training_dir()
    T["capture_turn"]("good", turn_events())
    path = list(pathlib.Path(TRAINING_DIR).glob("turns-*.jsonl"))[0]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("{not json}\n")
    records = T["list_turns"]()
    assert len(records) == 1, records


def test_stats_and_dedupe():
    records = seed_turns(4)
    stats = T["dataset_stats"](records)
    assert stats["total"] == 4 and stats["usable"] == 4
    assert stats["with_tools"] == 4
    assert stats["providers"] == {"openrouter": 4}
    assert stats["days"] == 1

    duplicated = records + [records[0]]
    assert len(T["dedupe"](duplicated)) == 4


def test_export_sft_filters_and_writes_messages():
    reset_training_dir()
    T["capture_turn"]("short one", turn_events("too short"))
    T["capture_turn"]("good one", turn_events("A long enough answer to survive the filter."))
    T["capture_turn"]("broken", turn_events("A long enough answer but the turn failed.", error="boom"))

    dest = os.path.join(TRAINING_DIR, "sft-test.jsonl")
    result = T["export_sft"](dest=dest)
    assert result["rows"] == 1, result

    rows = [json.loads(l) for l in pathlib.Path(dest).read_text(encoding="utf-8").splitlines()]
    assert rows[0]["messages"][0] == {"role": "user", "content": "good one"}
    assert rows[0]["messages"][1]["role"] == "assistant"


def test_export_sft_can_include_a_system_prompt():
    records = seed_turns(2)
    dest = os.path.join(TRAINING_DIR, "sft-system.jsonl")
    T["export_sft"](dest=dest, records=records, system="You are DAgent.")
    first = json.loads(pathlib.Path(dest).read_text(encoding="utf-8").splitlines()[0])
    assert first["messages"][0] == {"role": "system", "content": "You are DAgent."}


def test_split_is_deterministic_and_proportional():
    records = seed_turns(10)
    train_a, val_a = T["split_dataset"](records, val_ratio=0.2, seed=7)
    train_b, val_b = T["split_dataset"](records, val_ratio=0.2, seed=7)
    assert [r["id"] for r in val_a] == [r["id"] for r in val_b]
    assert len(val_a) == 2 and len(train_a) == 8
    assert not ({r["id"] for r in train_a} & {r["id"] for r in val_a})

    try:
        T["split_dataset"](records, val_ratio=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------- retrieval

DOCS = {
    "notes/ollama.md": "Ollama serves an OpenAI compatible API on port 11434. Pull models with ollama pull.",
    "notes/llamacpp.md": "llama.cpp builds llama-server. Serve a GGUF file on port 8080 for local inference.",
    "notes/billing.txt": "Invoices are generated on the first day of the month and emailed to finance.",
    "notes/skip.bin": "binary-ish content that must be ignored",
}


def make_docs():
    folder = os.path.join(TMP, "docs")
    shutil.rmtree(folder, ignore_errors=True)
    for rel, text in DOCS.items():
        path = pathlib.Path(folder, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return folder


def test_chunking_respects_size_and_overlap():
    text = ("paragraph one. " * 40) + "\n\n" + ("paragraph two. " * 40)
    chunks = T["chunk_text"](text, size=200, overlap=20)
    assert chunks and all(len(c) <= 200 for c in chunks), [len(c) for c in chunks]
    assert "paragraph" in chunks[0]

    long_block = "x" * 1000
    hard = T["chunk_text"](long_block, size=300, overlap=50)
    assert len(hard) >= 3 and all(len(c) <= 300 for c in hard)

    assert T["chunk_text"]("") == []
    for bad in ((0, 0), (100, 100), (100, -1)):
        try:
            T["chunk_text"]("abc", size=bad[0], overlap=bad[1])
        except ValueError:
            continue
        raise AssertionError("expected ValueError for %r" % (bad,))


def test_tokenize():
    assert T["tokenize"]("Hello, World! a b 123") == ["hello", "world", "123"]
    assert T["tokenize"](None) == []


def test_index_and_search_rank_the_right_document_first():
    folder = make_docs()
    index = T["build_index"](folder)
    assert index["n_chunks"] >= 3
    assert "skip.bin" not in " ".join(c["source"] for c in index["chunks"])

    hits = T["search"](index, "how do I serve a gguf file locally", k=2)
    assert hits and hits[0]["source"] == "notes/llamacpp.md", hits

    hits = T["search"](index, "when are invoices emailed", k=1)
    assert hits[0]["source"] == "notes/billing.txt", hits


def test_search_returns_nothing_for_unrelated_queries():
    index = T["build_index"](make_docs())
    assert T["search"](index, "zzzz qqqq", k=3) == []
    assert T["search"]({"chunks": []}, "anything") == []


def test_index_round_trips_through_disk():
    folder = make_docs()
    index = T["build_index"](folder)
    path = os.path.join(TMP, "index.json")
    T["save_index"](index, path)
    loaded = T["load_index"](path)
    assert loaded["n_chunks"] == index["n_chunks"]
    hits = T["search"](loaded, "ollama port", k=1)
    assert hits and hits[0]["source"] == "notes/ollama.md", hits

    bad = os.path.join(TMP, "bad.json")
    pathlib.Path(bad).write_text('{"nope": 1}', encoding="utf-8")
    try:
        T["load_index"](bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_empty_folder_raises():
    empty = os.path.join(TMP, "empty")
    os.makedirs(empty, exist_ok=True)
    try:
        T["build_index"](empty)
    except ValueError as exc:
        assert "No indexable text" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    try:
        T["build_index"](os.path.join(TMP, "does-not-exist"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


def test_context_block_is_capped_and_empty_safe():
    assert T["context_block"]([]) == ""
    hits = [{"source": "a.md", "text": "x" * 3000, "score": 1.0},
            {"source": "b.md", "text": "y" * 3000, "score": 0.5}]
    block = T["context_block"](hits, max_chars=1000)
    assert len(block) <= 1100, len(block)
    assert "a.md" in block


# ---------------------------------------------------------------- tunnel

def test_rouge_l_bounds():
    assert T["rouge_l"]("the cat sat", "the cat sat") == 1.0
    assert T["rouge_l"]("alpha beta gamma", "delta epsilon zeta") == 0.0
    partial = T["rouge_l"]("the cat sat on the mat", "the cat sat")
    assert 0.0 < partial < 1.0, partial
    assert T["rouge_l"]("", "anything") == 0.0


def test_evaluate_answers_scores_a_predictor():
    pairs = [{"task": "q1", "answer": "alpha beta"}, {"task": "q2", "answer": "gamma delta"}]
    perfect = T["evaluate_answers"](pairs, lambda task: {"q1": "alpha beta", "q2": "gamma delta"}[task])
    assert perfect["n"] == 2 and perfect["mean_rouge_l"] == 1.0 and perfect["exact"] == 2

    def boom(task):
        raise RuntimeError("no model")

    broken = T["evaluate_answers"](pairs, boom)
    assert broken["mean_rouge_l"] == 0.0
    assert "no model" in broken["worst"][0]["error"]


def test_compare_answers_reports_a_verdict():
    pairs = [{"task": "q", "answer": "alpha beta gamma"}]
    better = T["compare_answers"](pairs, lambda t: "zzz", lambda t: "alpha beta gamma")
    assert better["verdict"] == "better" and better["delta_rouge_l"] > 0

    worse = T["compare_answers"](pairs, lambda t: "alpha beta gamma", lambda t: "zzz")
    assert worse["verdict"] == "worse" and worse["delta_rouge_l"] < 0


def test_style_pack_picks_long_answers_and_fits_the_budget():
    records = seed_turns(3)
    pack = T["build_style_pack"](records, n=2, max_chars=300)
    assert "User: question" in pack and "Assistant:" in pack
    assert len(pack) <= 400, len(pack)
    assert T["build_style_pack"]([]) == ""
    assert T["style_pack_prompt"]("   ") == ""
    assert "Match the tone" in T["style_pack_prompt"]("User: a\nAssistant: b")


def test_tunnel_plan_is_honest_about_blockers():
    records = seed_turns(3)
    plan = T["tunnel_plan"](records)
    assert [t["tier"] for t in plan["tiers"]] == [1, 2, 3]
    assert plan["tiers"][0]["available"] is True
    assert plan["tiers"][1]["available"] is False
    assert "3/20" in plan["tiers"][1]["blocker"], plan["tiers"][1]
    assert plan["tiers"][2]["available"] is False
    assert plan["tiers"][2]["blocker"]
    assert "cuda" in plan["gpu"] or "torch" in plan["gpu"]


def test_tunnel_plan_opens_tier2_with_enough_data():
    records = seed_turns(21)
    plan = T["tunnel_plan"](records)
    assert plan["tiers"][1]["available"] is True
    assert plan["tiers"][1]["blocker"] == ""


def test_lora_config_and_script_are_written_and_marked_unverified():
    path = os.path.join(TMP, "lora_config.json")
    config = T["write_lora_config"](path, r=8, lora_alpha=16)
    assert config["r"] == 8 and config["task_type"] == "CAUSAL_LM"
    assert json.loads(pathlib.Path(path).read_text(encoding="utf-8"))["lora_alpha"] == 16

    script = T["lora_train_script"](os.path.join(TMP, "lora_train.py"))
    text = pathlib.Path(script).read_text(encoding="utf-8")
    assert "UNVERIFIED" in text
    assert "SFTTrainer" in text and "LoraConfig" in text


def test_holdout_pairs_are_deterministic():
    records = seed_turns(5)
    a = T["holdout_pairs"](records, n=3, seed=3)
    b = T["holdout_pairs"](records, n=3, seed=3)
    assert [p["task"] for p in a] == [p["task"] for p in b]
    assert len(a) == 3 and all(p["task"] and p["answer"] for p in a)


def test_gpu_status_never_raises():
    status = T["gpu_status"]()
    assert set(status) >= {"torch", "cuda", "device", "reason"}
    assert isinstance(status["cuda"], bool)


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
