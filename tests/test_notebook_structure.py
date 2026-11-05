"""Structural checks for the notebook itself.

Every other test extracts cells by index, so an inserted cell would silently
make them test the wrong code. This file is the guard for that, and it also
proves that every code cell is valid Python once IPython magics are neutralised.

Run:  python tests/test_notebook_structure.py
"""
import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = json.loads((ROOT / "DAgent.ipynb").read_text(encoding="utf-8"))
CELLS = NB["cells"]

MAGIC_RE = re.compile(r"^\s*(?:![^=]|%%?[A-Za-z])")

# Index -> marker that must appear in that cell's source. This is the contract
# the rest of the test suite relies on.
EXPECTED = {
    6: "class AgentConfig",
    8: "def tools_for_openai",
    25: "async def execute_tool_call",
    29: "class OpenAICompatibleAdapter",
    33: "class DAgent",
    37: "_run_chat_turn",
    42: "LOCAL_METHODS = {",
    43: "_LOCAL_UI = ",
    44: "TRAINING_DIR",
    45: "RAG_TEXT_EXTENSIONS",
    46: "TUNNEL_MIN_TURNS_TIER2",
}


def neutralise(source):
    """Turn IPython magics (`!pip`, `%matplotlib`) into `pass`, keeping lines."""
    out = []
    for line in source.splitlines():
        if MAGIC_RE.match(line):
            indent = len(line) - len(line.lstrip())
            out.append(" " * indent + "pass")
        else:
            out.append(line)
    return "\n".join(out)


def test_notebook_is_valid_json():
    assert NB.get("nbformat") == 4, NB.get("nbformat")
    assert isinstance(CELLS, list) and CELLS, "no cells"


def test_every_code_cell_parses_after_neutralising_magics():
    checked, broken = 0, []
    for index, cell in enumerate(CELLS):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not source.strip():
            broken.append((index, "empty code cell"))
            continue
        checked += 1
        try:
            ast.parse(neutralise(source))
        except SyntaxError as exc:
            broken.append((index, "%s (line %s)" % (exc.msg, exc.lineno)))
    assert checked >= 20, checked
    assert broken == [], broken


def test_expected_cells_are_where_the_tests_expect_them():
    problems = []
    for index, marker in EXPECTED.items():
        if index >= len(CELLS):
            problems.append("cell %d is missing" % index)
            continue
        source = "".join(CELLS[index]["source"])
        if marker not in source:
            problems.append("cell %d does not contain %r" % (index, marker))
    assert problems == [], problems


def test_cell_count_matches_the_index_contract():
    # Appending is safe; inserting anywhere before cell 46 is not.
    assert len(CELLS) == 47, (
        "cell count changed to %d - update EXPECTED and every hardcoded index "
        "in tests/ before trusting the suite" % len(CELLS)
    )


def test_chat_turn_records_itself_when_capture_exists():
    source = "".join(CELLS[37]["source"])
    assert "events.append(event)" in source
    assert 'globals().get("capture_turn")' in source
    assert "training capture skipped" in source


def test_no_cell_contains_a_hardcoded_api_key():
    patterns = [
        re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
        re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    ]
    hits = []
    for index, cell in enumerate(CELLS):
        source = "".join(cell["source"])
        for pattern in patterns:
            if pattern.search(source):
                hits.append((index, pattern.pattern[:12]))
    assert hits == [], hits


def test_notebook_has_no_committed_outputs_or_execution_counts():
    dirty = []
    for index, cell in enumerate(CELLS):
        if cell["cell_type"] != "code":
            continue
        if cell.get("outputs"):
            dirty.append((index, "outputs"))
        if cell.get("execution_count") is not None:
            dirty.append((index, "execution_count"))
    assert dirty == [], dirty


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
