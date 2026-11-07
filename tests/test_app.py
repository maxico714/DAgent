"""Checks for the launcher notebook, the assembled app, and their sync.

DAgent.ipynb (root) is the 3-cell launcher a user runs in Colab.
dagent_app.py is the whole application, assembled from
notebooks/DAgent-dev.ipynb by tools/assemble_app.py.

The most important check here is test_app_is_in_sync: it re-runs the
assembler in memory and compares the result with the committed file, so the
app can never silently drift from the cells the rest of the suite tests.

Run:  python tests/test_app.py
"""
import ast
import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "DAgent.ipynb"
APP = ROOT / "dagent_app.py"
DEV = ROOT / "notebooks" / "DAgent-dev.ipynb"

sys.path.insert(0, str(ROOT / "tools"))
import assemble_app  # noqa: E402  (import runs the "dev notebook exists" guard)

NB = json.loads(LAUNCHER.read_text(encoding="utf-8"))
CELLS = NB["cells"]
CODE_CELLS = [c for c in CELLS if c["cell_type"] == "code"]
LAUNCH_SRC = "".join(CODE_CELLS[0]["source"]) if CODE_CELLS else ""
APP_SRC = APP.read_text(encoding="utf-8") if APP.is_file() else ""

KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"sk-or-v1-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}"),
]

# Demos/smokes that must never run at import time (they need a live network
# call or a running model). Cells 15, 19, 27, 31, 35 and 39 of the dev
# notebook are excluded from the app on purpose.
FORBIDDEN_IN_APP = [
    'web_search("current Python stable release"',
    'print("=== LIST FILES ===")',
    "async def console_chat",
    "await _run_test()",
    "LLM adapter ready",
]


# ---------------------------------------------------------------- launcher

def test_launcher_is_exactly_three_cells():
    assert NB.get("nbformat") == 4, NB.get("nbformat")
    assert [c["cell_type"] for c in CELLS] == ["markdown", "code", "markdown"], [
        c["cell_type"] for c in CELLS
    ]
    assert CELLS[1]["id"] == "launch01"


def test_launcher_code_parses_as_plain_python():
    # No IPython magics: the cell must be testable (and readable) as real Python.
    ast.parse(LAUNCH_SRC)
    for line in LAUNCH_SRC.splitlines():
        stripped = line.lstrip()
        assert not stripped.startswith(("!", "%")), "magic line: %r" % line


def test_launcher_installs_dependencies_before_starting_the_app():
    assert '"pip", "install"' in LAUNCH_SRC or "'pip', 'install'" in LAUNCH_SRC
    for package in ("openai", "requests", "beautifulsoup4", "nest_asyncio"):
        assert package in LAUNCH_SRC, package
    # install must come before exec of the fetched app
    assert LAUNCH_SRC.index('"pip", "install"') < LAUNCH_SRC.index("exec(compile(")


def test_launcher_fetches_the_published_app():
    expected = "https://raw.githubusercontent.com/maxico714/DAgent/main/dagent_app.py"
    assert expected in LAUNCH_SRC, "raw URL missing or changed"
    # and a local copy still wins when present (clone / offline dev)
    assert 'pathlib.Path("dagent_app.py")' in LAUNCH_SRC


def test_launcher_has_no_hardcoded_keys():
    hits = [p.pattern[:14] for p in KEY_PATTERNS if p.search(LAUNCH_SRC)]
    assert hits == [], hits


def test_launcher_has_no_committed_outputs():
    for cell in CODE_CELLS:
        assert cell.get("outputs") == [], cell.get("outputs")
        assert cell.get("execution_count") is None


def test_launcher_documents_keys_and_data_location():
    notes = "".join(CELLS[2]["source"])
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        assert name in notes, name
    assert "/content/drive/MyDrive/DAgent" in notes


# ------------------------------------------------------------------- app

def test_app_exists_and_parses():
    assert APP.is_file(), "dagent_app.py is missing - run python tools/assemble_app.py"
    ast.parse(APP_SRC)


def test_app_is_in_sync_with_the_dev_notebook():
    """The committed app must equal a fresh assembly of the dev notebook."""
    rebuilt = assemble_app.build()
    on_disk = APP_SRC.replace("\r\n", "\n")
    assert rebuilt.replace("\r\n", "\n") == on_disk, (
        "dagent_app.py is stale - run: python tools/assemble_app.py"
    )


def test_assembler_input_is_the_dev_notebook():
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    assert len(dev["cells"]) == 47, len(dev["cells"])
    assert assemble_app.NB_PATH == DEV
    for index in assemble_app.APP_CELLS:
        assert dev["cells"][index]["cell_type"] == "code", index
    # the excluded cells are demos/smokes, not app code
    for index in (15, 19, 27, 31, 35, 39):
        assert index not in assemble_app.APP_CELLS, index


def test_app_excludes_demo_and_smoke_cells():
    for needle in FORBIDDEN_IN_APP:
        assert needle not in APP_SRC, "demo leaked into the app: %r" % needle


def test_app_degrades_outside_colab():
    for needle in (
        "except ImportError:      # plain python: no event loop to patch",
        "except ImportError:      # only the web tools need it",
        "except ImportError:      # plain python: rendering is a no-op",
        "except ImportError:      # only cloud providers need it",
        "OpenAI = None",
    ):
        assert needle in APP_SRC, "missing guard: %r" % needle
    # the client constructor must fail with an actionable message, not TypeError
    assert "The 'openai' package is required for %s" in APP_SRC


def test_agent_builds_its_llm_lazily():
    """Regression: constructing DAgent must not require a provider package."""
    assert "self.llm = None" in APP_SRC
    assert "if self.llm is None:" in APP_SRC
    assert "LLM setup failed" in APP_SRC
    # the eager form must be gone from the constructor
    assert "        self.llm = build_llm(config)" not in APP_SRC


def test_app_has_one_entry_point():
    assert "def main():" in APP_SRC
    assert 'if __name__ == "__main__":\n    main()' in APP_SRC
    for call in ("render_presentation()", "render_header()", "render_chat()",
                 "render_local_models()", "render_status()"):
        assert call in APP_SRC, call
    # main() must call every renderer, in order. Anchor on the docstring:
    # the LoRA script string also contains a "def main():", so a plain split
    # would measure the wrong region.
    marker = 'def main():\n    """Start the application.'
    assert marker in APP_SRC, "main() entry point moved"
    body = APP_SRC.split(marker, 1)[1]
    order = [body.index(call) for call in (
        "render_presentation()", "render_header()", "render_chat()",
        "render_local_models()", "render_status()")]
    assert order == sorted(order), order


def test_app_has_no_hardcoded_keys():
    hits = [p.pattern[:14] for p in KEY_PATTERNS if p.search(APP_SRC)]
    assert hits == [], hits


def test_app_imports_in_a_bare_environment():
    """Import the app where only requests exists: it must not crash.

    This is the check that catches import-time work (eager LLM construction,
    demo cells making live calls), which is how the app failed before.
    """
    code = (
        "import dagent_app as a; "
        "print('IMPORT_OK', a.IN_COLAB, len(a.PROVIDERS))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tempfile.mkdtemp(prefix="dagent-app-import-"),
        capture_output=True, text=True, timeout=180,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0, "import failed:\nSTDOUT:\n%s\nSTDERR:\n%s" % (
        proc.stdout[-2000:], proc.stderr[-2000:])
    assert "IMPORT_OK False" in proc.stdout, proc.stdout[-1000:]


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
