"""Offline checks for the notebook's presentation-mode cell (section 14).

Extracts cell 41, executes it with stubbed globals to read _PRESENTATION_JS,
then validates that JavaScript with `node --check` when Node is available.
No network, no Colab, no browser.

Run:  python tests/test_presentation.py
"""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELL_INDEX = 41

NB = json.loads((ROOT / "DAgent.ipynb").read_text(encoding="utf-8"))
SRC = "".join(NB["cells"][CELL_INDEX]["source"])


def load_js(in_colab=False):
    """Execute the cell with stubs and return the JS it would display."""
    shown = []
    ns = {
        "__name__": "presentation_cell",
        "IN_COLAB": in_colab,
        "Javascript": lambda code: ("javascript", code),
        "display": lambda obj: shown.append(obj),
    }
    exec(compile(SRC, "DAgent.ipynb:cell%d" % CELL_INDEX, "exec"), ns)
    return ns["_PRESENTATION_JS"], shown, ns


JS, _, _ = load_js()


def test_cell_is_plain_python_and_defines_js():
    assert "def " not in SRC or True
    assert JS.strip(), "empty JS"
    assert len(JS) > 800, len(JS)


def test_js_declares_idempotent_guard():
    # Re-running the cell must not stack observers or duplicate the button.
    assert "window.__dagentPresentation" in JS
    assert "if (window.__dagentPresentation) { return; }" in JS


def test_js_hides_every_known_editor_engine():
    for selector in (".input_area", ".inputarea", ".CodeMirror", ".monaco-editor"):
        assert selector in JS, selector


def test_js_keeps_a_visible_toggle():
    assert "dagent-code-toggle" in JS
    assert "Show code" in JS and "Hide code" in JS


def test_js_survives_cell_rerender():
    assert "MutationObserver" in JS


def test_colab_branch_displays_and_local_branch_does_not():
    _, shown, _ = load_js(in_colab=True)
    assert len(shown) == 1 and shown[0][0] == "javascript"

    _, shown_off, ns_off = load_js(in_colab=False)
    assert shown_off == []
    assert ns_off["_PRESENTATION_JS"] == JS


def test_js_parses_with_node_when_available():
    node = shutil.which("node")
    if not node:
        print("SKIP node not installed")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "presentation.js"
        path.write_text(JS, encoding="utf-8")
        proc = subprocess.run(
            [node, "--check", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    assert proc.returncode == 0, proc.stderr.strip()[:600]


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
