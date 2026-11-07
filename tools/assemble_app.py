"""Assemble dagent_app.py from the tested notebook cells.

One-way build: notebooks/DAgent-dev.ipynb -> dagent_app.py.
The assembler fails loudly if an anchor it expects has moved, so the app can
never silently drift from the cells the test suite covers.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV_NB = ROOT / "notebooks" / "DAgent-dev.ipynb"
OUT = ROOT / "dagent_app.py"

# The dev notebook is the only valid input. The root DAgent.ipynb is the
# 3-cell launcher, so falling back to it would silently assemble garbage.
if not DEV_NB.is_file():
    raise SystemExit("missing %s - the 47-cell dev notebook is the assembler input" % DEV_NB)
NB_PATH = DEV_NB

# Code cells that make up the application. Excluded on purpose:
#   2  - `!pip install` (IPython magic; ensure_dependencies does this in Colab)
#   15 - demo cell: prints list_files/get_file_info output at import
#   19 - demo cell: runs a live web_search at import
#   27, 31, 35, 39 - smoke tests that need a running model/loop
APP_CELLS = [4, 6, 8, 10, 13, 17, 21, 23, 25, 29, 33, 37, 41, 42, 43, 44, 45, 46]

HEADER = '''"""DAgent - single-file application.

Run this file (or the one-cell launcher in DAgent.ipynb) and the interface
appears. Everything below is generated from notebooks/DAgent-dev.ipynb by
tools/assemble_app.py, which is what the test suite exercises.

    python dagent_app.py
"""

# =============================================================================
# Imports and environment
# =============================================================================
'''

IMPORT_BLOCK = '''import nest_asyncio
import requests
from bs4 import BeautifulSoup
from IPython.display import display, HTML, Javascript, update_display

nest_asyncio.apply()'''

IMPORT_BLOCK_GUARDED = '''try:                     # optional outside Jupyter/Colab
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
'''

PRESENTATION_TAIL_OLD = '''if IN_COLAB:
    display(Javascript(_PRESENTATION_JS))
    print("Presentation mode on: code cells hidden. Use the Show code button to bring them back.")
else:
    print("Presentation mode is Colab-only; plain Jupyter keeps the cells visible.")'''

PRESENTATION_TAIL_NEW = '''def render_presentation():
    """Hide notebook code in Colab and leave a reversible Show code toggle."""
    if IN_COLAB:
        display(Javascript(_PRESENTATION_JS))
    return _PRESENTATION_JS'''

OPENAI_IMPORT_OLD = '''import requests
from openai import OpenAI'''

OPENAI_IMPORT_NEW = '''import requests

try:
    from openai import OpenAI
except ImportError:      # only cloud providers need it; local ones work without it
    OpenAI = None'''

CLIENT_OLD = '''        self.client = OpenAI(base_url=spec.base_url, api_key=key, default_headers=headers)'''

CLIENT_NEW = '''        if OpenAI is None:
            raise ProviderError(
                "The 'openai' package is required for %s. "
                "Run:  %%pip install openai" % spec.label
            )
        self.client = OpenAI(base_url=spec.base_url, api_key=key, default_headers=headers)'''

CHAT_TAIL_OLD = '''if IN_COLAB:
    display(HTML(_chat_ui))
    print("Chat ready. Pick a provider, wait for the model list, then Send.")
else:
    print("Not running in Colab — use the console chat loop in the next cell instead.")'''

CHAT_TAIL_NEW = '''def render_chat():
    """Show the chat box. Safe to call more than once."""
    display(HTML(_chat_ui))
    return _chat_ui'''

LOCAL_TAIL_OLD = '''if IN_COLAB:
    display(HTML(_LOCAL_UI))
    print("Local model picker ready: choose a method first, then a model.")
else:
    print("Not running in Colab - the local model picker is a Colab UI. Use local_plan()/download_gguf() directly.")'''

LOCAL_TAIL_NEW = '''def render_local_models():
    """Show the local model picker. Safe to call more than once."""
    display(HTML(_LOCAL_UI))
    return _LOCAL_UI'''

TUNNEL_TAIL_OLD = '''print("Retraining tunnel ready: %d tiers" % len(tunnel_plan()["tiers"]))'''
TUNNEL_TAIL_NEW = '''# tunnel_plan() is called by the UI, not at import time.'''

FOOTER = '''

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
'''


def cell_source(nb, index):
    cell = nb["cells"][index]
    if cell["cell_type"] != "code":
        raise SystemExit("cell %d is not code" % index)
    return "".join(cell["source"]).rstrip("\n")


def replace_once(text, old, new, label):
    if old not in text:
        raise SystemExit("anchor missing while assembling %s - refusing to guess" % label)
    return text.replace(old, new, 1)


def build():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    parts = [HEADER]

    for index in APP_CELLS:
        source = cell_source(nb, index)

        if index == 4:
            source = replace_once(source, IMPORT_BLOCK, IMPORT_BLOCK_GUARDED, "imports")
        elif index == 29:
            source = replace_once(source, OPENAI_IMPORT_OLD, OPENAI_IMPORT_NEW, "openai import")
            source = replace_once(source, CLIENT_OLD, CLIENT_NEW, "client construction")
        elif index == 37:
            source = replace_once(source, CHAT_TAIL_OLD, CHAT_TAIL_NEW, "chat tail")
        elif index == 41:
            source = replace_once(source, PRESENTATION_TAIL_OLD, PRESENTATION_TAIL_NEW, "presentation tail")
        elif index == 43:
            source = replace_once(source, LOCAL_TAIL_OLD, LOCAL_TAIL_NEW, "local tail")
        elif index == 46:
            source = replace_once(source, TUNNEL_TAIL_OLD, TUNNEL_TAIL_NEW, "tunnel tail")

        parts.append("\n# =============================================================================\n"
                     "# Notebook cell %d\n"
                     "# =============================================================================\n" % index)
        parts.append(source + "\n")

    parts.append(FOOTER)
    return "".join(parts)


if __name__ == "__main__":
    text = build()
    OUT.write_text(text, encoding="utf-8")
    print("wrote %s (%d lines, %d chars)" % (OUT.name, len(text.splitlines()), len(text)))
