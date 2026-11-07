"""Generate the 3-cell DAgent.ipynb launcher (the user-facing entry point).

The launcher is deliberately tiny: one markdown intro, one code cell that
installs what Colab is missing and starts dagent_app.py, one markdown with
keys/data/troubleshooting notes. Users never see or run the 47-cell dev
notebook (notebooks/DAgent-dev.ipynb), which is where the tested code lives.

Run:  python tools/build_launcher.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "DAgent.ipynb"

INTRO = """# DAgent

**One click: press Run on the next cell.** The interface appears below — chat,
provider picker, local-model picker and your data panel. There is no other cell
to run and no code to read.

- Any OpenAI-compatible provider (OpenRouter, OpenAI, Gemini, NVIDIA, custom)
  or a local one (Ollama, llama.cpp) — pick it in the UI, model lists load live.
- Your chat turns are recorded to Google Drive (when mounted) so you can
  retrain later; credentials are stripped before anything is written.
- The app ships with **no API key**. Add your own key (see the notes below).

Code, tests and development live in
[notebooks/DAgent-dev.ipynb](https://github.com/maxico714/DAgent/blob/main/notebooks/DAgent-dev.ipynb)
and `dagent_app.py`; this notebook is only the launcher.
"""

LAUNCHER = '''# DAgent launcher: install what Colab is missing, fetch the app, start it.
# The app itself is dagent_app.py - this cell never contains product logic.
import importlib.util
import os
import pathlib
import subprocess
import sys
import urllib.request

RAW = "https://raw.githubusercontent.com/maxico714/DAgent/main/dagent_app.py"
REQUIRED = [("openai", "openai"), ("requests", "requests"),
            ("bs4", "beautifulsoup4"), ("nest_asyncio", "nest_asyncio")]

missing = [pkg for mod, pkg in REQUIRED if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + missing)

# Keys come from the user, never from this repo. If the same names were added
# as Colab Secrets (the key icon), copy them into the environment so the
# provider layer can find them. Any failure here must not stop the app.
try:
    from google.colab import userdata
    for _name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
                  "NVIDIA_API_KEY", "CUSTOM_API_KEY"):
        if not os.environ.get(_name):
            try:
                os.environ[_name] = userdata.get(_name)
            except Exception:
                pass
except Exception:
    pass

# Prefer a local copy (a clone of this repo), otherwise fetch from GitHub.
_local = pathlib.Path("dagent_app.py")
if _local.is_file():
    print("Starting DAgent from the local dagent_app.py ...")
    _src = _local.read_text(encoding="utf-8")
else:
    print("Fetching dagent_app.py from GitHub ...")
    try:
        _src = urllib.request.urlopen(RAW, timeout=30).read().decode("utf-8")
    except Exception as exc:
        raise SystemExit(
            "Could not fetch dagent_app.py from\\n  %s\\n(%s: %s).\\n"
            "Check the network, or upload dagent_app.py next to this notebook "
            "and run this cell again." % (RAW, type(exc).__name__, exc))

_ns = {"__name__": "dagent_app"}
exec(compile(_src, "dagent_app.py", "exec"), _ns)
_ns["main"]()
'''

NOTES = """## What just ran

One cell installed the four packages Colab does not ship (`openai`, `requests`,
`beautifulsoup4`, `nest_asyncio`), fetched `dagent_app.py`, and started it. The
code cell above hides itself — use the **Show code** button (bottom right) if
you want to read it.

## Your API key (the app ships none)

| Provider | Environment variable |
|---|---|
| OpenRouter | `OPENROUTER_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| NVIDIA | `NVIDIA_API_KEY` |
| custom endpoint | `CUSTOM_API_KEY` |
| Ollama / llama.cpp | none needed |

Two ways to set one in Colab:

1. **Secrets panel** (key icon, left sidebar) → add e.g. `OPENROUTER_API_KEY`
   → enable *Notebook access*. The launcher copies it into the environment.
2. Or run `os.environ["OPENROUTER_API_KEY"] = "..."` in any cell.

Never paste a key into this notebook's code or a screenshot.

## Where your data goes

- **With Drive mounted** (recommended): `/content/drive/MyDrive/DAgent/` —
  chat memory, snapshots and `training_data/` (JSONL per day) persist.
- **Without Drive**: `/content/DAgent` for the session only — it disappears
  when the runtime resets.

Nothing is uploaded anywhere except the provider you choose to talk to.

## Troubleshooting

- **"Could not fetch dagent_app.py"** — network issue; reconnect and re-run,
  or upload `dagent_app.py` from the repo next to this notebook.
- **"The 'openai' package is required"** — re-run the cell (it installs on
  first run); `%pip install openai` also works.
- **Model list stays empty** — the key is missing or wrong for that provider;
  the picker shows which variable it needs.
- The dev notebook (`notebooks/DAgent-dev.ipynb`), tests (`pytest tests -q`)
  and the builder (`python tools/assemble_app.py`) are for development only.
"""


def cell(kind, source, cid):
    lines = [line + "\n" for line in source.split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    out = {"cell_type": kind, "id": cid, "metadata": {}, "source": lines}
    if kind == "code":
        out["execution_count"] = None
        out["outputs"] = []
    return out


def build():
    return {
        "cells": [
            cell("markdown", INTRO, "intro001"),
            cell("code", LAUNCHER, "launch01"),
            cell("markdown", NOTES, "notes001"),
        ],
        "metadata": {
            "colab": {"provenance": [], "name": "DAgent.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build()
    OUT.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote %s (%d cells)" % (OUT.name, len(nb["cells"])))
