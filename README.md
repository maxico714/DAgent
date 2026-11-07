# DAgent

[![tests](https://github.com/maxico714/DAgent/actions/workflows/tests.yml/badge.svg)](https://github.com/maxico714/DAgent/actions/workflows/tests.yml)

An autonomous AI agent with a **one-click Colab interface**: press Run on a
single cell and the chat app appears — no code to read, no 47 cells to execute.
Any OpenAI-compatible provider (OpenRouter, OpenAI, Gemini, NVIDIA, custom) or
a local one (Ollama, llama.cpp), a tool registry with permission levels, a
stable human-in-the-loop approval UI, Drive-backed memory, and a retraining
tunnel for your own captured conversations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxico714/DAgent/blob/main/DAgent.ipynb)

## 🚀 Features

- **One click, no visible code.** `DAgent.ipynb` is a 3-cell launcher: it
  installs what Colab is missing, fetches `dagent_app.py`, and renders the app.
  The launcher hides itself after running (Show code button brings it back).
- **Robust Tool Registry:** Fine-grained permissions (`read`, `network`, `write`, `execute`, `package`, `git`, `destructive`).
- **Colab-Safe Human-in-the-Loop UI:** Replaces fragile `ipywidgets` click bindings with Google Colab's native, stable JS-to-Python RPC bridge (`google.colab.kernel.invokeFunction`). No more broken buttons.
- **Built-in Toolsets:**
  - **Filesystem:** Sandboxed operations (list, read, write, search, delete, and automatic file snapshotting with undo support).
  - **Web Integration:** Direct scraping of DuckDuckGo HTML search results and readability fetching.
  - **Execution:** Live shell execution and package manager pip installs.
  - **Git Operations:** Status, diff, log, and commits.
- **Multi-Provider LLM Adapter:** One OpenAI-compatible adapter for every
  provider; model lists are discovered live, never hardcoded.
- **Interactive Chat Interface:** Provider/Model dropdowns, live runtime
  settings (mode and step limits), and per-turn tool activity.

## 📁 Repository Structure

```
DAgent.ipynb                  # the launcher users open in Colab (3 cells)
dagent_app.py                 # the whole application, generated - do not hand-edit
notebooks/DAgent-dev.ipynb    # the 47-cell dev notebook the tests exercise
tools/assemble_app.py         # dev notebook -> dagent_app.py
tools/build_launcher.py       # regenerates DAgent.ipynb
tests/                        # 113 offline checks (pytest tests -q)
```

All execution artifacts, workspace logs, snapshots and training data are stored
under the workspace (Google Drive when mounted: `/content/drive/MyDrive/DAgent`,
otherwise `/content/DAgent` in Colab).

## 🛠️ Getting Started

### Run it (Colab, one click)

1. Open the notebook: [colab.research.google.com/github/maxico714/DAgent/blob/main/DAgent.ipynb](https://colab.research.google.com/github/maxico714/DAgent/blob/main/DAgent.ipynb)
   (or click the badge above).
2. Press **Run** on the single code cell. It installs `openai`, `requests`,
   `beautifulsoup4` and `nest_asyncio` (only if missing), fetches the app, and
   the interface appears below the cell.
3. Add your key (the app ships **none**): Colab Secrets panel (🔑) → e.g.
   `OPENROUTER_API_KEY` → *Notebook access*. Or run
   `os.environ["OPENROUTER_API_KEY"] = "..."` in any cell.
4. Pick a provider, wait for the model list to load, type a message, **Send**.

With Drive mounted (recommended), chat memory, snapshots and `training_data/`
persist at `/content/drive/MyDrive/DAgent`; without it everything is
session-only.

## 🔌 Providers (bring your own model)

Pick the provider in the chat UI's **Provider** dropdown and add the matching
key in Colab Secrets. No provider list is hardcoded — `list_models()` asks the
provider.

| id | label | needs key | notes |
|---|---|---|---|
| `openrouter` | OpenRouter | `OPENROUTER_API_KEY` | model list is public; chat needs the key |
| `nvidia` | NVIDIA NIM | `NVIDIA_API_KEY` | |
| `openai` | OpenAI | `OPENAI_API_KEY` | |
| `gemini` | Google Gemini | `GEMINI_API_KEY` | OpenAI-compatible surface at `/v1beta/openai/` |
| `ollama` | Ollama (local) | none | must be running where the notebook runs |
| `llamacpp` | llama.cpp server | none | start `llama-server` with a GGUF first |
| `custom` | any endpoint | `CUSTOM_API_KEY` | set `CONFIG.base_url` |

Adding a provider is one row in `PROVIDERS` (section 11) — there is no per-vendor
adapter class. Every failure raises `ProviderError` with an actionable message
(401, 403, 404, unreachable, zero models) so the UI can show it verbatim.

### Chat UI

The chat box (section 13) has a **Provider** dropdown and a **Model** dropdown
that is filled from the provider's live catalogue on load and on every provider
change. If a provider cannot list models (no key, offline, unsupported
endpoint), the error is shown inline and the **Override model id** field still
lets you type a model name and send. Changing provider or model rebuilds the
adapter before the next turn — no kernel restart.

### Tests and development

```
pytest tests -q          # 113 offline checks: providers, agent loop, UI,
                         # training/RAG/tunnel, launcher and app sync
python tools/assemble_app.py    # regenerate dagent_app.py from the dev notebook
python tools/build_launcher.py  # regenerate DAgent.ipynb
```

The suite extracts cells from `notebooks/DAgent-dev.ipynb` and runs them with
stubbed globals — no network, no API keys, no Colab. `test_app.py` re-runs the
assembler in memory and compares it with the committed `dagent_app.py`, so the
app can never drift from the tested cells; it also imports the app in a bare
environment to prove nothing runs at import time that should not.

## 🧠 Your data → your model

Three sections, in the order the value arrives. Only the last one needs a GPU.

### 17 — Training data

Every chat turn is appended to `<WORKSPACE>/training_data/turns-YYYY-MM-DD.jsonl`
(Google Drive, when Drive is mounted), with the task, the final answer, the tools used,
the provider/model, and whether the turn succeeded. `TRAINING_ENABLED = False` stops it.

Credentials are removed **before** anything reaches disk: nine patterns covering
OpenAI/OpenRouter/GitHub/Hugging Face/Google/NVIDIA keys, `Bearer` headers and
`api_key = "..."` assignments. Specific prefixes are matched before the generic `sk-`,
so an OpenRouter key is never mislabelled as an OpenAI key.

```python
stats = dataset_stats()                  # how much usable data exists
export_sft()                             # writes sft-YYYYMMDD.jsonl
train, val = split_dataset(list_turns()) # deterministic split
```

### 18 — Personal knowledge (no training)

A small TF-IDF index over your own files. Retrieval at question time, not training.

```python
index = build_index("/content/drive/MyDrive/notes")
hits  = search(index, "how do we deploy the API?", k=4)
print(context_block(hits))               # paste into the prompt
```

Chunking splits on paragraphs and hard-wraps long ones; the index is plain JSON, so it
saves and loads without a vector database or an embedding API.

### 19 — Retraining tunnel

| Tier | What it does | Needs | Available today |
|---|---|---|---|
| 1 | Retrieval over your files | a folder of notes | yes |
| 2 | Style pack: your best answers as few-shot examples | 20+ usable turns | yes |
| 3 | LoRA fine-tune | 200+ usable turns **and** a CUDA GPU | config + script generated |

`tunnel_plan()` reports each tier with its exact blocker (`"3/20 usable turns"`,
`"no CUDA device visible"`) instead of pretending everything is ready.

Tier 3 is deliberately **not** automated into a button. `lora_train_script()` writes a
script marked `UNVERIFIED` in its own docstring — it was generated, not executed — and
`compare_answers(pairs, before, after)` exists so a fine-tune has to prove it helped on
held-out prompts before anyone trusts it. The metric is ROUGE-L, which is lexical: two
correct answers in different words score low. Treat it as a smoke detector, not a grade.

```
python tests/test_training.py
```

28 offline checks across all three sections.

## 💾 Local models

Section 15/16 ask for the **download method first**, then show the models — because the
same model has a different reference, size and command per method.

| Method | What it does | Needs |
|---|---|---|
| `ollama` | `ollama pull <model>` | Ollama running on the host |
| `gguf` | downloads one `.gguf` file here, then `llama-server` | llama.cpp built locally |
| `hf` | full-precision weights for transformers/vLLM/fine-tuning | `huggingface_hub`, token for gated repos |

Seven catalogue entries (Qwen2.5 1.5B/3B/7B, Qwen2.5 Coder 7B, Llama 3.2 1B/3B,
Gemma 2 2B) with **real file sizes read from the Hugging Face API** and their real
licences. Two are flagged in the UI: `qwen2.5-3b` carries an `other` licence, and the
Llama/Gemma repos are gated upstream.

The GGUF downloader is resumable and safe to kill:

- writes to `<file>.part` and renames only when complete, so a killed download never
  looks finished
- resumes with a `Range` request; if the server ignores the range it restarts cleanly
  instead of appending and corrupting the file
- handles `416` on an already-complete part file, verifies the final size, refuses
  non-http(s) URLs

```
python tests/test_local_models.py
python tests/test_local_ui.py
```

27 offline checks. The downloader ones run against a real HTTP server on `127.0.0.1`
and cover fresh download, resume, range-ignoring servers, `416`, and truncation.

## 🖥️ Presentation mode

Section 14 hides every code cell so the notebook reads as an app: only the chat box
and the outputs stay visible, and a floating **Show code** button brings the editors
back for the session. It hides whichever editor Colab is using (`.input_area`,
`.inputarea`, CodeMirror, Monaco) and re-applies itself when Colab re-renders a cell
after execution, so running a cell does not un-hide it. Nothing is deleted.

```
python tests/test_presentation.py
```

7 offline checks, including `node --check` on the injected JavaScript when Node is
installed.

## 🛡️ Permission Modes

- **Safe (Default):** All actions tagged as `write`, `execute`, `package`, `git`, or `destructive` will pause execution and prompt you for live approval, modification, or denial.
- **Autopilot:** The agent runs freely without pauses or prompts.
- **Dry Run:** The agent analyzes and details the plan of actions it would perform without executing them.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

