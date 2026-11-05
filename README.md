# DAgent

[![tests](https://github.com/maxico714/DAgent/actions/workflows/tests.yml/badge.svg)](https://github.com/maxico714/DAgent/actions/workflows/tests.yml)

An autonomous, self-contained agentic notebook designed to run seamlessly inside Google Colab (and fallback gracefully in plain Jupyter environments). It features a robust tool registry with permission levels, a stable human-in-the-loop approval interface, and persistent memory.

## 🚀 Features

- **Robust Tool Registry:** Fine-grained permissions (`read`, `network`, `write`, `execute`, `package`, `git`, `destructive`).
- **Colab-Safe Human-in-the-Loop UI:** Replaces fragile `ipywidgets` click bindings with Google Colab's native, stable JS-to-Python RPC bridge (`google.colab.kernel.invokeFunction`). No more broken buttons.
- **Built-in Toolsets:**
  - **Filesystem:** Sandboxed operations (list, read, write, search, delete, and automatic file snapshotting with undo support).
  - **Web Integration:** Direct scraping of DuckDuckGo HTML search results and readability fetching.
  - **Execution:** Live shell execution and package manager pip installs.
  - **Git Operations:** Status, diff, log, and commits.
- **Multi-Provider LLM Adapter:** Built-in adapters supporting OpenRouter (Gemini, etc.) and NVIDIA NIM (DeepSeek models) with robust native tool calling.
- **Interactive Chat Interface:** Rich HTML/CSS Chat UI appended right inside the notebook, complete with live runtime settings (mode and step constraints).

## 📁 Repository Structure

├── DAgent.ipynb # Main Colab interactive notebook └── README.md # Project documentation


All execution artifacts, workspace logs, and snapshots are persistently stored under a customizable drive workspace (default: `/content/drive/MyDrive/DAgent`).

## 🛠️ Getting Started

### 1. Requirements

Upload this notebook directly to your **Google Colab** environment.
Ensure you have configured your secrets in the Colab Secrets Panel (🔑 icon on the left panel):
- `OPENROUTER_API_KEY` (if using OpenRouter)
- `NVIDIA_API_KEY` (if using NVIDIA integration)

### 2. How to Run

1. Run the cells sequentially from top to bottom.
2. When prompted, perform the **Approval System Smoke Test** to verify that your JS-to-Python callback is functioning.
3. Scroll to the bottom and use the interactive **DAgent Chat Box** to instruct the agent to execute complex tasks autonomously.

## 🔌 Providers (bring your own model)

Set `CONFIG.provider` in section 3 and add the matching key in Colab Secrets.
No provider list is hardcoded — `list_models()` asks the provider.

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

### Tests

```
python tests/test_providers.py
```

11 offline checks that extract the provider cell from the notebook and run it
with stubbed globals — no network, no API keys.

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

