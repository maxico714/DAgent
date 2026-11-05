# DAgent

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

