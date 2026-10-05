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


All execution artifacts, workspace logs, and snapshots are persistently stored under a customizable drive workspace (default: `/content/drive/MyDrive/KernelAgent`).

## 🛠️ Getting Started

### 1. Requirements

Upload this notebook directly to your **Google Colab** environment.
Ensure you have configured your secrets in the Colab Secrets Panel (🔑 icon on the left panel):
- `OPENROUTER_API_KEY` (if using OpenRouter)
- `NVIDIA_API_KEY` (if using NVIDIA integration)

### 2. How to Run

1. Run the cells sequentially from top to bottom.
2. When prompted, perform the **Approval System Smoke Test** to verify that your JS-to-Python callback is functioning.
3. Scroll to the bottom and use the interactive **KernelAgent Chat Box** to instruct the agent to execute complex tasks autonomously.

## 🛡️ Permission Modes

- **Safe (Default):** All actions tagged as `write`, `execute`, `package`, `git`, or `destructive` will pause execution and prompt you for live approval, modification, or denial.
- **Autopilot:** The agent runs freely without pauses or prompts.
- **Dry Run:** The agent analyzes and details the plan of actions it would perform without executing them.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

