# Nohlee LLM Workbench

An open-source LLM workbench for experimenting with:

- prompt engineering
- batch model evaluation
- response parameter tuning
- model comparison
- model response detail reporting (for example, metadata such as actual temperature or reasoning settings used)

This project helps you run repeatable LLM experiments, save results as JSON, and inspect outputs in a local UI.

Currently supported provider: OpenAI (OpenAI models only).

## Optional: Install Codex CLI
If you want a local coding agent CLI workflow, install Codex globally:

```bash
npm install -g @openai/codex
```

Maintained by Jeonghun Noh (Nohlee Tech inc.).

![nohlee-llm-workbench banner](assets/github-banner.jpg)

## Roadmap
This project currently supports OpenAI only.
We plan to expand support to additional LLM providers in future releases.

## Docs
- Docs index: [docs/index.md](docs/index.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## Support
For questions, bugs, or feature requests, please open a GitHub Issue in this repository.

## Requirements
- Python 3.10+
- An OpenAI API key in `OPENAI_API_KEY`

## Install `uv` (if needed)
If `uv` is not installed, use one of these:

macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:
```bash
uv --version
```

## Setup (uv)
```bash
uv sync
```

Run all commands below from the repository root (`nohlee-llm-workbench`).

Optional: create a local `.env` file from `.envexample` and set your key there.
```bash
cp .envexample .env
```

## Docker
Run API + UI together with Docker Compose:
```bash
docker compose up --build
```

Endpoints:
- UI: `http://localhost:5173`
- API: `http://localhost:8000`

Notes:
- Set `OPENAI_API_KEY` in your shell or `.env` file before running compose.
- Results are persisted to local `output/` via a bind mount.
- If Docker commands fail, install Docker first:
  - macOS/Windows: install and run Docker Desktop.
  - Linux: install Docker Engine and make sure the Docker daemon is running.
- Example daemon error: `Cannot connect to the Docker daemon ...` means Docker is installed but not running yet.

## Setup without `uv`
You can run this app without `uv` using `venv` + `pip`.

What is `.venv`?
- `.venv` is a local Python environment folder created inside this project.
- It keeps this project’s packages separate from your system Python packages.

Create it:
```bash
python -m venv .venv
# if python is not found
python3 -m venv .venv
```

Activate it (macOS/Linux):
```bash
source .venv/bin/activate
```

Activate it (Windows PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install --upgrade pip
pip install openai python-dotenv fastapi uvicorn
pip install -e .
```

If you plan to use the UI, install frontend dependencies once:
```bash
(cd results_ui && npm install)
```

If `npm` is not installed (`command not found: npm`), install Node.js (which includes `npm`) first:

macOS (Homebrew):
```bash
brew install node
```

Windows:
- Install Node.js LTS from https://nodejs.org/en/download

Verify:
```bash
node --version
npm --version
```

Then install the UI dependencies:
```bash
(cd results_ui && npm install)
```

## Run once with `openai_client_runner.py`
Use this when you want a quick one-off run from the terminal.

1. Open `src/openai_lab/openai_client_runner.py` and set your manual config:
   - `RUN_HELLO_ONLY`: `True` runs only the hello check; `False` runs the batch.
   - `MODEL_NAMES`: models to test.
   - `MAX_TOKENS`: max output/completion tokens.
   - `TEMPERATURES`: list of temperatures (`None` means do not send temperature).
   - `API_TYPES`: `responses`, `chat_completions`, or `auto`.
   - `REASONING_EFFORTS`: list like `["auto", None, "minimal"]`.
   - `APPLY_PARAM_CORRECTIONS`: `True` (default safety/compat checks) or `False` (send reasoning/temperature as configured).
   - `USER_PROMPTS`: prompts/messages to run.
2. Run:
```bash
uv run python -m openai_lab.openai_client_runner
```
Outputs are written to `output/llm_results_<timestamp>.json`.

Without `uv`:
```bash
python -m openai_lab.openai_client_runner
```

## Run the UI + API (recommended)
For the full UI workflow, run both the FastAPI backend and the Vite frontend.

1. Start FastAPI (Terminal 1):
```bash
uv run uvicorn openai_lab.server:app --host 0.0.0.0 --port 8000
```
2. Start Vite UI (Terminal 2):
```bash
cd results_ui
npm install
npm run dev
```
3. Open the Vite URL (typically `http://localhost:5173`).

If a default port is already in use:
- FastAPI: change `--port 8000` to an open port (for example `8001`).
- Vite: it will automatically select the next open port.

Without `uv`, FastAPI command is:
```bash
uvicorn openai_lab.server:app --host 0.0.0.0 --port 8000
```

Why both are needed:
- Vite serves and compiles the React UI.
- FastAPI provides backend endpoints used by the UI: `/files`, `/output`, `/run`, `/models`.

If FastAPI is not running, the UI can still load uploaded JSON files locally, but server-backed features (file list, run submission, model loading) will fail.

## Submit Runs from the UI
In the **Run Configuration** panel, submit a batch run with:
- `model_names`, `max_tokens`, `temperatures`, `api_types`, `reasoning_efforts`, `apply_param_corrections`
- `requests` (each entry can include `label`, `prompt`, and/or `messages`)
Submitting a run writes a new results file to `output/` and refreshes the file list.

## Beginner Mode - UI Usage Guide
Use this section for your first UI run, then reuse the same workflow for later experiments.

The UI has two main areas:
1. **Run Configuration** for creating a run.
2. **Output Files** for loading and comparing results.

### 1. Run Configuration Panel
This panel defines the run matrix (models x temperatures x API types x reasoning efforts x prompts).

![Configuration panel top](assets/configuration-panel-top.png)

#### Step 1: Select models
- Choose one or more in **Model Names**.
- Multi-select with `Ctrl` (Windows/Linux) or `Cmd` (macOS).
- Good starter set: `gpt-5.2`, `gpt-5-mini`.
- Adding more models increases total comparisons and runtime.

#### Step 2: Set max tokens
- **Max Tokens** limits output length.
- Start with `300` for quick checks.
- Use `800` to `1200` for longer reasoning prompts.
- Larger values usually increase latency and cost.

#### Step 3: Add temperatures
- `0.0` = deterministic and stricter.
- `0.7` = more creative/varied.
- `auto` = do not force a value, use model default behavior.
- Leaving the list empty also means temperature is not sent.
- Add multiple values to compare behavior under different randomness levels.

#### Step 4: Choose API type
- `responses` is the recommended default.
- `chat_completions` is useful when you want to compare behavior across endpoint styles.
- You can select both to compare the same prompt across endpoints.

#### Step 5: Choose reasoning effort
- Start with `auto`.
- Common comparison set for reasoning models: `minimal`, `high`.
- Some models only support a subset of efforts. If `apply_param_corrections` is enabled, unsupported combinations are handled safely.

#### Step 6: Keep auto-correct enabled
- **Auto-correct params** ON (recommended): adjusts unsupported parameter combinations.
- OFF: sends values exactly as configured, useful when debugging parameter behavior.

![Configuration panel bottom](assets/configuration-panel-bottom.png)

#### Step 7: Add prompt(s)
- Add a short label and a prompt text.
- Use one focused prompt first, then add more once the pipeline works.
- Example prompt: `Explain quantum computing in simple terms.`
- Multiple prompts run in one batch and make comparisons easier.

#### Step 8: Submit
- Click **Submit** in the Run Configuration panel.
- You should see a submitting state and then a success result.
- Each successful run writes `output/llm_results_<timestamp>.json`.
- The file list refreshes after submission.

### 2. Output Files Panel
This panel loads and inspects run outputs.

![Output files panel bottom](assets/output-files-panel-bottom.png)

#### Load a run
- Select a file from **Output Files**.
- Or use **Upload JSON** to inspect a local results file.
- After loading, the UI shows how many records were read.

![Output files panel top](assets/output-files-panel-top.png)

#### Read result cards
- Each card represents one configuration.
- Cards include model name, API type, temperature, and a run SID.
- Expand `llm_call_config` for parameters actually sent.
- Expand `llm_request` for request payload details.
- Expand `llm_response` for returned output and metadata.
- Drag cards to reorder for easier side-by-side comparison.

#### Compare quickly
- Good starter comparison:
```text
Models: gpt-5.2, gpt-5-mini
Temperatures: 0.0, 0.7
API Types: responses
Reasoning Efforts: auto
Prompts: 1
```
- This gives 4 result cards and makes differences obvious without creating too much noise.

### Beginner Walkthrough Example
Use this minimal first run:

```text
Model Names: gpt-5.2, gpt-5-mini
Max Tokens: 300
Temperatures: 0.0, 0.7
API Types: responses
Reasoning Efforts: auto
Prompt: Explain quantum computing in simple terms.
```

Expected outcome:
1. One new JSON file appears in `output/`.
2. Loading that file shows 4 cards.
3. You can compare deterministic vs creative responses across two models.

### Tips For Better Experiments
- Change one variable at a time.
- Keep prompt wording fixed during comparisons.
- Start small, then add dimensions.
- Keep `apply_param_corrections` enabled unless testing edge cases.
- Reuse labels so runs are easier to scan later.

### Troubleshooting
- Models not loading: check FastAPI is running and `OPENAI_API_KEY` is set.
- Submit fails: inspect backend terminal logs and confirm model names from `/models`.
- Empty file list: confirm JSON files exist in `output/`.
- UI works but submit does not: verify frontend can reach backend at `http://localhost:8000`.

## Advanced Mode - UI Usage Guide
The UI is split into two main panels:

**Run Configuration**
- Collapsible panel for creating new batch runs.
- Model Names, API Types, and Reasoning Efforts are multi-select dropdowns.
- Temperatures are free-form multi-values: add entries like `0.0`, `0.7`, and `auto`.
- Requests are a friendly list of prompts (label + prompt).
- Use **Add Prompt** to append new requests, and **Remove** to delete one.

**Output Files**
- Lists saved results from `output/`.
- Use **Load File** to open the selected results file.
- **Expand All** is enabled by default after a successful load.
- You can also upload a JSON results file directly.

## Advanced Mode - FastAPI API Only (No UI)
If you only need the API (for scripts or curl/postman), run:
```bash
uv run uvicorn openai_lab.server:app --host 0.0.0.0 --port 8000
```

Without `uv`:
```bash
uvicorn openai_lab.server:app --host 0.0.0.0 --port 8000
```

Example 1: list available models/options
```bash
curl -s http://localhost:8000/models | jq
```

Example 2: submit one batch run
```bash
curl -s -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_names": ["gpt-5.2"],
    "requests": [{"label": "hello", "prompt": "Hi! Tell me a joke."}],
    "max_tokens": 300,
    "temperatures": [1.0],
    "api_types": ["responses"],
    "reasoning_efforts": ["auto"],
    "apply_param_corrections": true,
    "return_results": false
  }' | jq
```

`"auto"` means "use the model default reasoning effort".  
Backend behavior: the API accepts this string and forwards it through the run pipeline.

### OpenAI Python SDK vs OpenAI API

This codebase is designed to help you work with the OpenAI Python SDK. If you also want to test the OpenAI API directly (without the SDK), you can run:
```bash
curl -s -X POST "https://api.openai.com/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.2",
    "max_completion_tokens": 20,
    "temperature": 0.5,
    "messages": [{"role":"user","content":"Say hello in a creative way."}]
  }' | jq .
```
