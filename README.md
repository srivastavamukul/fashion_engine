# Fashion Engine

Minimal instructions to run the project locally.

Prerequisites
- Python 3.11+ (3.12 recommended)
- Create and activate a virtualenv or conda env

Install

```powershell
pip install -r requirements.txt
```

Run the CLI demo

```powershell
python main.py
```

Run the API

```powershell
uvicorn api.server:app --reload --port 8000
```

Notes
- The repo uses a mock video generator that writes simulated MP4 files to `outputs/` or `campaign_runs/` when the pipeline executes.
- If you want a dry run without file writes, run code that instantiates `FashionPipeline` and calls director methods but avoids `generator.generate()`.
