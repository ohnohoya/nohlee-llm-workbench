from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from .openai_utils import run_batch
from .schemas import RunRequest, RunResponse
from .model_registry import MODEL_REGISTRY

ROOT = Path(__file__).resolve()
PROJECT_ROOT = Path(os.getenv("OPENAI_LAB_PROJECT_ROOT", Path.cwd())).resolve()
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="OpenAI Tools API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.mount("/results_ui", StaticFiles(directory=PROJECT_ROOT / "results_ui"), name="results_ui")

@app.get("/models")
def list_models() -> dict[str, Any]:
    all_models = sorted(MODEL_REGISTRY.keys())
    reasoning_efforts = sorted(
        {effort for cfg in MODEL_REGISTRY.values() for effort in (cfg.reasoning_efforts or ())}
    )
    return {
        "models": all_models,
        "api_types": ["responses", "chat_completions"],
        "reasoning_efforts": reasoning_efforts,
    }

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/files")
def list_files() -> dict[str, Any]:
    files = []
    if OUTPUT_DIR.exists():
        for p in sorted(OUTPUT_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            files.append({"name": p.name, "path": f"output/{p.name}"})
    return {"files": files}


@app.post("/run", response_model=RunResponse)
async def run_jobs(payload: RunRequest) -> RunResponse:
    model_names = payload.model_names
    requests = [item.model_dump() for item in payload.requests]

    output_path = payload.output_path
    if output_path in (None, "", "string"):
        output_path = None

    result = await run_batch(
        model_names=model_names,
        max_tokens=payload.max_tokens,
        temperatures=payload.temperatures,
        api_types=payload.api_types,
        reasoning_efforts=payload.reasoning_efforts or [None],
        requests=requests,
        apply_param_corrections=payload.apply_param_corrections,
        concurrency=payload.concurrency,
        output_path=output_path,
        close_client=False,
    )

    failed_count = int(result.get("failed_count") or 0)
    if failed_count > 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "One or more run tasks failed. Check output logs for details.",
                "failed_count": failed_count,
                "output_path": result.get("output_path"),
            },
        )

    response = RunResponse(
        status="ok",
        output_path=result.get("output_path"),
        count=len(result.get("results") or []),
        results=result.get("results") if payload.return_results else None,
    )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("openai_lab.server:app", host="0.0.0.0", port=8000, reload=False)
