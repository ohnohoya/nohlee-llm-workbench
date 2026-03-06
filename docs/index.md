# openai-lab Docs

This docs index is a crawl-friendly entry point for the project.

## What is openai-lab?
openai-lab is a lightweight toolkit for experimenting with OpenAI models:
- Run batch prompts from Python
- Serve a FastAPI backend (`/models`, `/run`, `/files`, `/output`)
- Inspect results in a local React/Vite UI
- Run locally or with Docker Compose

## Start Here
- Main README: [../README.md](../README.md)
- Changelog: [../CHANGELOG.md](../CHANGELOG.md)

## Core Commands
- Runner: `uv run python -m openai_lab.openai_client_runner`
- API: `uv run uvicorn openai_lab.server:app --host 0.0.0.0 --port 8000`
- Docker: `docker compose up --build`

## Notes for Search/Indexing
All core setup and usage documentation is available in Markdown files (`README.md`, `docs/index.md`, `CHANGELOG.md`) so important content is not hidden behind JS-only rendering.
