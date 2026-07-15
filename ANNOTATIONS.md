# Repository annotations

## Overview
- This project is a local LLaVA-style agent starter centered around prompt understanding and local image artifact generation.
- The repo contains a FastAPI UI, a small command-line client, and a lightweight local agent wrapper.

## Key files
- [run.py](run.py): single-command launcher that installs the needed dependencies, prompts for a query, and starts the local web server with uvicorn.
- [app.py](app.py): FastAPI-based local web UI for the LLaVA agent workflow.
- [server.py](server.py): FastAPI entry-point for the local agent API.
- [client.py](client.py): command-line client for sending prompts to the local agent API.
- [model.py](model.py): local prompt-understanding and image-artifact generation wrapper.
- [pyproject.toml](pyproject.toml): Python dependency manifest for the repository.
- [env.example](env.example): example environment variable file for model configuration.

## Launch commands
- Start the UI: `python run.py`
- Skip dependency installation if already present: `python run.py --skip-install`
- Start the API server: `python server.py`
- Send a prompt to the API: `python client.py --prompt "A dreamy skyline at dusk"`
