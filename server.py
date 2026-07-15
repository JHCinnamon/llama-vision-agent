"""A lightweight FastAPI server for the local LLaVA agent workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import LocalLLaVAAgent

app = FastAPI(title="LLaVA Agent API")
agent = LocalLLaVAAgent()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    prompt: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
def generate(req: PromptRequest) -> dict[str, str]:
    output_path = agent.generate_artifact(req.prompt)
    return {
        "prompt": req.prompt,
        "interpretation": agent.describe_prompt(req.prompt),
        "image_path": str(output_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the LLaVA agent API server")
    parser.add_argument("--port", type=int, default=8002, help="Port number to run the server on")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port)