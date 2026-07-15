"""Entry-point for the local LLaVA agent UI.

This script makes the repository runnable from a single command. It checks for the
small set of Python dependencies required by the web interface, installs any
missing packages with pip, prompts the user for a text query, and starts a local
FastAPI server via uvicorn.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import requests
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def _module_available(module_name: str) -> bool:
    """Return True when the requested module is importable."""
    return importlib.util.find_spec(module_name) is not None


def install_dependencies(packages: Sequence[str]) -> None:
    """Install any missing Python packages with a simple progress bar."""
    missing = [package for package in packages if not _module_available(package)]
    if not missing:
        return

    print("Installing the Ollama agent dependencies...")
    from tqdm import tqdm

    for package in tqdm(missing, desc="packages", unit="pkg"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def parse_args() -> argparse.Namespace:
    """Parse the command-line flags for the launcher."""
    parser = argparse.ArgumentParser(description="Start the local Ollama LLaVA agent UI")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for uvicorn")
    parser.add_argument("--port", type=int, default=8000, help="Port for the web UI")
    parser.add_argument("--query", default=None, help="Optional prompt to preload in the UI")
    parser.add_argument(
        "--allow-flux-runtime-fail",
        action="store_true",
        help="Allow startup when FLUX runtime probing fails (use only with a configured secondary backend).",
    )
    return parser.parse_args()


def ask_for_query(default_query: str) -> str:
    """Prompt the user for a text query and fall back to a default value."""
    prompt = input(f"Enter a prompt for the LLaVA agent [{default_query}]: ").strip()
    return prompt or default_query


def _normalize_model_name(model_name: str) -> str:
    """Return a tagless base model name for matching against Ollama listings."""
    return model_name.split(":", 1)[0]


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def pull_ollama_model(model_name: str) -> None:
    """Pull a model from Ollama and surface a helpful error if the CLI is missing."""
    if shutil.which("ollama") is None:
        raise RuntimeError("The 'ollama' CLI was not found on PATH. Install Ollama and make sure it is available before running the app.")

    print(f"Pulling Ollama model: {model_name}")
    try:
        subprocess.check_call(["ollama", "pull", model_name])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to pull Ollama model '{model_name}'.") from exc


def probe_flux_runtime(host: str, image_model: str) -> None:
    """Ensure the configured FLUX image model can actually generate on this machine."""
    payload = {
        "model": image_model,
        "prompt": "runtime probe image",
        "stream": False,
        "options": {"size": "256x256"},
    }
    response = requests.post(f"{host}/api/generate", json=payload, timeout=180)
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("FLUX runtime probe returned a non-JSON response from Ollama.") from exc

    if response.status_code >= 400 or data.get("error"):
        message = data.get("error") if isinstance(data, dict) else response.text
        raise RuntimeError(f"FLUX runtime probe failed: {message}")

    if not data.get("image") and not data.get("images"):
        raise RuntimeError("FLUX runtime probe completed but did not return image data.")


def probe_secondary_backend() -> None:
    """Check optional secondary backend availability when configured."""
    backend = os.getenv("SECONDARY_IMAGE_BACKEND", "").strip().lower()
    if not backend:
        return

    if backend != "automatic1111":
        raise RuntimeError(f"Unsupported secondary backend '{backend}'. Supported: automatic1111")

    base_url = os.getenv("SECONDARY_IMAGE_BACKEND_URL", "http://127.0.0.1:7860").rstrip("/")
    try:
        response = requests.get(f"{base_url}/sdapi/v1/sd-models", timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Secondary backend '{backend}' is configured but not reachable at {base_url}. "
            "Start Automatic1111 with --api or update SECONDARY_IMAGE_BACKEND_URL."
        ) from exc


def ensure_ollama_ready(allow_flux_runtime_fail: bool) -> None:
    """Check that Ollama is reachable and pull the required models when they are missing."""
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    models = [os.getenv("IMAGE_MODEL", "x/flux2-klein"), os.getenv("PLANNER_MODEL", "qwen2.5:3b")]

    print("Checking Ollama availability...")
    from tqdm import tqdm

    try:
        response = requests.get(f"{host}/api/tags", timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama is not reachable at {host}. Start Ollama first and make sure it is running. ({exc})") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Ollama returned an unexpected response. Check that the service is healthy.") from exc

    available = {_normalize_model_name(item.get("name", "")) for item in data.get("models", [])}
    missing = [model for model in models if _normalize_model_name(model) not in available]

    for _ in tqdm(range(1), desc="ollama", unit="check"):
        pass

    if missing:
        print("The following Ollama models are missing and will be pulled automatically:")
        for model in missing:
            print(f" - {model}")

        for model in tqdm(missing, desc="pulling models", unit="model"):
            pull_ollama_model(model)

        response = requests.get(f"{host}/api/tags", timeout=15)
        response.raise_for_status()
        data = response.json()
        available = {_normalize_model_name(item.get("name", "")) for item in data.get("models", [])}
        still_missing = [model for model in models if _normalize_model_name(model) not in available]
        if still_missing:
            raise RuntimeError("Ollama could not verify the following models after pulling: " + ", ".join(still_missing))

    probe_secondary_backend()

    require_flux_runtime = _as_bool(os.getenv("REQUIRE_FLUX_RUNTIME", "1")) and not allow_flux_runtime_fail
    try:
        probe_flux_runtime(host, models[0])
    except Exception as exc:
        if require_flux_runtime:
            raise RuntimeError(
                "FLUX runtime is required but failed probing. "
                "Fix Ollama runtime or start with --allow-flux-runtime-fail and configure SECONDARY_IMAGE_BACKEND. "
                f"Details: {exc}"
            ) from exc
        print(f"Warning: FLUX runtime probe failed, continuing due to override. Details: {exc}")

    print("Ollama check passed. Required models are available.")


def main() -> None:
    """Run the installation flow and then launch the web application."""
    args = parse_args()

    if not args.skip_install:
        install_dependencies(["fastapi", "uvicorn", "tqdm", "pillow", "python-multipart", "requests", "ollama", "python-dotenv"])

    ensure_ollama_ready(allow_flux_runtime_fail=args.allow_flux_runtime_fail)

    default_query = args.query or "A dreamy sunset over a calm lake"
    user_query = ask_for_query(default_query)
    os.environ["INITIAL_QUERY"] = user_query

    print(f"Launching the Ollama LLaVA agent UI at http://{args.host}:{args.port}")
    import uvicorn

    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
