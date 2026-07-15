"""Ollama-powered local agent wrapper.

The workflow now uses an Ollama-backed image generation model for creating images,
while a lightweight prompt-planning model helps refine the prompt before calling
that generator. The UI remains local and saves each generated image in the outputs
folder.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import ollama
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


class LocalLLaVAAgent:
    def __init__(self) -> None:
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.image_model = os.getenv("IMAGE_MODEL", "x/flux2-klein")
        self.planner_model = os.getenv("PLANNER_MODEL", "qwen2.5:3b")
        self.secondary_backend = os.getenv("SECONDARY_IMAGE_BACKEND", "").strip().lower()
        self.secondary_backend_url = os.getenv("SECONDARY_IMAGE_BACKEND_URL", "http://127.0.0.1:7860")
        self.last_error: Optional[str] = None
        self.last_backend = ""

    @staticmethod
    def _decode_image_payload(payload: object) -> bytes:
        """Decode an Ollama image payload into raw bytes."""
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, str):
            cleaned = payload.strip()
            if cleaned.startswith("data:image") and "," in cleaned:
                cleaned = cleaned.split(",", 1)[1]
            try:
                return base64.b64decode(cleaned, validate=True)
            except binascii.Error as exc:
                raise ValueError("The Ollama response did not include a valid base64-encoded image.") from exc
        raise TypeError("Unsupported image payload type from Ollama.")

    def _generate_with_secondary_backend(self, prompt: str) -> bytes:
        """Generate an image from a secondary local backend when configured."""
        if self.secondary_backend != "automatic1111":
            raise RuntimeError("No supported secondary image backend is configured.")

        endpoint = f"{self.secondary_backend_url.rstrip('/')}/sdapi/v1/txt2img"
        payload = {
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "steps": 28,
            "sampler_name": "Euler a",
        }
        response = requests.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        images = data.get("images") or []
        if not images:
            raise RuntimeError("Secondary backend did not return image data.")
        return self._decode_image_payload(images[0])

    def plan_prompt(self, prompt: str) -> str:
        """Use a lightweight planning model to improve the image-generation prompt."""
        payload = {
            "model": self.planner_model,
            "prompt": f"Turn this into a concise image-generation prompt: {prompt}",
            "stream": False,
            "options": {"num_predict": 80},
        }
        try:
            response = requests.post(f"{self.ollama_host}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", prompt).strip() or prompt
        except Exception as exc:  # pragma: no cover - best-effort local planning
            self.last_error = str(exc)
            return prompt

    def describe_prompt(self, prompt: str) -> str:
        """Return the agent-style interpretation of the prompt."""
        planned = self.plan_prompt(prompt)
        if self.last_error:
            return f"Using Ollama planning fallback. Prompt refined with: {planned}"
        return f"Ollama prompt plan: {planned}"

    def generate_artifact(self, prompt: str) -> Path:
        """Generate a local image artifact from the prompt using the Ollama Flux model."""
        self.last_error = None
        self.last_backend = ""
        planned_prompt = self.plan_prompt(prompt)
        try:
            response = ollama.generate(
                model=self.image_model,
                prompt=planned_prompt,
                stream=False,
                options={"size": "1024x1024"},
            )
            image_base64 = response.get("image")
            if not image_base64:
                raise ValueError("The Ollama model did not return an image.")

            image_bytes = base64.b64decode(image_base64)
            slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-") or "prompt"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = OUTPUTS_DIR / f"{slug}-{timestamp}.png"
            output_path.write_bytes(image_bytes)
            self.last_backend = f"ollama:{self.image_model}"
            return output_path
        except Exception as exc:  # pragma: no cover - optional secondary backend if FLUX runtime fails
            self.last_error = str(exc)
            if not self.secondary_backend:
                raise RuntimeError(f"Image generation failed via Ollama FLUX: {exc}") from exc

            try:
                image_bytes = self._generate_with_secondary_backend(planned_prompt)
            except Exception as secondary_exc:
                raise RuntimeError(
                    f"Image generation failed via Ollama FLUX and secondary backend: {exc}; {secondary_exc}"
                ) from secondary_exc

            slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-") or "prompt"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = OUTPUTS_DIR / f"{slug}-{timestamp}.png"
            output_path.write_bytes(image_bytes)
            self.last_backend = f"secondary:{self.secondary_backend}"
            return output_path