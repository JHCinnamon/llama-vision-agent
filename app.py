"""A lightweight FastAPI UI for an Ollama-powered local agent.

The server serves a single-page form, accepts a user prompt, and uses an
Ollama-backed Flux model to generate an image artifact. A lightweight planner
model helps refine the prompt before the image model is invoked.
"""

from __future__ import annotations

import os
import re
from html import escape
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse
from tqdm import tqdm
from PIL import Image, ImageDraw

from model import LocalLLaVAAgent

REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Ollama LLaVA Agent")
agent = LocalLLaVAAgent()


def _slugify(text: str) -> str:
    """Convert a prompt into a safe filename fragment."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned or "prompt"


def build_image(prompt: str) -> Path:
    """Create a simple generated image for the supplied prompt."""
    image = Image.new("RGB", (1024, 768), color=(15, 23, 42))
    draw = ImageDraw.Draw(image)

    words = [word for word in re.split(r"\s+", prompt) if word]
    color = (120 + (len(words) * 7) % 135, 80 + (len(words) * 13) % 120, 200)
    draw.ellipse((120, 90, 900, 650), fill=color)
    draw.rectangle((180, 180, 760, 560), outline=(255, 255, 255), width=8)

    accent = (255, 215, 0)
    draw.rectangle((220, 240, 720, 500), fill=accent)
    draw.text((260, 290), "Prompt-driven image", fill=(15, 23, 42))
    draw.text((260, 360), prompt[:90], fill=(15, 23, 42))

    if words:
        first_word = words[0].title()
        draw.text((260, 430), f"Theme: {first_word}", fill=(15, 23, 42))

    output_path = OUTPUTS_DIR / f"{_slugify(prompt)}.png"
    image.save(output_path)
    return output_path


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """Serve the form-based UI."""
    initial_query = os.getenv("INITIAL_QUERY", "A dreamy sunset over a calm lake")
    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
        <title>Ollama LLaVA Agent</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
            form {{ max-width: 640px; margin: 0 auto; background: #111827; padding: 1.5rem; border-radius: 12px; }}
            textarea, button {{ width: 100%; padding: 0.8rem; margin-top: 0.75rem; border-radius: 8px; border: 1px solid #334155; }}
            textarea {{ min-height: 110px; resize: vertical; background: #0f172a; color: #f8fafc; }}
            button {{ background: #2563eb; color: white; cursor: pointer; }}
            img {{ max-width: 100%; margin-top: 1.5rem; border-radius: 12px; }}
            .progress {{ width: 100%; height: 12px; background: #334155; border-radius: 999px; overflow: hidden; margin-top: 0.6rem; }}
            .progress > div {{ height: 100%; width: 0%; background: linear-gradient(90deg, #38bdf8, #2563eb); animation: pulse 1.2s ease-in-out infinite alternate; transition: width 0.3s ease; }}
            @keyframes pulse {{ from {{ opacity: 0.8; }} to {{ opacity: 1; }} }}
        </style>
    </head>
    <body>
        <form action=\"/generate\" method=\"post\" enctype=\"multipart/form-data\">
            <h1>Ollama LLaVA Agent</h1>
            <p>Enter an image-generation prompt and the app will use Ollama's Flux model to create and display the result locally.</p>
            <p id=\"status\">Preparing generation...</p>
            <div class=\"progress\"><div id=\"progressBar\"></div></div>
            <textarea name=\"query\" required>{initial_query}</textarea>
            <button type=\"submit\">Generate image</button>
            <script>
                const form = document.querySelector('form');
                const status = document.getElementById('status');
                const progressBar = document.getElementById('progressBar');
                let progressValue = 0;
                const interval = window.setInterval(() => {{
                    progressValue = (progressValue + 8) % 100;
                    progressBar.style.width = progressValue + '%';
                    status.textContent = 'Generating image with Ollama... ' + progressValue + '%';
                }}, 300);
                form.addEventListener('submit', () => {{
                    window.clearInterval(interval);
                    progressBar.style.width = '100%';
                    status.textContent = 'Generating image with Ollama... 100%';
                }});
            </script>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/generate", response_class=HTMLResponse)
def generate(query: str = Form(...)) -> HTMLResponse:
    """Generate a local PNG image and show it in the browser."""
    try:
        output_path = agent.generate_artifact(query)
    except Exception as exc:
        output_path = None
        agent.last_error = str(exc)

    image_url = f"/images/{output_path.name}" if output_path else ""
    try:
        interpretation = agent.describe_prompt(query)
    except Exception as exc:
        interpretation = f"Unable to build an interpretation: {exc}"

    backend = agent.last_backend or "unknown"
    error_block = f"<p><strong>Warning:</strong> {escape(agent.last_error)}</p>" if agent.last_error else ""
    image_block = f'<img src="{image_url}" alt="Generated image" />' if output_path else "<p>No image was produced.</p>"

    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
        <title>Result</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
            .card {{ max-width: 900px; margin: 0 auto; background: #111827; padding: 1.5rem; border-radius: 12px; }}
            img {{ max-width: 100%; margin-top: 1rem; border-radius: 12px; }}
        </style>
    </head>
    <body>
        <div class=\"card\">
            <h1>Generated image</h1>
            <p><strong>Prompt:</strong> {escape(query)}</p>
            <p><strong>Agent interpretation:</strong> {escape(interpretation)}</p>
            <p><strong>Runtime backend:</strong> {escape(backend)}</p>
            {error_block}
            <p>Saved to: {output_path or 'not available'}</p>
            <p>The image below was generated locally and saved in the outputs folder.</p>
            {image_block}
            <p><a href=\"/\">Generate another image</a></p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/images/{filename}")
def serve_image(filename: str) -> FileResponse:
    """Serve the generated image files from the outputs directory."""
    return FileResponse(OUTPUTS_DIR / filename)


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health-check endpoint for local development."""
    return {"status": "ok"}
