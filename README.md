# LLaVA Local Agent

This repository is now centered on a local LLaVA-style agent workflow. A prompt-understanding step is used to interpret the user request, and the project produces a simple local image artifact from that prompt.

## What this repo now provides
- A local FastAPI UI for entering prompts and viewing generated outputs
- A local prompt-understanding layer for turning a request into a richer instruction
- A local image-artifact workflow that saves a PNG file in the outputs directory
- A runnable launcher for installing the needed packages and starting the app

## Setup
1. Install Python 3.10+ and Poetry.
2. Install the dependencies:

```bash
poetry install
```

3. If you want to swap in a real local LLaVA checkpoint later, point the configuration in [model.py](model.py) at that model.

## Run

```bash
python run.py
```

You can skip dependency installation if everything is already available:

```bash
python run.py --skip-install
```

## API usage

You can also run the API directly:

```bash
python server.py
```

And then call it with:

```bash
python client.py --prompt "A dreamy skyline at dusk"
```

## Notes
- Primary image runtime is Ollama FLUX via `x/flux2-klein`.
- Startup now probes FLUX runtime; by default the launcher fails fast if FLUX cannot generate.
- Optional secondary real-image backend is supported through Automatic1111 WebUI API.

## Runtime Controls

### Strict FLUX requirement (default)
- Default behavior: startup fails if FLUX runtime probe fails.
- Env flag: `REQUIRE_FLUX_RUNTIME=1` (default)

### Allow startup when FLUX probe fails
- CLI override: `python run.py --allow-flux-runtime-fail`
- Recommended only if secondary backend is configured.

### Secondary backend (Automatic1111)
Set environment variables before launch:

```bash
SECONDARY_IMAGE_BACKEND=automatic1111
SECONDARY_IMAGE_BACKEND_URL=http://127.0.0.1:7860
```

When configured, generation falls back to the secondary backend only if FLUX fails at request time.

### Windows quick enable after FLUX probe failure
1. Start Automatic1111 WebUI with API enabled (`--api`) on port `7860`.
2. In your project `.env`, set:

```bash
REQUIRE_FLUX_RUNTIME=0
SECONDARY_IMAGE_BACKEND=automatic1111
SECONDARY_IMAGE_BACKEND_URL=http://127.0.0.1:7860
```

3. Start the app with:

```bash
python run.py --allow-flux-runtime-fail
```

4. Submit a prompt in the UI and check `Runtime backend` on the result page.
