# Multimodal AI Agent

An image-question-answering app powered by NVIDIA NIM, with a simple browser interface, FastAPI API, live WebSocket video support, lazy initialization, and response caching.

## What changed in v2

- The API starts immediately instead of loading multiple large models during import.
- The default model is NVIDIA Nemotron 3 Nano Omni 30B A3B Reasoning, producing detailed multimodal answers instead of one-word labels.
- Image uploads are validated, size-limited, and processed in memory.
- Blocking inference runs outside FastAPI's event loop.
- Repeated image questions use a bounded, thread-safe cache.
- HTTP and WebSocket inference share one implementation.
- Health and model warmup endpoints expose runtime state.
- Live camera/video inference no longer writes every frame to disk.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a free NVIDIA API key at [build.nvidia.com](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning), then configure it:

```bash
cp .env.example .env
```

Open `.env` and replace `nvapi-your-key-here` with your key. The key stays server-side and is never sent to the browser.

## Start the API

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` for the simple image-question interface. Developer API documentation remains available at `http://127.0.0.1:8000/docs`.

The server starts without loading the model. Either send the first query or preload it:

```bash
curl -X POST http://127.0.0.1:8000/warmup
```

Ask a question about an image:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -F 'image=@/path/to/image.jpg' \
  -F 'question=What is in this image?'
```

Check runtime state:

```bash
curl http://127.0.0.1:8000/health
```

## Live video

For a local camera or video file:

```bash
python live/live_agent.py --source 0 --fps 0.5
python live/live_agent.py --source sample.mp4 --question 'What is happening?'
```

Browser and mobile clients can send encoded JPEG, PNG, or WebP frames to:

```text
ws://127.0.0.1:8000/ws/video?question=What%20is%20happening?
```

The server responds with JSON: `{"response": "..."}`.

## Configuration

- `NVIDIA_API_KEY` authenticates with NVIDIA NIM.
- `NVIDIA_API_URL` defaults to NVIDIA's hosted NIM endpoint and can point to a self-hosted NIM.
- `AGENT_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` selects the NVIDIA model.
- `MAX_UPLOAD_MB=10` controls the upload limit.

## Test

The tests use a fake reasoner and do not download model weights:

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py inference utils live tests
```

Run a real-image benchmark after the model has been downloaded:

```bash
python benchmark.py /path/to/image.jpg 'What is in this image?' --expected dog
```
