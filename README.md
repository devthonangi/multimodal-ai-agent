# Multimodal AI Agent

A local image-question-answering service with a FastAPI API, live WebSocket video support, lazy model loading, response caching, and CPU, Apple Silicon, or CUDA execution.

## What changed in v2

- The API starts immediately instead of loading multiple large models during import.
- The default model is the practical `Salesforce/blip-vqa-base` rather than BLIP-2 FLAN-T5-XL.
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

Models are downloaded from Hugging Face on first use.

## Start the API

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API.

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

- `AGENT_DEVICE=auto|cpu|mps|cuda` selects the inference device.
- `AGENT_MODEL=Salesforce/blip-vqa-base` selects a compatible BLIP VQA model.
- `MAX_UPLOAD_MB=10` controls the upload limit.

Example for Apple Silicon:

```bash
AGENT_DEVICE=mps uvicorn app:app
```

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
