import io
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from inference.multimodal_agent import MultimodalAgent


MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
WEB_INDEX = Path(__file__).resolve().parent / "web" / "index.html"

app = FastAPI(
    title="Multimodal AI Agent",
    description="Local image question answering over HTTP and WebSocket.",
    version="2.0.0",
)
agent = MultimodalAgent()


class QueryResponse(BaseModel):
    response: str
    model: str
    device: str


def validate_image(data):
    if not data:
        raise ValueError("Image is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Image exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Uploaded data is not a valid image") from error


@app.get("/")
async def root():
    return FileResponse(WEB_INDEX)


@app.get("/health")
async def health():
    return {"status": "ok", **agent.status()}


@app.post("/warmup")
async def warmup():
    try:
        await run_in_threadpool(agent.load)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Model failed to load: {error}") from error
    return {"status": "ready", **agent.status()}


@app.post("/query", response_model=QueryResponse)
async def query_image(
    image: UploadFile = File(...),
    question: str = Form(..., min_length=1, max_length=500),
    context: str | None = Form(default=None, max_length=2000),
):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Use a JPEG, PNG, or WebP image")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    try:
        validate_image(data)
        answer = await run_in_threadpool(agent.process_bytes, data, question, context)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Inference failed: {error}") from error
    return QueryResponse(response=answer, model=agent.model_name, device=agent.device)


@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    question = websocket.query_params.get("question", "What is happening in this scene?")
    try:
        while True:
            data = await websocket.receive_bytes()
            try:
                validate_image(data)
                answer = await run_in_threadpool(agent.process_bytes, data, question, None)
                await websocket.send_json({"response": answer})
            except (ValueError, UnidentifiedImageError) as error:
                await websocket.send_json({"error": str(error)})
            except Exception as error:
                await websocket.send_json({"error": f"Inference failed: {error}"})
    except WebSocketDisconnect:
        return
