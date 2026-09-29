# Multimodal Vision-Language Agent

**PyTorch · LLaVA · BLIP-2 · LangChain · FAISS · FastAPI · MLX**

- Built a multimodal AI system integrating LLaVA, BLIP-2, LangChain, and FAISS to enable image understanding, visual question answering, and context-aware reasoning through a scalable FastAPI inference service.
- Implemented GPU-accelerated inference and retrieval pipelines, enabling low-latency multimodal interactions through semantic search and vector-based retrieval.

## Runtime support

The agent automatically selects an inference backend for the available hardware:

- **NVIDIA CUDA:** LLaVA and BLIP-2 with PyTorch, LangChain, and FAISS.
- **Apple silicon (optional):** Qwen2.5-VL 3B through MLX for conversational local image understanding on Mac.
- **CPU fallback:** BLIP VQA for lightweight image questions.

The full LLaVA and BLIP-2 pipeline is too large for the limited memory available on a Jetson Nano. Use the lightweight CUDA configuration on that device:

```env
AGENT_DEVICE=cuda
AGENT_MODEL=Salesforce/blip-vqa-base
```

Jetson deployments require the PyTorch build supplied for the installed NVIDIA JetPack version.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. Model files are downloaded on the first request and reused from the local cache afterward.
