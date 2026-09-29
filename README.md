# Multimodal Vision-Language Agent

Local image question answering with conversational follow-ups.

## High-level view

```text
Image + question + recent history
                |
             FastAPI
                |
        Hardware detection
          /      |      \
      CUDA      MLX     CPU
   LLaVA +    Qwen2.5   BLIP
    BLIP-2       VL      VQA
          \      |      /
       Grounded response
                |
        Browser conversation
```

- FastAPI validates the image and handles HTTP requests.
- The agent selects a model backend based on the available hardware.
- Recent turns remain scoped to the selected image for follow-up questions.
- FAISS supports retrieval on the CUDA pipeline; all backends use response caching.

## Models

- NVIDIA CUDA: LLaVA + BLIP-2
- Apple silicon: Qwen2.5-VL 3B with MLX
- CPU: BLIP VQA

For Jetson Nano:

```env
AGENT_DEVICE=cuda
AGENT_MODEL=Salesforce/blip-vqa-base
```

Use the PyTorch build compatible with your JetPack version.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.
