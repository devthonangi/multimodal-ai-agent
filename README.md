# Multimodal Vision-Language Agent

Local image question answering with conversational follow-ups.

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
