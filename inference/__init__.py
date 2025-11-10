"""
inference package

Contains core modules for the multimodal reasoning agent:
- multimodal_agent.py : orchestrates vision + text reasoning
- vision_encoder.py   : extracts visual embeddings (CLIP)
- text_reasoner.py    : generates answers using BLIP-2 or LLaVA
"""

from .multimodal_agent import MultimodalAgent
from .vision_encoder import VisionEncoder
from .text_reasoner import TextReasoner

__all__ = ["MultimodalAgent", "VisionEncoder", "TextReasoner"]
