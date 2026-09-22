import hashlib
import io
import os
from pathlib import Path
from threading import RLock

from PIL import Image

from inference.text_reasoner import DEFAULT_MODEL, TextReasoner
from utils.cache_manager import CacheManager


def select_device(requested="nvidia-nim"):
    return requested or "nvidia-nim"


class MultimodalAgent:
    """Lazy-loading, thread-safe image question-answering service."""

    def __init__(self, device=None, model_name=None, cache_size=128, reasoner_factory=TextReasoner):
        self.device = select_device(device or "nvidia-nim")
        self.provider = "nvidia-nim"
        self.model_name = model_name or os.getenv("AGENT_MODEL", DEFAULT_MODEL)
        self.cache = CacheManager(cache_size)
        self._reasoner_factory = reasoner_factory
        self._reasoner = None
        self._load_lock = RLock()
        self._inference_lock = RLock()

    @property
    def is_loaded(self):
        return self._reasoner is not None

    def load(self):
        if self._reasoner is None:
            with self._load_lock:
                if self._reasoner is None:
                    self._reasoner = self._reasoner_factory(self.device, self.model_name)
        return self

    def process_query(self, image_path, query, context=None):
        with Image.open(Path(image_path)) as image:
            return self.process_image(image.convert("RGB"), query, context)

    def process_bytes(self, image_bytes, query, context=None):
        with Image.open(io.BytesIO(image_bytes)) as image:
            return self.process_image(image.convert("RGB"), query, context)

    def process_image(self, image, query, context=None):
        query = query.strip()
        if not query:
            raise ValueError("Question cannot be empty")
        image = image.convert("RGB")
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        key = (digest, query, context or "", self.provider, self.model_name)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        self.load()
        with self._inference_lock:
            answer = self._reasoner.generate(query, context, image)
        self.cache.put(key, answer)
        return answer

    def status(self):
        return {
            "loaded": self.is_loaded,
            "device": self.device,
            "provider": self.provider,
            "model": self.model_name,
            "configured": bool(os.getenv("NVIDIA_API_KEY")),
            "cached_responses": len(self.cache),
        }
