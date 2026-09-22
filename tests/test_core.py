import io
import unittest

from PIL import Image

from app import validate_image
from inference.multimodal_agent import MultimodalAgent
from utils.cache_manager import CacheManager


class FakeReasoner:
    loads = 0
    calls = 0

    def __init__(self, device, model_name):
        self.device = device
        self.model_name = model_name
        FakeReasoner.loads += 1

    def generate(self, query, context, image):
        FakeReasoner.calls += 1
        return f"{query}:{image.size}"


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (8, 6), "red").save(output, format="PNG")
    return output.getvalue()


class AgentTests(unittest.TestCase):
    def setUp(self):
        FakeReasoner.loads = 0
        FakeReasoner.calls = 0

    def test_lazy_loading_and_response_cache(self):
        agent = MultimodalAgent(device="cpu", model_name="fake", reasoner_factory=FakeReasoner)
        self.assertFalse(agent.is_loaded)
        first = agent.process_bytes(image_bytes(), "What color?")
        second = agent.process_bytes(image_bytes(), "What color?")
        self.assertEqual(first, "What color?:(8, 6)")
        self.assertEqual(first, second)
        self.assertEqual(FakeReasoner.loads, 1)
        self.assertEqual(FakeReasoner.calls, 1)

    def test_empty_question_is_rejected(self):
        agent = MultimodalAgent(device="cpu", model_name="fake", reasoner_factory=FakeReasoner)
        with self.assertRaisesRegex(ValueError, "empty"):
            agent.process_bytes(image_bytes(), "   ")


class CacheTests(unittest.TestCase):
    def test_lru_eviction(self):
        cache = CacheManager(max_items=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")
        cache.put("c", 3)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), 1)


class UploadTests(unittest.TestCase):
    def test_image_validation(self):
        validate_image(image_bytes())
        with self.assertRaisesRegex(ValueError, "valid image"):
            validate_image(b"not an image")


if __name__ == "__main__":
    unittest.main()
