import torch
from inference.vision_encoder import VisionEncoder
from inference.text_reasoner import TextReasoner
from utils.rag_utils import RAGRetriever
from utils.cache_manager import CacheManager

class MultimodalAgent:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        print(f"[INFO] Initializing Multimodal Agent on {device}")
        self.device = device
        self.vision_encoder = VisionEncoder(device)
        self.text_reasoner = TextReasoner(device)
        self.retriever = RAGRetriever(device)
        self.cache = CacheManager()

    def process_query(self, image_path, query):
        # Step 1: Compute or retrieve cached visual embedding
        vision_features = self.cache.get_or_compute(
            image_path, lambda: self.vision_encoder.encode(image_path)
        )

        # Step 2: Retrieve relevant textual context
        context = self.retriever.retrieve(vision_features, query)

        # Step 3: Generate multimodal response
        response = self.text_reasoner.generate(query, context, vision_features)
        return response
