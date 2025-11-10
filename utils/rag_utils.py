from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGRetriever:
    def __init__(self, device):
        print("[INFO] Initializing FAISS retriever...")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        self.index = faiss.IndexFlatL2(384)
        self.documents = [
            "This image may depict a natural landscape.",
            "There are humans interacting with objects.",
            "It might contain animals, vehicles, or text elements."
        ]
        self.embeddings = self.embedder.encode(self.documents)
        self.index.add(np.array(self.embeddings).astype("float32"))

    def retrieve(self, vision_features, query):
        q_vec = self.embedder.encode([query])
        D, I = self.index.search(np.array(q_vec).astype("float32"), 1)
        return self.documents[I[0][0]]
