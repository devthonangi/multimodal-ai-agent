import os


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RetrievalEngine:
    """Build a small in-memory FAISS index for each image request."""

    def __init__(self, model_name=None):
        from langchain_huggingface import HuggingFaceEmbeddings

        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    def retrieve(self, query, sources, limit=3):
        from langchain_community.vectorstores import FAISS

        documents = []
        for source in sources:
            if not source:
                continue
            documents.extend(chunk.strip() for chunk in source.splitlines() if chunk.strip())
        if not documents:
            return ""
        store = FAISS.from_texts(documents, self.embeddings)
        matches = store.similarity_search(query, k=min(limit, len(documents)))
        return "\n".join(document.page_content for document in matches)
