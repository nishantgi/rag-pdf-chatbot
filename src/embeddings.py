import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from .config import EMBEDDING_MODEL_NAME

class EmbeddingModel:
    """Singleton embedding model - load once, use everywhere"""
    
    _model = None
    
    @classmethod
    def get_model(cls):
        if cls._model is None:
            print(f"🔄 Loading embedding model: {EMBEDDING_MODEL_NAME}")
            cls._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("✅ Model loaded! Shape per sentence: 384-dim")
        return cls._model
    
    @classmethod
    def encode_texts(cls, texts: List[str]) -> np.ndarray:
        """Convert list of texts → 2D numpy array (N, 384)"""
        model = cls.get_model()
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        return embeddings


def test_embeddings():
    """Quick test"""
    model = EmbeddingModel.get_model()
    test_texts = [
        "Quick sort worst case complexity",
        "Binary tree maximum nodes at level n"
    ]
    embeddings = EmbeddingModel.encode_texts(test_texts)
    print(f"📊 Test embeddings shape: {embeddings.shape}")
    print(f"🔢 Cosine similarity (should be low): {embeddings[0] @ embeddings[1]:.3f}")


if __name__ == "__main__":
    test_embeddings()
