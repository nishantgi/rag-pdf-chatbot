import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple
from .config import FAISS_INDEX_PATH, CHUNKS_META_PATH

class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.chunks_metadata = []
    
    def build_index(self, embeddings: np.ndarray, metadata: List[dict]):
        """Create FAISS index from embeddings + metadata"""
        assert embeddings.shape[0] == len(metadata), "Embeddings and metadata mismatch!"
        
        # Normalize embeddings (for cosine similarity)
        faiss.normalize_L2(embeddings)
        
        # IndexFlatIP = exact cosine similarity (inner product on unit vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.chunks_metadata = metadata
        self._save_index()
        print(f"✅ Index built: {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Tuple[str, float]]:
        """Search similar chunks: returns (chunk_text, similarity_score)"""
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        scores, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), top_k)
        results = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks_metadata):
                chunk = self.chunks_metadata[idx]
                results.append((chunk["text"], float(score)))
        
        return results
    
    def _save_index(self):
        """Save index + metadata to disk"""
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(CHUNKS_META_PATH, 'w') as f:
            json.dump(self.chunks_metadata, f)
    
    @classmethod
    def load_index(cls):
        """Load existing index"""
        store = cls()
        if FAISS_INDEX_PATH.exists():
            store.index = faiss.read_index(str(FAISS_INDEX_PATH))
            with open(CHUNKS_META_PATH, 'r') as f:
                store.chunks_metadata = json.load(f)
            print(f"✅ Loaded index: {store.index.ntotal} vectors")
            return store
        return None


if __name__ == "__main__":
    print("VectorStore ready!")
