from pathlib import Path
import numpy as np
from .loader import load_pdf_paths, pdf_to_text
from .splitter import split_text_into_chunks
from .embeddings import EmbeddingModel
from .vector_store import VectorStore

def build_full_index():
    """PDF → chunks → embeddings → FAISS index (complete indexing pipeline)"""
    print("🚀 Building complete RAG index...")
    
    # 1. Load + chunk
    pdf_paths = load_pdf_paths()
    if not pdf_paths:
        print("❌ No PDFs!")
        return
    
    pdf_path = pdf_paths[0]
    print(f"📄 Processing: {pdf_path.name}")
    
    text = pdf_to_text(pdf_path)
    chunks = split_text_into_chunks(text)
    print(f"✂️  {len(chunks)} chunks created")
    
    # 2. Embeddings
    chunk_texts = [c["text"] for c in chunks]
    embeddings = EmbeddingModel.encode_texts(chunk_texts)
    print(f"🔢 {embeddings.shape[0]} embeddings ({embeddings.shape[1]}-dim)")
    
    # 3. Build index
    store = VectorStore()
    store.build_index(embeddings, chunks)
    
    # 4. Test search
    query = "Quick sort worst case complexity"
    query_emb = EmbeddingModel.encode_texts([query])[0]
    results = store.search(query_emb)
    
    print("\n🔍 Test query:", query)
    for i, (text, score) in enumerate(results):
        print(f"\n--- Top {i+1} (score: {score:.3f}) ---")
        print(text[:200] + "...")

if __name__ == "__main__":
    build_full_index()
