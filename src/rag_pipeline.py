from typing import List, Dict
import json
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .config import TOP_K

def format_context(chunks: List[Dict]) -> str:
    """Convert retrieved chunks → LLM-friendly context"""
    context = []
    for i, (text, score) in enumerate(chunks):
        context.append(f"Source {i+1} (relevance: {score:.3f}):\n{text}\n")
    return "\n---\n".join(context)

def mock_llm(query: str, context: str) -> str:
    """Mock LLM - formats retrieved chunks into natural answer"""
    prompt = f"""
    Answer the question based ONLY on the provided context.
    If answer not in context, say "Not found in document".

    Question: {query}

    Context:
    {context}

    Answer:"""
    
    # Mock LLM logic: extract key info from top chunks
    answer = f"Based on document:\n\n"
    
    for i, line in enumerate(context.split('\n')[:10]):  # Top sources
        if 'O(n^2)' in line or 'worst-case' in line.lower():
            answer += f"• {line.strip()}\n"
    
    if "Not found" not in answer:
        answer += f"\nSources: Top-{TOP_K} chunks from document."
    else:
        answer = "Answer not found in the provided document."
    
    return answer

def rag_query(question: str, store: VectorStore) -> Dict:
    """
    Full RAG pipeline:
    1. Question → embedding
    2. FAISS search → top-K chunks
    3. Context → mock LLM → answer
    """
    # 1. Embed query
    query_emb = EmbeddingModel.encode_texts([question])[0]
    
    # 2. Retrieve
    chunks = store.search(query_emb, top_k=TOP_K)
    context = format_context(chunks)
    
    # 3. Generate
    answer = mock_llm(question, context)
    
    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": len(chunks),
        "top_scores": [score for _, score in chunks],
        "context_preview": context[:500] + "..."
    }


if __name__ == "__main__":
    # Load index
    store = VectorStore.load_index()
    if not store:
        print("❌ No index found. Run demo_rag_index.py first!")
    else:
        queries = [
            "Quick sort worst case complexity",
            "doubly linked list insert end",
            "binary tree maximum nodes"
        ]
        
        for query in queries:
            result = rag_query(query, store)
            print(f"\n🔍 Q: {result['question']}")
            print(f"🤖 A: {result['answer'][:100]}...")
            print(f"📊 Retrieved: {result['retrieved_chunks']} chunks")
