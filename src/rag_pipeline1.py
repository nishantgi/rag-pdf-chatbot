import os
import requests
from dotenv import load_dotenv
from typing import List
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .config import TOP_K

load_dotenv()

def groq_llm(query: str, context: str) -> str:
    """Bulletproof Groq API"""
    api_key = os.getenv("GROQ_API_KEY")
    print(f"🔑 API key valid: {len(api_key) if api_key else 0} chars")
    
    if not api_key:
        return "❌ No API key"
    
    # EXACT Groq working models (Dec 2025)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": "Answer from context only. Be concise."},
            {"role": "user", "content": f"Q: {query}\n\n{context}\n\nAnswer:"}
        ],
        "max_tokens": 300,
        "temperature": 0.1
    }
    
    # TRY DIFFERENT MODELS (one will work)
    # models = ["llama3-8b-8192", "llama-3.2-1b-preview", "gemma2-9b-it"]
    models = ["llama-3.1-8b-instant","llama3-groq-8b-8192-tool-use-preview","mixtral-8x7b-32768","gemma-7b-it"]
    
    for model in models:
        payload["model"] = model
        print(f"🔄 Trying model: {model}")
        
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload, timeout=10
            )
            print(f"📡 Status: {resp.status_code}")
            
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                print(f"❌ {model}: {resp.text[:100]}")
                
        except Exception as e:
            print(f"❌ {model}: {str(e)[:50]}")
    
    return "❌ All models failed"

def format_context(chunks: List) -> str:
    return "\n".join([f"[{i+1}] {text[:200]}..." for i, (text, _) in enumerate(chunks)])

def rag_query(question: str, store: VectorStore):
    query_emb = EmbeddingModel.encode_texts([question])[0]
    chunks = store.search(query_emb, top_k=TOP_K)
    context = format_context(chunks)
    answer = groq_llm(question, context)
    return {"question": question, "answer": answer}

if __name__ == "__main__":
    store = VectorStore.load_index()
    if store:
        result = rag_query("Quick sort worst case complexity", store)
        print(f"\n🔍 Q: {result['question']}")
        print(f"\n🤖 A: {result['answer']}")
    else:
        print("❌ No index!")
