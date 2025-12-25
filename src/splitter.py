from typing import List, Dict
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    cleaned = text.replace("\r", " ").replace("\n", " ")
    cleaned = " ".join(cleaned.split())
    
    chunks = []
    text_length = len(cleaned)
    start = 0
    chunk_id = 0
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = cleaned[start:end].strip()
        
        if chunk_text:
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "start": start,
                "end": end,
            })
            chunk_id += 1
        
        start = start + chunk_size - chunk_overlap
        if start <= 0:
            start = end
    
    return chunks
