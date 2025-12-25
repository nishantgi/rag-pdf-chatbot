from pathlib import Path
from .loader import load_pdf_paths, pdf_to_text
from .splitter import split_text_into_chunks

def demo():
    pdf_paths = load_pdf_paths()
    if not pdf_paths:
        print("No PDFs found. Please add a PDF into the data/pdfs folder.")
        return

    pdf_path = pdf_paths[0]
    print("📄 Using PDF:", pdf_path.name)

    full_text = pdf_to_text(pdf_path)
    print("📊 Full text length (chars):", len(full_text))

    chunks = split_text_into_chunks(full_text)
    print("✂️  Number of chunks:", len(chunks))
    print("📏  Avg chunk size:", sum(len(c["text"]) for c in chunks) // len(chunks))

    # Show first 3 chunks
    print("\n📋 First 3 chunks preview:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {chunk['id']} (chars {chunk['start']}-{chunk['end']}) ---")
        print(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])

if __name__ == "__main__":
    demo()
