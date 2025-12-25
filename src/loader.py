from pathlib import Path
from typing import List
from pypdf import PdfReader
from .config import PDF_DIR

def load_pdf_paths() -> List[Path]:
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    return pdf_paths

def pdf_to_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)

if __name__ == "__main__":
    paths = load_pdf_paths()
    if not paths:
        print("No PDFs found in:", PDF_DIR)
    else:
        print("Found PDFs:", paths)
        sample_text = pdf_to_text(paths[0])
        print("Sample text snippet:")
        print(sample_text[:1000])
