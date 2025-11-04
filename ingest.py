import os
import json
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

PDF_FOLDER = "data"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
OUTPUT_FILE = "chunks_data.json"

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def ingest_pdfs():
    model = SentenceTransformer(EMBED_MODEL)
    all_chunks = []

    for pdf_file in os.listdir(PDF_FOLDER):
        if pdf_file.endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, pdf_file)
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            chunks = split_text(text)
            print(f"Ingesting {pdf_file} -> {len(chunks)} chunks")

            embeddings = model.encode(chunks).tolist()

            for c, emb in zip(chunks, embeddings):
                all_chunks.append({
                    "pdf": pdf_file,
                    "text": c,
                    "embedding": emb
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ Embeddings créés et sauvegardés dans {OUTPUT_FILE}")

if __name__ == "__main__":
    ingest_pdfs()
