import json
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_FILE = "chunks_data.json"
EMBED_MODEL = "all-MiniLM-L6-v2"

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, top_k=3):
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    model = SentenceTransformer(EMBED_MODEL)
    query_emb = model.encode([query])[0]

    scores = []
    for c in chunks:
        sim = cosine_similarity(query_emb, np.array(c["embedding"]))
        scores.append((sim, c))

    scores.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scores[:top_k]

    print("\n=== Paragraphes pertinents ===")
    for i, (score, c) in enumerate(top_chunks):
        print(f"\n[{i+1}] PDF: {c['pdf']} | Score: {score:.4f}")
        print(c['text'][:500] + ("..." if len(c['text']) > 500 else ""))  # tronqué

    return [c for _, c in top_chunks]
