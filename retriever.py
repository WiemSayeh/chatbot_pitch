import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ======================
# ⚙️ Configuration
# ======================
CHUNK_FILE = "chunks_data.json"   # Fichier JSON des PDF découpés
EMBED_MODEL = "all-MiniLM-L6-v2"

print("🧠 Chargement du modèle d'embedding...")
embedding_model = SentenceTransformer(EMBED_MODEL)

print("📂 Chargement des embeddings depuis le fichier JSON...")
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)

EMBEDDINGS = np.array([c["embedding"] for c in CHUNKS])
EMBED_NORMS = np.linalg.norm(EMBEDDINGS, axis=1)

def retrieve(query, top_k=5):
    """
    Récupère les passages les plus pertinents pour une requête.
    """
    query_emb = embedding_model.encode([query])[0]
    query_norm = np.linalg.norm(query_emb)
    sims = np.dot(EMBEDDINGS, query_emb) / (EMBED_NORMS * query_norm + 1e-10)

    top_idx = np.argsort(sims)[::-1][:top_k]
    top_chunks = [CHUNKS[i] for i in top_idx]

    print("\n=== Paragraphes pertinents ===")
    for i, idx in enumerate(top_idx):
        c = CHUNKS[idx]
        print(f"\n[{i+1}] PDF: {c['pdf']} | Score: {sims[idx]:.4f}")
        print(c['text'][:400] + ("..." if len(c['text']) > 400 else ""))

    return top_chunks
