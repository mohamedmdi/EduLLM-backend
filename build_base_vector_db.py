# build_base_vector_db.py
import os, json
import numpy as np
import faiss
from rag_engine import read_file, chunk_text, embed_chunks

BASE_DIR = "data"
EMBEDDING_DIR = "embeddings"

def build_vector_db():
    chunks = []
    for fname in os.listdir(BASE_DIR):
        path = os.path.join(BASE_DIR, fname)
        try:
            text = read_file(path)
            chunks += chunk_text(text)
        except Exception as e:
            print(f"❌ Failed to read {fname}: {e}")

    embeddings = embed_chunks(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    with open(f"{EMBEDDING_DIR}/base_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    np.save(f"{EMBEDDING_DIR}/base_embeddings.npy", embeddings)
    faiss.write_index(index, f"{EMBEDDING_DIR}/base.index")
    print("✅ Base vector DB built and saved.")

if __name__ == "__main__":
    build_vector_db()
