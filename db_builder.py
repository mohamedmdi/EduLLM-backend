# build_full_rag_index.py
import os
from rag_engine import add_document

BASE_DIR = "data"

def build_full_rag_index():
    for fname in os.listdir(BASE_DIR):
        path = os.path.join(BASE_DIR, fname)
        try:
            add_document(path)
        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")

if __name__ == "__main__":
    build_full_rag_index()
