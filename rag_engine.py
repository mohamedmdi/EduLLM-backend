import os
import json
import numpy as np
import faiss
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import asyncio
from dotenv import load_dotenv

# === Configs ===
load_dotenv(dotenv_path="/app/.env")
api_key = os.getenv("API_KEY")
print("🔑 Using", api_key)
EMBEDDING_DIR = "embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

BASE_INDEX_PATH = os.path.join(EMBEDDING_DIR, "base.index")
BASE_EMBEDDINGS_PATH = os.path.join(EMBEDDING_DIR, "base_embeddings.npy")
BASE_CHUNKS_PATH = os.path.join(EMBEDDING_DIR, "base_chunks.json")

groq_client = Groq(api_key=api_key)

# === Model Init ===
print("🧠 Loading SentenceTransformer model...")
model = SentenceTransformer("all-mpnet-base-v2")
print("✅ Model loaded.\n")


# === File Reader ===
def read_file(path):
    ext = os.path.splitext(path)[1].lower()
    print(f"📄 Reading file: {path}")
    if ext == ".pdf":
        return "\n".join(page.get_text() for page in fitz.open(path))
    elif ext == ".docx":
        return "\n".join(p.text for p in Document(path).paragraphs)
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError(f"Unsupported file type: {ext}")


# === Text Chunking ===
def chunk_text(text, chunk_size=300, overlap=50):
    print(
        f"🛠️ Chunking text into pieces of ~{chunk_size} words with {overlap} overlap..."
    )
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    print(f"✅ Created {len(chunks)} chunks.")
    return chunks


# === Embedding Generation ===
def embed_chunks(chunks):
    print(f"🧠 Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print("✅ Embeddings generated.")
    return embeddings


# === Base Index Handling (shared embeddings) ===
def load_base_index():
    if not os.path.exists(BASE_INDEX_PATH):
        raise FileNotFoundError("Base FAISS index not found.")
    print("💾 Loading base FAISS index and embeddings...")
    index = faiss.read_index(BASE_INDEX_PATH)
    embeddings = np.load(BASE_EMBEDDINGS_PATH)
    with open(BASE_CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print("✅ Base index loaded.")
    return index, chunks, embeddings


def save_base_index(index, chunks, embeddings):
    print("💾 Saving FAISS index, embeddings, and chunks...")
    faiss.write_index(index, BASE_INDEX_PATH)
    np.save(BASE_EMBEDDINGS_PATH, embeddings)
    with open(BASE_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print("✅ Index and embeddings saved.")


def initialize_base_index(dim):
    index = faiss.IndexFlatL2(dim)
    return index, [], np.empty((0, dim), dtype=np.float32)


def augment_base_index(new_chunks, new_embeddings):
    print("📊 Augmenting base FAISS index with new data...")
    if os.path.exists(BASE_INDEX_PATH):
        index, old_chunks, old_embeddings = load_base_index()
    else:
        print("⚠️ Base index not found. Initializing a new one.")
        index, old_chunks, old_embeddings = initialize_base_index(
            new_embeddings.shape[1]
        )

    index.add(new_embeddings)
    all_chunks = old_chunks + new_chunks
    all_embeddings = np.vstack([old_embeddings, new_embeddings])

    save_base_index(index, all_chunks, all_embeddings)
    print(f"✅ Augmented index with {len(new_chunks)} new chunks.")


def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine sim
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


# === User Embedding Storage Handling ===
def get_user_embedding_paths(user_id):
    # Store each user embeddings and chunks separately on disk
    user_dir = os.path.join(EMBEDDING_DIR, f"user_{user_id}")
    os.makedirs(user_dir, exist_ok=True)
    return {
        "index_path": os.path.join(user_dir, "index.index"),
        "embeddings_path": os.path.join(user_dir, "embeddings.npy"),
        "chunks_path": os.path.join(user_dir, "chunks.json"),
    }


def user_data_exists(user_id):
    paths = get_user_embedding_paths(user_id)
    return (
        os.path.exists(paths["index_path"])
        and os.path.exists(paths["embeddings_path"])
        and os.path.exists(paths["chunks_path"])
    )


def load_user_index(user_id):
    paths = get_user_embedding_paths(user_id)
    if not user_data_exists(user_id):
        print(f"⚠️ No embedding data found for user {user_id}")
        return (
            None,
            [],
            np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32),
        )

    print(f"💾 Loading FAISS index and embeddings for user {user_id}...")
    index = faiss.read_index(paths["index_path"])
    embeddings = np.load(paths["embeddings_path"])
    with open(paths["chunks_path"], encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"✅ Loaded user {user_id} index.")
    return index, chunks, embeddings


def save_user_index(user_id, index, chunks, embeddings):
    paths = get_user_embedding_paths(user_id)
    print(f"💾 Saving FAISS index, embeddings, and chunks for user {user_id}...")
    faiss.write_index(index, paths["index_path"])
    np.save(paths["embeddings_path"], embeddings)
    with open(paths["chunks_path"], "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"✅ Saved user {user_id} data.")


def initialize_user_index(dim):
    index = faiss.IndexFlatL2(dim)
    return index, [], np.empty((0, dim), dtype=np.float32)


def augment_user_index(user_id, new_chunks, new_embeddings):
    print(f"📊 Augmenting user {user_id} FAISS index with new data...")
    if user_data_exists(user_id):
        index, old_chunks, old_embeddings = load_user_index(user_id)
    else:
        print(f"⚠️ User {user_id} index not found. Initializing a new one.")
        index, old_chunks, old_embeddings = initialize_user_index(
            new_embeddings.shape[1]
        )

    index.add(new_embeddings)
    all_chunks = old_chunks + new_chunks
    all_embeddings = np.vstack([old_embeddings, new_embeddings])

    save_user_index(user_id, index, all_chunks, all_embeddings)
    print(f"✅ Augmented user {user_id} index with {len(new_chunks)} new chunks.")


# === Caching user embeddings in memory (optional) ===
_user_chunks_cache = {}
_user_embeddings_cache = {}


def cache_user_content(user_id, user_file_content):
    print(f"📃 Caching content embeddings for user {user_id} in memory...")
    chunks = chunk_text(user_file_content)
    embeddings = model.encode(chunks)
    faiss.normalize_L2(embeddings)
    _user_chunks_cache[user_id] = chunks
    _user_embeddings_cache[user_id] = embeddings


def clear_user_cache(user_id):
    _user_chunks_cache.pop(user_id, None)
    _user_embeddings_cache.pop(user_id, None)


# === Search & Retrieval ===
def search_combined_simple(user_query, user_file_content=None, top_k=5):
    """
    Simple search combining base embeddings and user_file_content embeddings if provided.
    This version doesn't use user_id or persist user embeddings.
    Kept for compatibility if needed.
    """
    print(f"🔍 Performing semantic search for query: '{user_query}'")
    query_emb = model.encode([user_query])
    faiss.normalize_L2(query_emb)

    results = []

    if os.path.exists(BASE_EMBEDDINGS_PATH):
        print("📂 Loading base data for search...")
        base_embeddings = np.load(BASE_EMBEDDINGS_PATH)
        with open(BASE_CHUNKS_PATH, encoding="utf-8") as f:
            base_chunks = json.load(f)
        base_index = create_faiss_index(base_embeddings)

        D, I = base_index.search(query_emb, top_k)
        for score, idx in zip(D[0], I[0]):
            results.append((score, base_chunks[idx]))
    else:
        print("⚠️ No base embeddings found. Skipping base search.")
        base_chunks = []

    if user_file_content:
        print("📃 Processing user-provided content...")
        user_chunks = chunk_text(user_file_content)
        user_embeddings = model.encode(user_chunks)
        faiss.normalize_L2(user_embeddings)
        user_index = create_faiss_index(user_embeddings)

        D, I = user_index.search(query_emb, top_k)
        for score, idx in zip(D[0], I[0]):
            results.append((score, user_chunks[idx]))

    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
    print(f"✅ Retrieved top {len(results)} results.")
    return [chunk for _, chunk in results]


def mmr(
    query_embedding, candidate_embeddings, candidate_texts, top_k=5, lambda_param=0.7
):
    selected = []
    selected_idxs = []

    query_sim = cosine_similarity(query_embedding, candidate_embeddings)[0]
    candidate_sim = cosine_similarity(candidate_embeddings)

    for _ in range(min(top_k, len(candidate_texts))):
        mmr_score = []
        for idx in range(len(candidate_texts)):
            if idx in selected_idxs:
                mmr_score.append(-np.inf)
                continue
            sim_to_query = query_sim[idx]
            sim_to_selected = max(
                [candidate_sim[idx][j] for j in selected_idxs], default=0
            )
            score = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
            mmr_score.append(score)

        selected_idx = np.argmax(mmr_score)
        selected.append(candidate_texts[selected_idx])
        selected_idxs.append(selected_idx)

    return selected


def search_combined_mmr(
    user_query, user_id=None, user_file_content=None, top_k=5, lambda_param=0.7
):
    print(f"🔍 Performing MMR search for query: '{user_query}'")
    query_emb = model.encode([user_query])
    faiss.normalize_L2(query_emb)

    all_chunks = []
    all_embeddings = []

    # Load base embeddings
    if os.path.exists(BASE_EMBEDDINGS_PATH):
        print("📂 Loading base chunks...")
        base_embeddings = np.load(BASE_EMBEDDINGS_PATH)
        with open(BASE_CHUNKS_PATH, encoding="utf-8") as f:
            base_chunks = json.load(f)
        faiss.normalize_L2(base_embeddings)
        all_chunks.extend(base_chunks)
        all_embeddings.append(base_embeddings)

    # Load user embeddings from disk
    if user_id:
        if user_data_exists(user_id):
            user_index, user_chunks, user_embeddings = load_user_index(user_id)
            faiss.normalize_L2(user_embeddings)
            all_chunks.extend(user_chunks)
            all_embeddings.append(user_embeddings)
        elif user_file_content:
            # If no stored user embeddings, embed and cache user content
            cache_user_content(user_id, user_file_content)
            all_chunks.extend(_user_chunks_cache[user_id])
            all_embeddings.append(_user_embeddings_cache[user_id])
        else:
            print(f"⚠️ No embeddings found for user {user_id} and no content provided.")

    # In case user_file_content provided but no user_id, embed on the fly
    if user_id is None and user_file_content is not None:
        chunks = chunk_text(user_file_content)
        embeddings = model.encode(chunks)
        faiss.normalize_L2(embeddings)
        all_chunks.extend(chunks)
        all_embeddings.append(embeddings)

    if not all_chunks:
        print("⚠️ No chunks available for search.")
        return []

    all_embeddings = np.vstack(all_embeddings)
    return mmr(
        query_emb, all_embeddings, all_chunks, top_k=top_k, lambda_param=lambda_param
    )


# === API for user to add content and save embeddings ===
def add_user_content(user_id, user_file_content):
    """
    Add user content to the user's personal embedding store on disk,
    augmenting their FAISS index and saving chunks/embeddings.
    """
    print(f"➕ Adding content for user {user_id}...")
    chunks = chunk_text(user_file_content)
    embeddings = embed_chunks(chunks)

    # Augment user index on disk
    augment_user_index(user_id, chunks, embeddings)


# === Query-Focused Summarization ===
def get_sentences(text):
    import re

    return [s.strip() for s in re.split(r"(?<=[.!?]) +", text) if s.strip()]


def summarize_chunk(chunk, user_query, max_sentences=2, max_sentences_to_consider=5):
    sentences = get_sentences(chunk)[:max_sentences_to_consider]
    if not sentences:
        return ""

    sent_embeddings = model.encode(sentences)
    query_emb = model.encode([user_query])[0]
    sims = cosine_similarity([query_emb], sent_embeddings)[0]

    top_idx = np.argsort(sims)[::-1][:max_sentences]
    return " ".join(sentences[i] for i in sorted(top_idx))


# === Groq Streaming ===
async def query_llm(prompt, model_name="meta-llama/llama-4-scout-17b-16e-instruct"):
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            stream=True,
            temperature=1,
        )
        for chunk in chat_completion:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                yield delta.content
    except Exception as e:
        yield f"❌ Error: {e}"


# === RAG Workflow ===
async def answer_query(user_query, user_file_text=None, user_id=None):
    """
    Answer query using base index + user-specific index (if user_id),
    and optionally augment user embeddings if user_file_text provided.
    """
    print(f"🤖 Answering query: '{user_query}'")

    # If user_file_text is given without user_id, just cache in memory (or ignore)
    if user_id and user_file_text:
        add_user_content(user_id, user_file_text)

    # Search combined base + user-specific embeddings
    chunks = search_combined_mmr(
        user_query, user_id=user_id, user_file_content=None, top_k=10
    )

    if not chunks:
        print("⚠️ No relevant content found for query.")
        yield "⚠️ No relevant content found."
        return

    print(f"🔍 Found {len(chunks)} relevant chunks. Preparing prompt...")
    context = "\n\n".join(chunks)

    prompt = f"""Use the following context to answer the question as precisely as possible:

{context}

Question: {user_query}
Answer:"""

    print("🧠 Sending prompt to LLM...")
    async for part in query_llm(prompt):
        yield part
