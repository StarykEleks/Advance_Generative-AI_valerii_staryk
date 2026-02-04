import os
from typing import List, Dict, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Vector DB
CHROMA_DIR = os.getenv("CHROMA_DIR", "chromadb")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_col = _client.get_or_create_collection(COLLECTION_NAME)
_embedder = SentenceTransformer(EMBED_MODEL_NAME)

def retrieve(query: str, k: int = 5) -> List[Dict]:
    q_emb = _embedder.encode([query]).tolist()[0]
    res = _col.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"text": doc, "meta": meta, "distance": dist})
    return out

def format_context(chunks: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Returns:
      context_text for LLM
      citations list with (source, page)
    """
    ctx_lines = []
    cites = []
    for i, ch in enumerate(chunks, start=1):
        src = ch["meta"].get("source")
        page = ch["meta"].get("page")
        cites.append({"source": src, "page": page})
        tag = f"[{i}] {src}" + (f" p.{page}" if page else "")
        ctx_lines.append(f"{tag}\n{ch['text']}")
    return "\n\n".join(ctx_lines), cites

def should_offer_ticket(chunks: List[Dict], threshold: float = 0.75) -> bool:
    if not chunks:
        return True
    best = min(ch["distance"] for ch in chunks)
    return best > threshold
