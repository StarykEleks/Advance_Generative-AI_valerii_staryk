import os
from typing import List, Dict
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Vector DB
CHROMA_DIR = os.getenv("CHROMA_DIR", "chromadb")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def read_pdf_pages(path: str) -> List[Dict]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"text": text, "page": i + 1})
    return pages

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, chunk_size=900, overlap=150) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = max(end - overlap, start + 1)
    return chunks

def ingest(data_dir="data"):
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    col = client.get_or_create_collection(COLLECTION_NAME)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    ids, docs, metas = [], [], []
    id_counter = 0

    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)

        if fname.lower().endswith(".pdf"):
            pages = read_pdf_pages(path)
            for p in pages:
                for c in chunk_text(p["text"]):
                    ids.append(f"{fname}-{id_counter}")
                    docs.append(c)
                    metas.append({"source": fname, "page": p["page"]})
                    id_counter += 1

        elif fname.lower().endswith((".txt", ".md")):
            txt = read_text_file(path)
            for c in chunk_text(txt):
                ids.append(f"{fname}-{id_counter}")
                docs.append(c)
                metas.append({"source": fname, "page": None})
                id_counter += 1

    if not ids:
        raise RuntimeError("No documents found in /data.")

    embeddings = embedder.encode(docs, show_progress_bar=True).tolist()

    MAX_BATCH = 10000
    for start in range(0, len(ids), MAX_BATCH):
        end = start + MAX_BATCH
        col.upsert(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metas[start:end],
            embeddings=embeddings[start:end],
        )

    print(f"Ingested {len(ids)} chunks into Chroma at {CHROMA_DIR} in batches of {MAX_BATCH}")

if __name__ == "__main__":
    ingest()
