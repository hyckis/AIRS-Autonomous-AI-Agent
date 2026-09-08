from pathlib import Path
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# paths
BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "corpus"
PROCESSED_DIR = CORPUS_DIR / "processed"
PROFILE_DIR = PROCESSED_DIR / "bank_profiles"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
INDEX_DIR = CORPUS_DIR / "index"
CHUNKS_PATH = INDEX_DIR / "finance_chunk.json"
EMBEDDINGS_PATH = INDEX_DIR / "finance_embeddings.npy"
# exclude fixed context - bank_profiles and fed_scenarios are fixed context
RETRIEVAL_CATEGORIES = {
    "fomc",
    "fdic",
    "fed_fsr",
    "regional_banks",
}

# embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None

def get_embedding_model():
    """
    Lazy-load the embedding model so importing this module
    does not immediately load SentenceTransformer.
    """
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _embedding_model

def load_bank_profile(bank_id):
    bank_id = bank_id.upper()
    if bank_id not in {"A", "B"}: raise ValueError("bank_id must be either 'A' or 'B'")
    filename = (
        "bank_a.json" if bank_id == "A" else "bank_b.json"
    )
    path = PROFILE_DIR / filename
    if not path.exists(): raise FileNotFoundError(f"Bank profile not found: {path}")
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def load_manifest():
    if not MANIFEST_PATH.exists(): raise FileNotFoundError(f"manifest.json not found: {MANIFEST_PATH}")
    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as f: manifest = json.load(f)
    if not isinstance(manifest, list): raise ValueError("manifest.json must contain a JSON list.")
    return manifest

# load documents
def _read_text_file(path):
    with open(path, "r", encoding="utf-8") as f: return f.read()

def _read_json_file(path):
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    )

def load_documents():
    manifest = load_manifest()
    documents = []

    for item in manifest:
        if not item.get("retrievable", False): continue
        if not item.get("cutoff_valid", False): continue

        relative_path = Path(item["path"])
        # Example: processed/fomc/...
        parts = relative_path.parts
        if len(parts) < 2: continue

        category = parts[1]
        if category not in RETRIEVAL_CATEGORIES: continue

        path = CORPUS_DIR / relative_path
        if not path.exists(): raise FileNotFoundError(f"Corpus document not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}: text = _read_text_file(path)
        elif suffix == ".json": text = _read_json_file(path)
        else: continue  # Skip unsupported files rather than silently attempting PDF parsing here.
    
        documents.append(
            {
                "doc_id": item["doc_id"],
                "source": item.get("source"),
                "document_type": item.get(
                    "document_type"
                ),
                "publication_date": item.get(
                    "publication_date"
                ),
                "path": str(relative_path),
                "text": text,
            }
        )

    return documents

# chunking
def _split_markdown_sections(text):
    sections = []
    current_heading = "Document"
    current_lines = []

    for line in text.splitlines():
        heading_match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$",line)
        if heading_match:
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append({
                        "section": current_heading,
                        "text": section_text,
                    })
            current_heading = heading_match.group(1).strip()
            current_lines = []
        else: current_lines.append(line)

    if current_lines: 
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append({
                "section": current_heading,
                "text": section_text,
            })

    return sections

def _split_long_text(text, max_chars=1800, overlap_chars=200,):
    text = text.strip()
    if len(text) <= max_chars: return [text]
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""
    for paragraph in paragraphs: 
        paragraph = paragraph.strip()
        if not paragraph: continue

        candidate = (
            f"{current}\n\n{paragraph}".strip()
            if current
            else paragraph
        )
        if len(candidate) <= max_chars: current = candidate
        else:
            if current: chunks.append(current)
            # If one paragraph alone is too large, fall back to character windows.
            if len(paragraph) > max_chars:
                start = 0
                while start < len(paragraph):
                    end = start + max_chars
                    chunks.append(paragraph[start:end])
                    start = (end - overlap_chars)
            else: current = paragraph

    if current: chunks.append(current)
    return chunks

SKIP_SECTIONS = {
    "Corpus Tags", "Tags", "Metadata",
}
def chunk_documents(documents, max_chars=1800, overlap_chars=200):
    chunks = []
    for document in documents:
        sections = _split_markdown_sections(document["text"])
        chunk_number = 0

        for section in sections:
            if section["section"].strip() in SKIP_SECTIONS: continue
            section_chunks = _split_long_text(
                section["text"],
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )

            for text_chunk in section_chunks:
                chunk_id = (f"{document['doc_id']}_{chunk_number:03d}")
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": document["doc_id"],
                    "source": document["source"],
                    "document_type": document["document_type"],
                    "publication_date": document["publication_date"],
                    "section": section["section"],
                    "text": text_chunk.strip(),
                })
                chunk_number += 1

    return chunks

# build index
def build_index(force=False):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if (not force and CHUNKS_PATH.exists() and EMBEDDINGS_PATH.exists()):
        print("Finance index already exists. Use force=True to rebuild.")
        return
    
    documents = load_documents()
    if not documents: raise RuntimeError("No retrieval documents were loaded.")

    chunks = chunk_documents(documents)
    if not chunks: raise RuntimeError("No chunks were generated.")

    texts = [ chunk["text"] for chunk in chunks ]
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32,)

    with open(CHUNKS_PATH, "w", encoding="utf-8",) as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2,
        )
    np.save(EMBEDDINGS_PATH, embeddings,)

    print(
        f"Built finance index:"
        f"\n  documents: {len(documents)}"
        f"\n  chunks: {len(chunks)}"
        f"\n  embedding shape: "
        f"{embeddings.shape}"
    )

# load index
def load_index():
    if (not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists()):
        raise FileNotFoundError("Finance index has not been built. Run build_index() first.")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f: chunks = json.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    if len(chunks) != embeddings.shape[0]:
        raise ValueError("finance_chunks.json and finance_embeddings.npy have different lengths.")

    return chunks, embeddings

def retrieve_finance(query, top_k=5):
    if not query.strip(): raise ValueError("Retrieval query cannot be empty.")

    chunks, embeddings = load_index()
    model = get_embedding_model()

    query_embedding = model.encode(query, normalize_embeddings=True)
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    scores = embeddings @ query_embedding
    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for index in top_indices:
        result = dict(chunks[int(index)])
        result["score"] = float(scores[index])
        results.append(result)

    return results

# formatting for prompts
def get_evidence_role(chunk):
    document_type = chunk.get("document_type", "")
    source = chunk.get("source", "")

    if document_type == "regional_bank_10k":
        return "COMPARATOR_BANK"

    if document_type in {
        "quarterly_banking_profile",
        "financial_stability_report",
    }:
        return "INDUSTRY_CONTEXT"

    if document_type == "fomc_statement":
        return "MACRO_CONTEXT"

    return "GENERAL_CONTEXT"

def format_retrieved_context(results):
    blocks = []

    for i, result in enumerate(results, start=1):

        evidence_role = get_evidence_role(result)

        block = (
            f"[Retrieved Evidence {i}]\n"
            f"CHUNK_ID: {result['chunk_id']}\n"
            f"EVIDENCE_ROLE: {evidence_role}\n"
            f"SOURCE: {result.get('source')}\n"
            f"DATE: {result.get('publication_date')}\n"
            f"SECTION: {result.get('section')}\n"
            f"RETRIEVAL_SCORE: {result.get('score', 0):.4f}\n\n"
            f"{result['text']}"
        )

        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


# Manual test
if __name__ == "__main__":
    # First run:
    build_index(force=True)
    test_query = ("How could rising interest rates and deposit outflows create liquidity stress for a bank with large securities holdings?")
    retrieved = retrieve_finance(test_query, top_k=5)
    print("\n\n" + format_retrieved_context(retrieved))