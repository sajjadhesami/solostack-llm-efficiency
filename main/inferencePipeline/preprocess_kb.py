import json
import os
import re
from pathlib import Path

"""
preprocess_kb.py

This script takes your existing knowledge base files (algebra, geography, chinese),
parses any format (documents with chunks, enhanced chunks, inconsistent fields),
and outputs a clean, flattened, tag-aware, retriever-optimized JSON file:

    {subject}_knowledge_preprocessed.json

Each flattened "chunk" becomes a standalone retrievable entry with fields:
    - chunk_id
    - title
    - text
    - tags
    - doc_meta

This format is exactly what the new rag.py expects and maximizes accuracy
for Qwen-1.7B or any small model using RAG.
"""

# --------------------------
# Utility
# --------------------------


def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def guess_tags(text: str) -> list:
    """Infer tags from text heuristically."""
    t = text.lower()
    tags = []

    if any(w in t for w in ["theorem", "lemma", "identity", "inequality"]):
        tags.append("theorem")
    if "example" in t:
        tags.append("example")
    if any(w in t for w in ["olympiad", "imo", "problem", "solve", "find"]):
        tags.append("problem")
    if "definition" in t:
        tags.append("definition")

    if not tags:
        tags.append("general")

    return list(set(tags))


# --------------------------
# Core flattening
# --------------------------

def flatten_kb(subject: str, path: str, out_dir: str):
    """
    Load a KB json file in *any* of your formats and flatten it into
    {subject}_knowledge_preprocessed.json
    """

    # Accept input as either a directory or a single JSON file
    if os.path.isdir(path):
        # Try to locate subject-related file
        candidates = [f for f in os.listdir(path) if f.endswith(
            ".json") and subject in f.lower()]
        if not candidates:
            raise ValueError(
                f"No JSON KB file for subject '{subject}' found in {path}")
        kb_file = os.path.join(path, candidates[0])
    else:
        kb_file = path

    print(f"[INFO] Loading: {kb_file}")
    with open(kb_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Try to find documents
    documents = []
    problem_patterns = {}

    if isinstance(data, dict):
        if "documents" in data:
            documents = data["documents"]
            problem_patterns = data.get("problem_patterns", {})
        else:
            for k, v in data.items():
                if isinstance(v, dict) and "documents" in v:
                    documents = v["documents"]
                    problem_patterns = v.get("problem_patterns", {})
                    break
        # If still empty, treat all values as possible documents
        if not documents:
            docs_guess = []
            for k, v in data.items():
                if isinstance(v, dict):
                    if any(key in v for key in ("chunks", "content", "text")):
                        v2 = dict(v)
                        v2["id"] = v.get("id", k)
                        docs_guess.append(v2)
            if docs_guess:
                documents = docs_guess

    elif isinstance(data, list):
        documents = data

    if not documents:
        raise ValueError(f"Could not locate 'documents' in {kb_file}")

    # Flatten
    flat_chunks = []
    for doc in documents:
        doc_id = doc.get("id") or doc.get("title") or doc.get("name") or None
        doc_title = doc.get("title") or doc.get("topic") or ""
        doc_meta = {k: v for k, v in doc.items() if k not in (
            "chunks", "content", "text")}

        # If doc is already a single chunk
        if "chunks" not in doc and ("content" in doc or "text" in doc):
            chunk_text = normalize_text(
                doc.get("content") or doc.get("text") or "")
            tags = doc.get("tags") or guess_tags(chunk_text)
            flat_chunks.append({
                "chunk_id": doc_id or f"{subject}_chunk_{len(flat_chunks)}",
                "title": doc_title,
                "text": chunk_text,
                "tags": tags,
                "doc_meta": doc_meta,
            })
            continue

        # Multi-chunk document
        for i, c in enumerate(doc.get("chunks", [])):
            if isinstance(c, str):
                chunk_text = normalize_text(c)
                tags = guess_tags(chunk_text)
            elif isinstance(c, dict):
                chunk_text = normalize_text(
                    c.get("text") or c.get("content") or "")
                tags = c.get("tags") or guess_tags(chunk_text)
            else:
                chunk_text = normalize_text(str(c))
                tags = guess_tags(chunk_text)

            chunk_id = None
            if isinstance(c, dict):
                chunk_id = c.get("chunk_id")

            flat_chunks.append({
                "chunk_id": chunk_id or f"{doc_id}_{i}",
                "title": doc_title,
                "text": chunk_text,
                "tags": tags,
                "doc_meta": doc_meta,
            })

    # Save output
    out = {
        "documents": flat_chunks,
        "problem_patterns": problem_patterns
    }

    out_path = os.path.join(out_dir, f"{subject}_knowledge_preprocessed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Preprocessed KB saved → {out_path}")
    return out_path


# --------------------------
# CLI Entry
# --------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess KB for improved RAG")
    parser.add_argument("--subject", type=str, required=True,
                        choices=["algebra", "geography", "chinese"],
                        help="Which subject KB to preprocess")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to KB JSON file or directory")
    parser.add_argument("--out", type=str, default="./",
                        help="Where to save the preprocessed KB")

    args = parser.parse_args()

    flatten_kb(args.subject, args.input, args.out)
