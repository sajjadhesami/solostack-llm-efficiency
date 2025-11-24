"""
Enhanced RAG (Retrieval-Augmented Generation) module.

Key improvements vs original:
- Robust loading of enhanced knowledge-base JSON formats (supports top-level 'documents',
  and documents containing 'chunks' arrays).
- Flattens nested chunks into standalone retrievable chunks with preserved metadata.
- Tag boosting (theorem/example/problem/...).
- Pattern-based boosts using 'problem_patterns' if present in KB.
- Fast top-k selection using a heap, then light reranking based on token overlap and tag boosts.
- Better formatting of context for prompts (includes chunk id, tags, title).
"""

import os
import json
import re
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
import heapq
from collections import Counter, defaultdict

# Pre-compile regex pattern for faster tokenization
_TOKEN_PATTERN = re.compile(r'\b\w+\b')

# Default top_k fallback (will prefer config.RAG_TOP_K when used)
DEFAULT_TOP_K = 3


class SimpleRAGRetriever:
    """Enhanced simple RAG retriever optimized for small models (Qwen-1.7B)."""

    def __init__(self, knowledge_base_path: str, subject: str, top_k: int = DEFAULT_TOP_K):
        """
        Args:
            knowledge_base_path: directory containing knowledge base files
            subject: subject name (algebra, geography, chinese)
            top_k: how many final chunks to return (after rerank)
        """
        self.subject = subject.lower()
        self.top_k = top_k
        self.knowledge_base_path = os.path.expanduser(knowledge_base_path)
        # flat list of chunk dicts (each chunk is a standalone retrievable object)
        self.knowledge_chunks: List[Dict] = []
        self.chunk_texts: List[str] = []
        self.chunk_token_sets: List[Set[str]] = []
        # optional problem patterns loaded from KB (name -> pattern dict)
        self.problem_patterns: Dict[str, Dict] = {}
        # small cached token counts for reranking
        self.chunk_token_counts: List[Counter] = []
        # load on init
        self._load_knowledge_base()

    # ------------------------
    # Loading & flattening KB
    # ------------------------
    def _find_kb_file_candidates(self) -> List[str]:
        """
        Return candidate file paths that may contain the KB for this subject.
        Tries a few reasonable file name patterns.
        """
        candidates = []
        base = self.knowledge_base_path
        names = [
            f"{self.subject}_knowledge.json",
            f"{self.subject}_knowledge_combined.json",
            f"{self.subject}_knowledge_combined_with_patterns.json",
            f"{self.subject}.json",
            "knowledge.json",
            "kb.json",
        ]
        for n in names:
            candidates.append(os.path.join(base, n))
        # also include any json file in dir as fallback
        try:
            for entry in os.listdir(base):
                if entry.endswith(".json") and entry not in names:
                    candidates.append(os.path.join(base, entry))
        except Exception:
            # ignore listing errors
            pass
        # keep order but remove duplicates
        seen = set()
        filtered = []
        for c in candidates:
            if c not in seen:
                filtered.append(c)
                seen.add(c)
        return filtered

    def _load_knowledge_base(self):
        """Robust loader — finds kb file, flattens nested documents/chunks into self.knowledge_chunks."""
        candidates = self._find_kb_file_candidates()
        loaded = False

        for kb_file in candidates:
            if not os.path.exists(kb_file):
                continue
            try:
                with open(kb_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                # skip unreadable file
                continue

            # Try to find documents and optionally problem_patterns
            documents = []
            problem_patterns = {}
            # Case: top-level dict with keys
            if isinstance(data, dict):
                # If file matches pattern subject_knowledge -> nested dict
                if f"{self.subject}_knowledge" in data and isinstance(data[f"{self.subject}_knowledge"], dict):
                    sec = data[f"{self.subject}_knowledge"]
                    documents = sec.get("documents", []) or sec.get("docs", [])
                    problem_patterns = sec.get(
                        "problem_patterns", {}) or sec.get("patterns", {})
                elif "documents" in data:
                    documents = data.get("documents", [])
                    problem_patterns = data.get(
                        "problem_patterns", {}) or data.get("patterns", {})
                else:
                    # try to find the first value that contains 'documents'
                    for key, val in data.items():
                        if isinstance(val, dict) and 'documents' in val:
                            documents = val.get('documents', [])
                            problem_patterns = val.get(
                                'problem_patterns', {}) or val.get('patterns', {})
                            break
                    # otherwise, if the dict looks like {id: {title, chunks}} treat each value as a document
                    if not documents:
                        possible_docs = []
                        for key, val in data.items():
                            if isinstance(val, dict) and ('chunks' in val or 'content' in val or 'text' in val):
                                # preserve top-level key as id/title
                                doc = dict(val)
                                doc.setdefault('id', key)
                                possible_docs.append(doc)
                        if possible_docs:
                            documents = possible_docs

            elif isinstance(data, list):
                # assume list of documents
                documents = data

            # Filter and flatten documents: ensure all are dicts, and expand any nested lists
            if documents:
                # First pass: collect all dict documents, expanding any lists
                filtered_docs = []

                def collect_dicts(items):
                    """Recursively collect all dict items from a list, expanding nested lists."""
                    for item in items:
                        if isinstance(item, dict):
                            filtered_docs.append(item)
                        elif isinstance(item, list):
                            # Recursively process nested lists
                            collect_dicts(item)
                        else:
                            # Skip other types (str, int, etc.)
                            print(
                                f"  Warning: Skipping non-dict, non-list item in documents for subject '{self.subject}': {type(item).__name__}")

                collect_dicts(documents)
                documents = filtered_docs

            # If we found documents, flatten them
            if documents:
                # Flatten each document's chunks into per-chunk objects
                flat_chunks = []
                for doc in documents:
                    # At this point, doc should always be a dict, but add safety check
                    if not isinstance(doc, dict):
                        # Skip any non-dict items (shouldn't happen after filtering)
                        print(
                            f"  Warning: Skipping non-dict document item for subject '{self.subject}': {type(doc).__name__}")
                        continue

                    # Document-level metadata (doc is confirmed to be a dict)
                    doc_id = doc.get('id') or doc.get(
                        'title') or doc.get('name')
                    doc_title = doc.get('title') or doc.get(
                        'name') or doc.get('topic') or ""
                    doc_meta = {k: v for k, v in doc.items() if k not in (
                        'chunks', 'content', 'text', 'example')}
                    # If doc already is a chunk-like object (has 'content' or 'text') then treat as single chunk
                    if 'chunks' not in doc and ('content' in doc or 'text' in doc):
                        chunk_text = doc.get('content', doc.get('text', ''))
                        chunk_obj = {
                            "chunk_id": doc_id or f"chunk_{len(flat_chunks)}",
                            "title": doc_title,
                            "text": chunk_text,
                            "tags": doc.get('tags', []),
                            "doc_meta": doc_meta,
                        }
                        flat_chunks.append(chunk_obj)
                    else:
                        # doc contains 'chunks' list
                        for i, c in enumerate(doc.get('chunks', [])):
                            # c may be a string or dict
                            if isinstance(c, str):
                                chunk_text = c
                                tags = []
                            elif isinstance(c, dict):
                                chunk_text = c.get('text') or c.get(
                                    'content') or c.get('excerpt') or ""
                                tags = c.get('tags', []) or c.get('tag', [])
                            else:
                                chunk_text = str(c)
                                tags = []

                            chunk_id = c.get('chunk_id') if isinstance(
                                c, dict) and c.get('chunk_id') else f"{doc_id}_{i}"
                            chunk_title = c.get('title') if isinstance(
                                c, dict) and c.get('title') else doc_title
                            flat_chunks.append({
                                "chunk_id": chunk_id,
                                "title": chunk_title,
                                "text": chunk_text,
                                "tags": tags,
                                "doc_meta": doc_meta,
                            })
                # store flattened chunks and patterns
                self.knowledge_chunks = flat_chunks
                # Normalize problem_patterns: convert list format to dict if needed
                was_list = isinstance(problem_patterns, list)
                if was_list:
                    # Convert list of pattern objects to dict format
                    normalized_patterns = {}
                    for pattern_obj in problem_patterns:
                        if isinstance(pattern_obj, dict):
                            pattern_name = pattern_obj.get('pattern') or pattern_obj.get(
                                'name') or str(len(normalized_patterns))
                            # Extract keywords from techniques or keywords fields
                            keywords = pattern_obj.get(
                                'keywords', []) or pattern_obj.get('techniques', [])
                            title_match = pattern_obj.get(
                                'title_match', []) or [pattern_name]
                            normalized_patterns[pattern_name] = {
                                'keywords': keywords if isinstance(keywords, list) else [keywords] if keywords else [],
                                'title_match': title_match if isinstance(title_match, list) else [title_match] if title_match else []
                            }
                        else:
                            print(
                                f"  Warning: Skipping non-dict pattern object for subject '{self.subject}': {type(pattern_obj).__name__}")
                    problem_patterns = normalized_patterns
                elif not isinstance(problem_patterns, dict):
                    # If it's neither list nor dict, default to empty dict
                    problem_patterns = {}
                self.problem_patterns = problem_patterns or {}
                # Warn if problem_patterns is empty
                if not self.problem_patterns:
                    if was_list:
                        print(
                            f"  Warning: problem_patterns list found for subject '{self.subject}' but was empty or contained no valid patterns")
                    else:
                        print(
                            f"  Warning: No problem_patterns found for subject '{self.subject}' in knowledge base")
                loaded = True
                break

        if not loaded:
            # no KB discovered
            print(
                f"  Warning: No knowledge base found for subject '{self.subject}' in {self.knowledge_base_path}")
            self.knowledge_chunks = []
            self.chunk_texts = []
            self.chunk_token_sets = []
            return

        # Prepare text/token arrays
        self.chunk_texts = []
        self.chunk_token_sets = []
        self.chunk_token_counts = []
        for chunk in self.knowledge_chunks:
            text = chunk.get('text', '') or chunk.get('content', '') or ""
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            self.chunk_texts.append(text)
            tokens = self._tokenize(text)
            token_set = set(tokens)
            self.chunk_token_sets.append(token_set)
            self.chunk_token_counts.append(Counter(tokens))

        print(
            f"  ✓ Loaded {len(self.knowledge_chunks)} flattened knowledge chunks for '{self.subject}'")

    # ------------------------
    # Tokenization & scoring
    # ------------------------
    def _tokenize(self, text: str) -> List[str]:
        text = (text or "").lower()
        return _TOKEN_PATTERN.findall(text)

    def _compute_base_similarity(self, query_token_set: Set[str], chunk_token_set: Set[str]) -> float:
        """
        Fast base similarity (Jaccard + term-match ratio).
        """
        if not query_token_set or not chunk_token_set:
            return 0.0

        intersection = query_token_set & chunk_token_set
        intersection_size = len(intersection)
        if intersection_size == 0:
            return 0.0

        union_size = len(query_token_set) + \
            len(chunk_token_set) - intersection_size
        if union_size == 0:
            return 0.0

        jaccard = intersection_size / union_size
        term_match_ratio = intersection_size / len(query_token_set)

        score = 0.6 * jaccard + 0.4 * term_match_ratio
        return float(score)

    def _tag_boost(self, base_score: float, chunk: Dict) -> float:
        """
        Boost score according to tags and doc_meta hints.
        """
        score = base_score
        tags = chunk.get('tags') or []
        if isinstance(tags, str):
            tags = [tags]
        tags = [t.lower() for t in tags]

        # simple multiplicative boosts
        if 'theorem' in tags or 'lemma' in tags:
            score *= 1.35
        if 'example' in tags:
            score *= 1.15
        if 'problem' in tags or 'olympiad' in tags:
            score *= 1.25
        if 'definition' in tags:
            score *= 1.05
        # small bonus if the chunk title contains 'inequality' and query includes 'inequality'
        return score

    def _pattern_boost_map(self, query: str) -> Dict[int, float]:
        """
        Create a small map of chunk_idx -> boost based on problem_patterns.
        problem_patterns is expected to be like:
        { "inequality": {"keywords": ["inequality",">=","≤"], "title_match": ["inequality"]}, ... }
        """
        boosts = {}
        if not self.problem_patterns:
            return boosts
        # Safety check: ensure problem_patterns is a dict
        if not isinstance(self.problem_patterns, dict):
            return boosts
        qlow = query.lower()
        for pname, pdata in self.problem_patterns.items():
            # Safety check: ensure pdata is a dict
            if not isinstance(pdata, dict):
                continue
            keywords = pdata.get('keywords', []) or pdata.get('kw', [])
            title_match = pdata.get(
                'title_match', []) or pdata.get('titles', [])
            match = False
            for kw in keywords:
                if kw and kw.lower() in qlow:
                    match = True
                    break
            if not match:
                for kw in title_match:
                    if kw and kw.lower() in qlow:
                        match = True
                        break
            if not match:
                continue

            # if matched, boost chunks whose title or tags include the pattern name or title_match words
            for idx, chunk in enumerate(self.knowledge_chunks):
                title = (chunk.get('title') or "").lower()
                tags = [t.lower() for t in (chunk.get('tags') or [])
                        ] if chunk.get('tags') else []
                if pname.lower() in title or any(tm.lower() in title for tm in title_match) or pname.lower() in tags:
                    boosts[idx] = boosts.get(idx, 1.0) * 1.3
                # also small boost if doc_meta mentions pattern name
                doc_meta = chunk.get('doc_meta') or {}
                if any(pname.lower() in str(v).lower() for v in doc_meta.values()):
                    boosts[idx] = boosts.get(idx, 1.0) * 1.12
        return boosts

    # ------------------------
    # Retrieval pipeline
    # ------------------------
    def retrieve(self, query: str, return_scores: bool = False) -> List[Dict]:
        """
        Retrieve top-k relevant chunks for the query.

        Steps:
          1) Quick base scoring using pre-tokenized sets + heap to get candidate set (~top 2*top_k)
          2) Compute light rerank on candidates using token-overlap counts + tag & pattern boosts
          3) Return final top self.top_k chunks (with relevance_score added)

        Args:
            query: user query/question
            return_scores: if True returns list of chunks with 'relevance_score' key

        Returns:
            list of chunk dicts (length <= self.top_k)
        """
        if not self.knowledge_chunks:
            return []

        q_tokens = self._tokenize(query)
        q_set = set(q_tokens)
        if not q_set:
            return []

        # quick heap: maintain top N candidates (we choose N = max(2*top_k, 10))
        candidate_limit = max(2 * self.top_k, 10)
        heap = []  # stores (-base_score, idx)

        for i, chunk_set in enumerate(self.chunk_token_sets):
            base = self._compute_base_similarity(q_set, chunk_set)
            if base <= 0.0:
                continue
            if len(heap) < candidate_limit:
                heapq.heappush(heap, (-base, i))
            else:
                if base > -heap[0][0]:
                    heapq.heapreplace(heap, (-base, i))

        if not heap:
            return []

        # Extract candidate indices
        candidates = [idx for (_, idx) in heap]
        # Build pattern boost map
        pattern_boosts = self._pattern_boost_map(query)

        # Now light rerank
        reranked = []
        # precompute q_token counts for overlap scoring
        q_counts = Counter(q_tokens)
        for idx in candidates:
            chunk = self.knowledge_chunks[idx]
            base = self._compute_base_similarity(
                q_set, self.chunk_token_sets[idx])
            # lightweight overlap score: sum of min counts between query and chunk
            overlap_count = sum(min(self.chunk_token_counts[idx].get(
                t, 0), q_counts.get(t, 0)) for t in q_counts)
            # normalized overlap (avoid dividing by zero)
            norm_overlap = overlap_count / (len(q_tokens) + 1)
            # apply tag boost
            score = base * 0.7 + norm_overlap * 0.3
            score = self._tag_boost(score, chunk)
            # apply pattern boost if any
            if idx in pattern_boosts:
                score *= pattern_boosts[idx]
            reranked.append((score, idx))

        # final sort and take top_k
        reranked.sort(key=lambda x: x[0], reverse=True)
        top = reranked[: self.top_k]

        results = []
        for score, idx in top:
            chunk_copy = dict(self.knowledge_chunks[idx])  # shallow copy
            chunk_copy['relevance_score'] = float(score)
            results.append(chunk_copy)

        if return_scores:
            return results
        else:
            # only return chunks (dicts include relevance_score)
            return results

    # ------------------------
    # Format context for prompt
    # ------------------------
    def format_context(self, retrieved_chunks: List[Dict], max_excerpt_chars: int = 800) -> str:
        """
        Format retrieved chunks into a single context string for the model prompt.

        Includes chunk id, title (if present), tags and a short excerpt.
        """
        if not retrieved_chunks:
            return ""

        parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            title = chunk.get('title') or ""
            cid = chunk.get('chunk_id') or f"chunk_{i}"
            tags = chunk.get('tags') or []
            text = (chunk.get('text') or chunk.get('content') or "").strip()
            excerpt = text[:max_excerpt_chars].strip()
            meta = []
            if title:
                meta.append(title)
            if tags:
                meta.append(
                    "tags: " + ",".join(tags if isinstance(tags, list) else [str(tags)]))
            meta_str = " | ".join(meta) if meta else ""
            header = f"[{i}] {cid}"
            if meta_str:
                header = f"{header} ({meta_str})"
            # include relevance score if present
            score = chunk.get('relevance_score')
            if score is not None:
                header = f"{header} - score: {score:.4f}"
            parts.append(f"{header}\n{excerpt}")
        return "\n\n".join(parts)


# ------------------------
# Global cache & helpers
# ------------------------
_rag_retrievers: Dict[str, SimpleRAGRetriever] = {}


def get_rag_retriever(subject: str, knowledge_base_dir: str, top_k: int = DEFAULT_TOP_K) -> Optional[SimpleRAGRetriever]:
    """
    Get-or-create a cached SimpleRAGRetriever for subject + KB dir + top_k.
    """
    if not subject:
        return None
    s = subject.lower()
    allowed = {'algebra', 'geography', 'chinese'}
    if s not in allowed:
        return None

    key = f"{s}::{knowledge_base_dir}::topk={top_k}"
    if key in _rag_retrievers:
        return _rag_retrievers[key]
    retriever = SimpleRAGRetriever(knowledge_base_dir, s, top_k)
    if retriever.knowledge_chunks:
        _rag_retrievers[key] = retriever
        return retriever
    return None


def retrieve_context(question: str, subject: str, knowledge_base_dir: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Convenience wrapper: returns formatted context string for the question.
    """
    retriever = get_rag_retriever(subject, knowledge_base_dir, top_k)
    if not retriever:
        return ""
    retrieved = retriever.retrieve(question)
    if not retrieved:
        return ""
    return retriever.format_context(retrieved)
