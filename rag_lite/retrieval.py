"""
Retrieval module for semantic search and reranking.

This module implements various retrieval strategies including:
- Semantic search using cosine similarity
- Keyword-based search (Jaccard, TF-IDF, BM25)
- Hybrid search combining both approaches
- LLM-based reranking
- Query expansion
"""

import json
import logging
import math
import re
from collections import Counter
from enum import Enum
from typing import List, Tuple, Optional, Dict, Callable

import ollama

logger = logging.getLogger(__name__)


class KeywordMethod(str, Enum):
    """Available keyword similarity methods."""
    JACCARD = "jaccard"      # Original: Jaccard + overlap ratio
    TFIDF = "tfidf"          # True TF-IDF scoring
    BM25 = "bm25"            # BM25 (Best Matching 25)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    return re.findall(r'\b\w+\b', text.lower())


def jaccard_overlap_similarity(query: str, chunk: str) -> float:
    """
    Calculate keyword-based similarity using Jaccard and overlap ratio.
    
    This is the original simple method - fast but doesn't account for 
    term frequency or document length.
    
    Args:
        query: Query text
        chunk: Document chunk text
        
    Returns:
        Similarity score between 0 and 1
    """
    query_words = set(_tokenize(query))
    chunk_words = set(_tokenize(chunk))
    
    if not query_words:
        return 0.0
    
    # Jaccard similarity: |intersection| / |union|
    intersection = len(query_words & chunk_words)
    union = len(query_words | chunk_words)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Word overlap ratio: what fraction of query words appear in chunk
    overlap_ratio = intersection / len(query_words)
    
    # Combined score (weighted average)
    return jaccard * 0.3 + overlap_ratio * 0.7


def tfidf_similarity(
    query: str, 
    chunk: str, 
    corpus: Optional[List[str]] = None,
    idf_cache: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate true TF-IDF similarity between query and chunk.
    
    TF-IDF = Term Frequency × Inverse Document Frequency
    - TF: log(1 + count) to dampen high frequencies
    - IDF: log(N / df) where N is corpus size, df is document frequency
    
    Args:
        query: Query text
        chunk: Document chunk text
        corpus: Optional corpus for IDF calculation (uses chunk alone if None)
        idf_cache: Optional pre-computed IDF values
        
    Returns:
        TF-IDF similarity score (normalized to 0-1 range)
    """
    query_tokens = _tokenize(query)
    chunk_tokens = _tokenize(chunk)
    
    if not query_tokens or not chunk_tokens:
        return 0.0
    
    # Calculate term frequencies in chunk (with log dampening)
    chunk_tf = Counter(chunk_tokens)
    chunk_tf_log = {term: math.log(1 + count) for term, count in chunk_tf.items()}
    
    # Calculate IDF values
    if idf_cache is not None:
        idf = idf_cache
    elif corpus is not None:
        # Calculate IDF from corpus
        N = len(corpus)
        df = Counter()
        for doc in corpus:
            doc_terms = set(_tokenize(doc))
            for term in doc_terms:
                df[term] += 1
        idf = {term: math.log(N / df_count) if df_count > 0 else 0 
               for term, df_count in df.items()}
    else:
        # Fallback: treat chunk as single-document corpus (IDF = 0 for all terms)
        # This reduces to just TF scoring
        idf = {}
    
    # Calculate TF-IDF score for query terms
    score = 0.0
    query_term_set = set(query_tokens)
    
    for term in query_term_set:
        if term in chunk_tf_log:
            tf = chunk_tf_log[term]
            term_idf = idf.get(term, 1.0)  # Default IDF of 1 if not in corpus
            score += tf * term_idf
    
    # Normalize by query length and max possible score
    max_tf = max(chunk_tf_log.values()) if chunk_tf_log else 1.0
    max_idf = max(idf.values()) if idf else 1.0
    max_possible = len(query_term_set) * max_tf * max_idf
    
    if max_possible > 0:
        return min(1.0, score / max_possible)
    return 0.0


def bm25_similarity(
    query: str,
    chunk: str,
    corpus: Optional[List[str]] = None,
    k1: float = 1.5,
    b: float = 0.75,
    avgdl: Optional[float] = None,
    idf_cache: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate BM25 similarity between query and chunk.
    
    BM25 (Best Matching 25) improves on TF-IDF with:
    - Term frequency saturation (controlled by k1)
    - Document length normalization (controlled by b)
    
    Formula for each query term qi:
    score(qi, D) = IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))
    
    Args:
        query: Query text
        chunk: Document chunk text
        corpus: Optional corpus for IDF and avgdl calculation
        k1: Term frequency saturation parameter (typically 1.2-2.0)
        b: Document length normalization (0=no normalization, 1=full normalization)
        avgdl: Average document length (computed from corpus if None)
        idf_cache: Optional pre-computed IDF values
        
    Returns:
        BM25 similarity score (normalized to 0-1 range)
    """
    query_tokens = _tokenize(query)
    chunk_tokens = _tokenize(chunk)
    
    if not query_tokens or not chunk_tokens:
        return 0.0
    
    # Document length
    doc_len = len(chunk_tokens)
    
    # Term frequencies in chunk
    chunk_tf = Counter(chunk_tokens)
    
    # Calculate corpus statistics if provided
    if corpus is not None:
        N = len(corpus)
        if avgdl is None:
            total_len = sum(len(_tokenize(doc)) for doc in corpus)
            avgdl = total_len / N if N > 0 else doc_len
        
        if idf_cache is not None:
            idf = idf_cache
        else:
            # Calculate IDF using BM25 formula: log((N - df + 0.5) / (df + 0.5))
            df = Counter()
            for doc in corpus:
                doc_terms = set(_tokenize(doc))
                for term in doc_terms:
                    df[term] += 1
            idf = {}
            for term, df_count in df.items():
                # BM25 IDF formula (with smoothing to avoid negative values)
                idf[term] = math.log((N - df_count + 0.5) / (df_count + 0.5) + 1)
    else:
        # Fallback: use chunk length as avgdl, uniform IDF
        avgdl = doc_len
        idf = {}
        N = 1
    
    # Calculate BM25 score
    score = 0.0
    query_term_set = set(query_tokens)
    
    for term in query_term_set:
        if term in chunk_tf:
            f = chunk_tf[term]  # Term frequency in document
            term_idf = idf.get(term, math.log(2))  # Default IDF if not in corpus
            
            # BM25 term score formula
            numerator = f * (k1 + 1)
            denominator = f + k1 * (1 - b + b * (doc_len / avgdl))
            score += term_idf * (numerator / denominator)
    
    # Normalize to 0-1 range
    # Max possible score: all query terms present with max frequency
    max_term_score = (k1 + 1)  # When f is very large, score approaches (k1+1) × IDF
    max_idf = max(idf.values()) if idf else math.log(2)
    max_possible = len(query_term_set) * max_term_score * max_idf
    
    if max_possible > 0:
        return min(1.0, score / max_possible)
    return 0.0


# Pre-computed corpus statistics for efficient batch processing
class KeywordScorer:
    """
    Stateful keyword scorer that pre-computes corpus statistics.
    
    Use this for batch scoring against the same corpus for efficiency.
    """
    
    def __init__(
        self, 
        corpus: List[str], 
        method: KeywordMethod = KeywordMethod.BM25,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize scorer with corpus statistics.
        
        Args:
            corpus: List of document texts
            method: Scoring method to use
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.corpus = corpus
        self.method = method
        self.k1 = k1
        self.b = b
        
        # Pre-compute corpus statistics
        self.N = len(corpus)
        self.df = Counter()  # Document frequency
        total_len = 0
        
        for doc in corpus:
            doc_tokens = _tokenize(doc)
            total_len += len(doc_tokens)
            for term in set(doc_tokens):
                self.df[term] += 1
        
        self.avgdl = total_len / self.N if self.N > 0 else 1.0
        
        # Pre-compute IDF values
        self.idf_tfidf = {
            term: math.log(self.N / df_count) if df_count > 0 else 0
            for term, df_count in self.df.items()
        }
        self.idf_bm25 = {
            term: math.log((self.N - df_count + 0.5) / (df_count + 0.5) + 1)
            for term, df_count in self.df.items()
        }
    
    def score(self, query: str, chunk: str) -> float:
        """
        Score a query-chunk pair using the configured method.
        
        Args:
            query: Query text
            chunk: Document chunk text
            
        Returns:
            Similarity score between 0 and 1
        """
        if self.method == KeywordMethod.JACCARD:
            return jaccard_overlap_similarity(query, chunk)
        elif self.method == KeywordMethod.TFIDF:
            return tfidf_similarity(
                query, chunk, 
                corpus=self.corpus,
                idf_cache=self.idf_tfidf
            )
        elif self.method == KeywordMethod.BM25:
            return bm25_similarity(
                query, chunk,
                corpus=self.corpus,
                k1=self.k1,
                b=self.b,
                avgdl=self.avgdl,
                idf_cache=self.idf_bm25
            )
        else:
            raise ValueError(f"Unknown keyword method: {self.method}")


def keyword_similarity(
    query: str, 
    chunk: str,
    method: KeywordMethod = KeywordMethod.JACCARD,
    corpus: Optional[List[str]] = None,
    k1: float = 1.5,
    b: float = 0.75
) -> float:
    """
    Calculate keyword-based similarity using the specified method.
    
    Available methods:
    - JACCARD: Simple Jaccard + overlap ratio (fast, no corpus needed)
    - TFIDF: True TF-IDF scoring (benefits from corpus for IDF)
    - BM25: Best Matching 25 (best quality, benefits from corpus)
    
    For batch processing against the same corpus, use KeywordScorer class
    for better performance (pre-computes IDF values).
    
    Args:
        query: Query text
        chunk: Document chunk text
        method: Scoring method to use
        corpus: Optional corpus for IDF calculation (TFIDF/BM25)
        k1: BM25 k1 parameter (term frequency saturation)
        b: BM25 b parameter (document length normalization)
        
    Returns:
        Similarity score between 0 and 1
    """
    if method == KeywordMethod.JACCARD:
        return jaccard_overlap_similarity(query, chunk)
    elif method == KeywordMethod.TFIDF:
        return tfidf_similarity(query, chunk, corpus=corpus)
    elif method == KeywordMethod.BM25:
        return bm25_similarity(query, chunk, corpus=corpus, k1=k1, b=b)
    else:
        raise ValueError(f"Unknown keyword method: {method}")


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).
    
    RRF is a robust rank aggregation method that doesn't require score normalization.
    
    Formula: RRF_score(d) = Σ 1/(k + rank_i(d))
    
    Args:
        ranked_lists: List of ranked result lists, each containing (chunk, score) tuples
                     Results should be sorted by score descending (best first)
        k: Ranking constant (default 60, standard value from original RRF paper)
        
    Returns:
        List of (chunk, rrf_score) tuples, sorted by RRF score descending
    """
    rrf_scores: Dict[str, float] = {}
    
    for ranked_list in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked_list, start=1):
            if chunk not in rrf_scores:
                rrf_scores[chunk] = 0.0
            rrf_scores[chunk] += 1.0 / (k + rank)
    
    # Sort by RRF score descending
    results = [(chunk, score) for chunk, score in rrf_scores.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def retrieve_keyword_ranked(
    query: str,
    corpus: List[str],
    top_k: int = 100,
    method: KeywordMethod = KeywordMethod.BM25,
    k1: float = 1.5,
    b: float = 0.75
) -> List[Tuple[str, float]]:
    """
    Retrieve documents using keyword search only, returning ranked results.
    
    Args:
        query: Search query
        corpus: List of document texts
        top_k: Number of top results to return
        method: Keyword scoring method (BM25, TFIDF, JACCARD)
        k1: BM25 k1 parameter
        b: BM25 b parameter
        
    Returns:
        List of (chunk, score) tuples sorted by score descending
    """
    # Initialize scorer with corpus statistics
    scorer = KeywordScorer(corpus=corpus, method=method, k1=k1, b=b)
    
    # Score all documents
    scored = [(chunk, scorer.score(query, chunk)) for chunk in corpus]
    
    # Sort by score descending and return top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def expand_query(
    query: str, 
    language_model: str, 
    num_alternatives: int = 2
) -> List[str]:
    """
    Expand query with related terms using the LLM.
    
    Args:
        query: Original query
        language_model: Name of the language model to use
        num_alternatives: Number of alternative queries to generate
        
    Returns:
        List of queries including original and alternatives
    """
    expansion_prompt = f"""Given the following question, generate 2-3 alternative phrasings or related questions that would help find relevant information. 
Return only the alternative questions, one per line, without numbering or bullets.

Original question: {query}

Alternative questions:"""
    
    try:
        response = ollama.chat(
            model=language_model,
            messages=[{'role': 'user', 'content': expansion_prompt}],
        )
        alternatives = [
            line.strip() 
            for line in response['message']['content'].strip().split('\n') 
            if line.strip()
        ]
        return [query] + alternatives[:num_alternatives]
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}, using original query")
        return [query]


def _parse_rerank_json(response_text: str, expected_ids: List[str]) -> Dict[str, float]:
    """
    Parse and validate JSON response from reranking LLM.
    
    Args:
        response_text: Raw response text from LLM
        expected_ids: List of expected passage IDs
        
    Returns:
        Dictionary mapping passage IDs to scores (0-100 scale)
    """
    scores = {}
    original_response = response_text
    
    # Try to extract JSON from the response
    # Remove markdown code blocks if present
    response_text = response_text.strip()
    if response_text.startswith("```"):
        # Remove markdown code blocks
        response_text = re.sub(r'```(?:json)?\s*\n?', '', response_text, flags=re.MULTILINE)
        response_text = re.sub(r'\n?```\s*$', '', response_text, flags=re.MULTILINE)
        response_text = response_text.strip()
    
    # Try multiple strategies to find JSON
    json_candidates = []
    
    # Strategy 1: Try parsing the entire response as JSON first (most reliable)
    json_candidates.append(response_text)
    
    # Strategy 2: Look for JSON object with "results" key (handle nested braces)
    # Find the first { and try to match balanced braces
    brace_start = response_text.find('{')
    if brace_start != -1:
        brace_count = 0
        for i in range(brace_start, len(response_text)):
            if response_text[i] == '{':
                brace_count += 1
            elif response_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_candidates.append(response_text[brace_start:i+1])
                    break
    
    # Strategy 3: Look for JSON array (handle nested brackets)
    bracket_start = response_text.find('[')
    if bracket_start != -1:
        bracket_count = 0
        for i in range(bracket_start, len(response_text)):
            if response_text[i] == '[':
                bracket_count += 1
            elif response_text[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    json_candidates.append(response_text[bracket_start:i+1])
                    break
    
    # Try parsing each candidate
    for candidate in json_candidates:
        try:
            data = json.loads(candidate)
            
            # Handle different JSON structures
            if isinstance(data, dict) and "results" in data:
                results = data["results"]
            elif isinstance(data, list):
                results = data
            elif isinstance(data, dict):
                # Maybe the dict itself contains id/score pairs
                results = [data]
            else:
                results = []
            
            # Extract scores
            for item in results:
                if isinstance(item, dict):
                    passage_id = item.get("id") or item.get("passage_id") or item.get("passageId")
                    score = item.get("score")
                    
                    if passage_id and score is not None:
                        # Keep score in 0-100 range
                        if isinstance(score, (int, float)):
                            scores[str(passage_id)] = max(0.0, min(100.0, float(score)))
            
            if scores:
                break  # Successfully parsed, stop trying other candidates
                
        except json.JSONDecodeError:
            continue  # Try next candidate
    
    # If JSON parsing failed, try regex extraction as fallback
    if not scores:
        logger.debug(f"JSON parsing failed, trying regex fallback. Response preview: {original_response[:200]}")
        
        # Try to extract individual score pairs with more flexible patterns
        for passage_id in expected_ids:
            # Pattern 1: "p0": 85 or "id": "p0", "score": 85
            patterns = [
                rf'["\']?{re.escape(passage_id)}["\']?\s*[:=]\s*(\d+(?:\.\d+)?)',  # "p0": 85
                rf'"id"\s*:\s*["\']?{re.escape(passage_id)}["\']?\s*,\s*"score"\s*:\s*(\d+(?:\.\d+)?)',  # {"id": "p0", "score": 85}
                rf'"score"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*"id"\s*:\s*["\']?{re.escape(passage_id)}["\']?',  # {"score": 85, "id": "p0"}
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    scores[passage_id] = max(0.0, min(100.0, score))
                    break
    
    if not scores:
        logger.warning(
            f"Could not parse any scores from response. "
            f"Response length: {len(original_response)}, "
            f"Preview: {original_response[:500]}"
        )
    
    return scores


def rerank_with_llm(
    query: str,
    candidates: List[Tuple[str, float, float]],
    language_model: str,
    rerank_weight: float = 0.8,
    original_score_weight: float = 0.2,
    max_retries: int = 1
) -> List[Tuple[str, float]]:
    """
    Rerank candidates using LLM to score relevance with strict JSON output.
    
    Uses stable passage IDs (p0, p1, p2, ...) and enforces strict JSON schema
    to ensure unambiguous score-to-passage mapping. Includes validation and
    repair logic with optional retry for missing items.
    
    This is more accurate than pure embedding similarity as the LLM
    can understand context and nuance better.
    
    Args:
        query: User query
        candidates: List of (chunk, combined_score, semantic_score) tuples
        language_model: Name of the language model to use
        rerank_weight: Weight for rerank score in final calculation
        original_score_weight: Weight for original score in final calculation
        max_retries: Maximum number of retries for missing scores
        
    Returns:
        List of (chunk, final_score) tuples, sorted by score descending
    """
    if not candidates:
        return []
    
    # Step 1: Assign stable IDs to each passage
    passage_ids = [f"p{i}" for i in range(len(candidates))]
    candidate_texts = [chunk.strip() for chunk, _, _ in candidates]
    
    # Step 2: Build prompt with strict JSON schema
    passages_text = "\n".join([
        f"[{pid}] {text}" 
        for pid, text in zip(passage_ids, candidate_texts)
    ])
    
    rerank_prompt = f"""You are a reranker. Score how relevant each passage is to the query.

CRITICAL: You must respond with ONLY valid JSON. No explanations, no markdown, no text before or after.

Required JSON format:
{{
  "results": [
    {{"id": "p0", "score": 85}},
    {{"id": "p1", "score": 72}},
    {{"id": "p2", "score": 45}}
  ]
}}

Rules:
1. Output ONLY the JSON object - no markdown code blocks, no explanations
2. You MUST include exactly one entry for each passage ID: {', '.join(passage_ids)}
3. Each entry must have "id" (exact string from list above) and "score" (integer 0-100)
4. Use the exact passage IDs provided (case-sensitive)
5. Score: 0 = not relevant, 100 = highly relevant

Query: {query}

Passages:
{passages_text}

Remember: Output ONLY the JSON, nothing else."""
    
    # Step 3: Call LLM and parse response
    scores = {}
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            response = ollama.chat(
                model=language_model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a JSON-only API. Always respond with valid JSON only. No explanations, no markdown, no additional text.'
                    },
                    {'role': 'user', 'content': rerank_prompt}
                ],
            )
            
            response_text = response['message']['content'].strip()
            print(response_text)
            # Log the raw response for debugging (first 500 chars)
            if retry_count == 0:  # Only log on first attempt to avoid spam
                logger.debug(f"LLM rerank response (first 500 chars): {response_text[:500]}")
            
            parsed_scores = _parse_rerank_json(response_text, passage_ids)
            scores.update(parsed_scores)
            print(parsed_scores)
            # Check for missing scores
            missing_ids = [pid for pid in passage_ids if pid not in scores]
            
            if not missing_ids:
                # All scores present, we're done
                break
            elif retry_count < max_retries:
                # Retry only for missing items
                logger.warning(
                    f"Missing scores for {len(missing_ids)} passages: {missing_ids}. "
                    f"Retrying (attempt {retry_count + 1}/{max_retries})..."
                )
                
                # Build retry prompt for missing items only
                # Create mapping for efficient lookup
                id_to_text = dict(zip(passage_ids, candidate_texts))
                missing_passages = [
                    f"[{pid}] {id_to_text[pid]}"
                    for pid in missing_ids
                ]
                
                # Generate example JSON with up to 3 missing IDs for clarity
                example_entries = [
                    f'    {{"id": "{pid}", "score": 85}}'
                    for pid in missing_ids[:3]
                ]
                example_json = ",\n".join(example_entries)
                if len(missing_ids) > 3:
                    example_json += "\n    // ... (include all passage IDs)"
                
                retry_prompt = f"""You are a reranker. Score how relevant each passage is to the query.

CRITICAL: You must respond with ONLY valid JSON. No explanations, no markdown, no text before or after.

Required JSON format:
{{
  "results": [
{example_json}
  ]
}}

Rules:
1. Output ONLY the JSON object - no markdown code blocks, no explanations
2. You MUST include exactly one entry for each passage ID: {', '.join(missing_ids)}
3. Each entry must have "id" (exact string from list above) and "score" (integer 0-100)
4. Use the exact passage IDs provided (case-sensitive)
5. Score: 0 = not relevant, 100 = highly relevant

Query: {query}

Passages:
{chr(10).join(missing_passages)}

Remember: Output ONLY the JSON, nothing else."""
                
                retry_response = ollama.chat(
                    model=language_model,
                    messages=[
                        {
                            'role': 'system',
                            'content': 'You are a JSON-only API. Always respond with valid JSON only. No explanations, no markdown, no additional text.'
                        },
                        {'role': 'user', 'content': retry_prompt}
                    ],
                )
                
                retry_scores = _parse_rerank_json(
                    retry_response['message']['content'].strip(), 
                    missing_ids
                )
                scores.update(retry_scores)
                
                # Check if we still have missing scores
                still_missing = [pid for pid in missing_ids if pid not in scores]
                if still_missing:
                    logger.warning(
                        f"Still missing scores after retry: {still_missing}. "
                        f"Using original scores as fallback."
                    )
                break
            else:
                # Max retries reached, use fallback
                logger.warning(
                    f"Max retries reached. Missing scores for: {missing_ids}. "
                    f"Using original scores as fallback."
                )
                break
                
        except Exception as e:
            logger.error(f"Reranking failed: {e}, using original scores")
            break
        
        retry_count += 1
    
    # Step 4: Combine scores with candidates
    reranked = []
    
    for i, (chunk, original_score, semantic_score) in enumerate(candidates):
        passage_id = passage_ids[i]
        
        if passage_id in scores:
            rerank_score_100 = scores[passage_id]  # Score in 0-100 range
            # Normalize to 0-1 for combination with original_score (which is 0-1)
            rerank_score = rerank_score_100 / 100.0
        else:
            # Fallback to original score if missing
            rerank_score = original_score
            logger.debug(
                f"Using original score ({original_score:.3f}) for passage {passage_id}"
            )
        
        # Combine original semantic score (0-1) with rerank score (normalized to 0-1)
        final_score = rerank_score * rerank_weight + original_score * original_score_weight
        reranked.append((chunk, final_score))
    
    # Step 5: Sort by final score
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


def retrieve(
    query: str,
    vector_db: List[Tuple[str, List[float]]],
    embedding_model: str,
    language_model: str,
    top_n: int = 3,
    retrieve_k: int = 20,
    use_reranking: bool = True,
    use_hybrid: bool = True,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    rerank_weight: float = 0.8,
    original_score_weight: float = 0.2,
    keyword_method: KeywordMethod = KeywordMethod.BM25,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> List[Tuple[str, float]]:
    """
    Retrieve relevant chunks using semantic and/or keyword search with optional reranking.
    
    Args:
        query: User query
        vector_db: List of (chunk, embedding) tuples
        embedding_model: Name of the embedding model
        language_model: Name of the language model for reranking
        top_n: Final number of results to return
        retrieve_k: Number of candidates to retrieve before reranking
        use_reranking: Whether to use LLM-based reranking
        use_hybrid: Whether to combine semantic and keyword search
        semantic_weight: Weight for semantic score in hybrid search
        keyword_weight: Weight for keyword score in hybrid search
        rerank_weight: Weight for rerank score in final calculation
        original_score_weight: Weight for original score in final calculation
        keyword_method: Method for keyword similarity (JACCARD, TFIDF, BM25)
        bm25_k1: BM25 term frequency saturation parameter (typically 1.2-2.0)
        bm25_b: BM25 document length normalization (0=none, 1=full)
        
    Returns:
        List of (chunk, score) tuples, sorted by score descending
    """
    # Step 1: Generate query embedding
    try:
        query_embedding = ollama.embed(
            model=embedding_model, 
            input=query
        )['embeddings'][0]
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        raise
    
    # Step 2: Initialize keyword scorer if using hybrid search
    # Pre-compute corpus statistics for efficient batch scoring
    keyword_scorer = None
    if use_hybrid:
        corpus = [chunk for chunk, _ in vector_db]
        keyword_scorer = KeywordScorer(
            corpus=corpus,
            method=keyword_method,
            k1=bm25_k1,
            b=bm25_b
        )
        logger.debug(f"Using keyword method: {keyword_method.value}")
    
    # Step 3: Score all candidates
    candidates = []
    for chunk, embedding in vector_db:
        semantic_score = cosine_similarity(query_embedding, embedding)
        
        if use_hybrid and keyword_scorer:
            keyword_score = keyword_scorer.score(query, chunk)
            combined_score = semantic_score * semantic_weight + keyword_score * keyword_weight
        else:
            combined_score = semantic_score
        
        candidates.append((chunk, combined_score, semantic_score))
    
    # Step 4: Sort and take top retrieve_k
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:retrieve_k]
    
    # Step 5: Reranking using LLM (if enabled)
    if use_reranking:
        # Only rerank if we have more candidates than needed
        # (if we have exactly top_n or fewer, reranking won't change the results)
        if len(top_candidates) > top_n:
            reranked = rerank_with_llm(
                query,
                top_candidates,
                language_model,
                rerank_weight,
                original_score_weight
            )
            return reranked[:top_n]
        else:
            # Not enough candidates to rerank, return as-is
            logger.debug(
                f"Skipping reranking: only {len(top_candidates)} candidates "
                f"(need {top_n + 1} or more to rerank)"
            )
            return [(chunk, score) for chunk, score, _ in top_candidates[:top_n]]
    else:
        # Reranking disabled, return top candidates directly
        return [(chunk, score) for chunk, score, _ in top_candidates[:top_n]]


def retrieve_hybrid_rrf(
    query: str,
    vector_db,  # VectorDB instance with search() method
    language_model: str,
    top_n: int = 3,
    retrieve_k: int = 50,
    fusion_k: int = 20,
    use_reranking: bool = False,
    keyword_method: KeywordMethod = KeywordMethod.BM25,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    rrf_k: int = 60,
    rerank_weight: float = 0.8,
    original_score_weight: float = 0.2,
) -> List[Tuple[str, float]]:
    """
    Retrieve relevant chunks using RRF-based hybrid search.
    
    This method:
    1. Uses ChromaDB's native HNSW search for semantic ranking (fast, O(log n))
    2. Uses BM25 for keyword ranking (scans corpus)
    3. Fuses both rankings using Reciprocal Rank Fusion (RRF)
    4. Optionally reranks with LLM
    
    RRF advantages over weighted scoring:
    - No need to tune weights
    - Robust to different score distributions
    - Proven in production (Elasticsearch, etc.)
    
    Args:
        query: User query
        vector_db: VectorDB instance with search() and get_all() methods
        language_model: Name of the language model for reranking
        top_n: Final number of results to return
        retrieve_k: Number of candidates from each search method (semantic/keyword)
        fusion_k: Number of candidates after RRF fusion (rerank pool)
        use_reranking: Whether to use LLM-based reranking
        keyword_method: Method for keyword scoring (BM25, TFIDF, JACCARD)
        bm25_k1: BM25 k1 parameter
        bm25_b: BM25 b parameter
        rrf_k: RRF constant (default 60)
        rerank_weight: Weight for rerank score if reranking enabled
        original_score_weight: Weight for original score if reranking enabled
        
    Returns:
        List of (chunk, score) tuples, sorted by score descending
    """
    # Step 1: Semantic search using ChromaDB's native HNSW (fast)
    logger.debug(f"Running semantic search for: {query[:50]}...")
    semantic_results = vector_db.search(query, n_results=retrieve_k)
    logger.debug(f"Semantic search returned {len(semantic_results)} results")
    
    # Step 2: Keyword search using BM25 (scans corpus)
    logger.debug(f"Running keyword search with {keyword_method.value}...")
    all_docs = vector_db.get_all()
    corpus = [chunk for chunk, _ in all_docs]
    
    keyword_results = retrieve_keyword_ranked(
        query=query,
        corpus=corpus,
        top_k=retrieve_k,
        method=keyword_method,
        k1=bm25_k1,
        b=bm25_b
    )
    logger.debug(f"Keyword search returned {len(keyword_results)} results")
    
    # Step 3: Fuse with RRF
    fused_results = reciprocal_rank_fusion(
        ranked_lists=[semantic_results, keyword_results],
        k=rrf_k
    )
    logger.debug(f"RRF fusion produced {len(fused_results)} unique results")
    
    # Step 4: Take top candidates for potential reranking
    top_candidates = fused_results[:fusion_k]
    
    # Step 5: Optional LLM reranking
    if use_reranking and len(top_candidates) > top_n:
        logger.debug(f"Reranking top {len(top_candidates)} candidates...")
        # Convert to format expected by rerank_with_llm: (chunk, combined_score, semantic_score)
        # Use RRF score as both combined and semantic score
        candidates_for_rerank = [
            (chunk, score, score) for chunk, score in top_candidates
        ]
        reranked = rerank_with_llm(
            query,
            candidates_for_rerank,
            language_model,
            rerank_weight,
            original_score_weight
        )
        return reranked[:top_n]
    
    return top_candidates[:top_n]
