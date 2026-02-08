"""
Retrieval module for semantic search and reranking.

This module implements retrieval strategies including:
- Semantic search using ChromaDB's HNSW index
- BM25 keyword search
- Hybrid search with RRF fusion
- LLM-based reranking
- Query expansion
"""

import json
import logging
import math
import re
from collections import Counter
from typing import List, Tuple, Optional, Dict

import ollama

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    return re.findall(r'\b\w+\b', text.lower())


class BM25Scorer:
    """
    BM25 scorer with pre-computed corpus statistics.
    
    BM25 (Best Matching 25) is the industry standard for keyword search,
    used by Elasticsearch, Lucene, and other production systems.
    """
    
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """
        Initialize scorer with corpus statistics.
        
        Args:
            corpus: List of document texts
            k1: Term frequency saturation (typically 1.2-2.0)
            b: Document length normalization (0=none, 1=full)
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        
        # Pre-compute corpus statistics
        self.N = len(corpus)
        self.df: Counter = Counter()  # Document frequency
        total_len = 0
        
        for doc in corpus:
            doc_tokens = _tokenize(doc)
            total_len += len(doc_tokens)
            for term in set(doc_tokens):
                self.df[term] += 1
        
        self.avgdl = total_len / self.N if self.N > 0 else 1.0
        
        # Pre-compute IDF values using BM25 formula
        self.idf = {
            term: math.log((self.N - df_count + 0.5) / (df_count + 0.5) + 1)
            for term, df_count in self.df.items()
        }
    
    def score(self, query: str, chunk: str) -> float:
        """
        Calculate BM25 score for a query-chunk pair.
        
        Args:
            query: Query text
            chunk: Document chunk text
            
        Returns:
            BM25 score (normalized to 0-1 range)
        """
        query_tokens = _tokenize(query)
        chunk_tokens = _tokenize(chunk)
        
        if not query_tokens or not chunk_tokens:
            return 0.0
        
        doc_len = len(chunk_tokens)
        chunk_tf = Counter(chunk_tokens)
        
        score = 0.0
        for term in set(query_tokens):
            if term in chunk_tf:
                f = chunk_tf[term]
                term_idf = self.idf.get(term, math.log(2))
                
                # BM25 formula
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += term_idf * (numerator / denominator)
        
        # Normalize to 0-1 range
        max_term_score = (self.k1 + 1)
        max_idf = max(self.idf.values()) if self.idf else math.log(2)
        max_possible = len(set(query_tokens)) * max_term_score * max_idf
        
        if max_possible > 0:
            return min(1.0, score / max_possible)
        return 0.0


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    rrf_weight: float = 0.7
) -> List[Tuple[str, float]]:
    """
    Combine multiple ranked lists using Weighted Reciprocal Rank Fusion (RRF).
    
    RRF is a robust rank aggregation method. It doesn't require score normalization.
    
    Formula: RRF_score(d) = Σ weight_i * 1/(k + rank_i(d))
    
    Args:
        ranked_lists: List of ranked result lists, each containing (chunk, score) tuples
        k: Ranking constant (default 60, from original RRF paper)
        rrf_weight: Weight for semantic (first) list; BM25 (second) gets 1 - rrf_weight
                   Default 0.7 means semantic=0.7, BM25=0.3
        
    Returns:
        List of (chunk, rrf_score) tuples, sorted by score descending
    """
    rrf_scores: Dict[str, float] = {}
    
    # Calculate weights: first list gets rrf_weight, second gets (1 - rrf_weight)
    # For >2 lists, remaining lists split the second weight equally
    if len(ranked_lists) == 2:
        weights = [rrf_weight, 1 - rrf_weight]
    else:
        weights = [1.0 / len(ranked_lists)] * len(ranked_lists)
    
    for list_idx, ranked_list in enumerate(ranked_lists):
        weight = weights[list_idx] if list_idx < len(weights) else weights[-1]
        for rank, (chunk, _) in enumerate(ranked_list, start=1):
            if chunk not in rrf_scores:
                rrf_scores[chunk] = 0.0
            rrf_scores[chunk] += weight * (1.0 / (k + rank))
    
    results = [(chunk, score) for chunk, score in rrf_scores.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def retrieve_bm25(
    query: str,
    corpus: List[str],
    top_k: int = 50,
    k1: float = 1.5,
    b: float = 0.75
) -> List[Tuple[str, float]]:
    """
    Retrieve documents using BM25 keyword search.
    
    Args:
        query: Search query
        corpus: List of document texts
        top_k: Number of top results to return
        k1: BM25 k1 parameter
        b: BM25 b parameter
        
    Returns:
        List of (chunk, score) tuples sorted by score descending
    """
    scorer = BM25Scorer(corpus=corpus, k1=k1, b=b)
    scored = [(chunk, scorer.score(query, chunk)) for chunk in corpus]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def expand_query(
    query: str, 
    language_model: str, 
    num_alternatives: int = 2
) -> List[str]:
    """
    Expand query with alternative phrasings using LLM.
    
    Args:
        query: Original query
        language_model: Name of the language model
        num_alternatives: Number of alternatives to generate
        
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
    
    # Remove markdown code blocks if present
    response_text = response_text.strip()
    if response_text.startswith("```"):
        response_text = re.sub(r'```(?:json)?\s*\n?', '', response_text, flags=re.MULTILINE)
        response_text = re.sub(r'\n?```\s*$', '', response_text, flags=re.MULTILINE)
        response_text = response_text.strip()
    
    # Try multiple strategies to find JSON
    json_candidates = [response_text]
    
    # Find balanced braces
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
    
    # Find balanced brackets
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
            
            if isinstance(data, dict) and "results" in data:
                results = data["results"]
            elif isinstance(data, list):
                results = data
            elif isinstance(data, dict):
                results = [data]
            else:
                results = []
            
            for item in results:
                if isinstance(item, dict):
                    passage_id = item.get("id") or item.get("passage_id") or item.get("passageId")
                    score = item.get("score")
                    
                    if passage_id and score is not None:
                        if isinstance(score, (int, float)):
                            scores[str(passage_id)] = max(0.0, min(100.0, float(score)))
            
            if scores:
                break
                
        except json.JSONDecodeError:
            continue
    
    # Regex fallback
    if not scores:
        logger.debug(f"JSON parsing failed, trying regex fallback")
        for passage_id in expected_ids:
            patterns = [
                rf'["\']?{re.escape(passage_id)}["\']?\s*[:=]\s*(\d+(?:\.\d+)?)',
                rf'"id"\s*:\s*["\']?{re.escape(passage_id)}["\']?\s*,\s*"score"\s*:\s*(\d+(?:\.\d+)?)',
                rf'"score"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*"id"\s*:\s*["\']?{re.escape(passage_id)}["\']?',
            ]
            for pattern in patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    scores[passage_id] = max(0.0, min(100.0, float(match.group(1))))
                    break
    
    if not scores:
        logger.warning(f"Could not parse scores from response. Preview: {original_response[:500]}")
    
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
    Rerank candidates using LLM to score relevance.
    
    Args:
        query: User query
        candidates: List of (chunk, combined_score, semantic_score) tuples
        language_model: Name of the language model
        rerank_weight: Weight for rerank score
        original_score_weight: Weight for original score
        max_retries: Maximum retries for missing scores
        
    Returns:
        List of (chunk, final_score) tuples, sorted by score descending
    """
    if not candidates:
        return []
    
    passage_ids = [f"p{i}" for i in range(len(candidates))]
    candidate_texts = [chunk.strip() for chunk, _, _ in candidates]
    
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
    
    scores = {}
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            response = ollama.chat(
                model=language_model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a JSON-only API. Always respond with valid JSON only.'
                    },
                    {'role': 'user', 'content': rerank_prompt}
                ],
            )
            
            response_text = response['message']['content'].strip()
            if retry_count == 0:
                logger.debug(f"LLM rerank response: {response_text[:500]}")
            
            parsed_scores = _parse_rerank_json(response_text, passage_ids)
            scores.update(parsed_scores)
            
            missing_ids = [pid for pid in passage_ids if pid not in scores]
            
            if not missing_ids:
                break
            elif retry_count < max_retries:
                logger.warning(f"Missing scores for {missing_ids}, retrying...")
                
                id_to_text = dict(zip(passage_ids, candidate_texts))
                missing_passages = [f"[{pid}] {id_to_text[pid]}" for pid in missing_ids]
                
                retry_prompt = f"""Score these passages for the query. Output ONLY JSON.

Query: {query}

Passages:
{chr(10).join(missing_passages)}

Format: {{"results": [{{"id": "p0", "score": 85}}]}}"""
                
                retry_response = ollama.chat(
                    model=language_model,
                    messages=[
                        {'role': 'system', 'content': 'You are a JSON-only API.'},
                        {'role': 'user', 'content': retry_prompt}
                    ],
                )
                
                retry_scores = _parse_rerank_json(retry_response['message']['content'].strip(), missing_ids)
                scores.update(retry_scores)
                break
            else:
                logger.warning(f"Max retries reached. Missing: {missing_ids}")
                break
                
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            break
        
        retry_count += 1
    
    reranked = []
    for i, (chunk, original_score, _) in enumerate(candidates):
        passage_id = passage_ids[i]
        
        if passage_id in scores:
            rerank_score = scores[passage_id] / 100.0
        else:
            rerank_score = original_score
        
        final_score = rerank_score * rerank_weight + original_score * original_score_weight
        reranked.append((chunk, final_score))
    
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


def retrieve(
    query: str,
    vector_db,  # VectorDB instance
    language_model: str,
    top_n: int = 3,
    retrieve_k: int = 50,
    fusion_k: int = 20,
    use_hybrid_search: bool = True,
    use_reranking: bool = False,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    rrf_k: int = 60,
    rrf_weight: float = 0.7,
    rerank_weight: float = 0.8,
    original_score_weight: float = 0.2,
) -> List[Tuple[str, float]]:
    """
    Retrieve relevant chunks using semantic search with optional hybrid (BM25 + RRF).
    
    Args:
        query: User query
        vector_db: VectorDB instance
        language_model: Language model for reranking
        top_n: Final results to return
        retrieve_k: Candidates from each search method
        fusion_k: Candidates after fusion (rerank pool)
        use_hybrid_search: Use semantic + BM25 with RRF fusion (False = semantic only)
        use_reranking: Enable LLM reranking
        bm25_k1: BM25 k1 parameter
        bm25_b: BM25 b parameter
        rrf_k: RRF constant
        rrf_weight: Weight for semantic in RRF (default 0.7); BM25 gets 1 - rrf_weight (0.3)
        rerank_weight: Rerank score weight
        original_score_weight: Original score weight
        
    Returns:
        List of (chunk, score) tuples
    """
    # Step 1: Semantic search (always)
    logger.debug(f"Semantic search for: {query[:50]}...")
    semantic_results = vector_db.search(query, n_results=retrieve_k)
    logger.debug(f"Semantic: {len(semantic_results)} results")
    
    if use_hybrid_search:
        # Step 2: FTS5 keyword search (fast, uses pre-built index)
        logger.debug("FTS5 keyword search...")
        keyword_results = vector_db.search_fts(query=query, n_results=retrieve_k)
        logger.debug(f"FTS5: {len(keyword_results)} results")
        
        # Step 3: RRF fusion (weighted: semantic gets rrf_weight, BM25 gets 1-rrf_weight)
        fused_results = reciprocal_rank_fusion(
            ranked_lists=[semantic_results, keyword_results],
            k=rrf_k,
            rrf_weight=rrf_weight
        )
        logger.debug(f"RRF fusion: {len(fused_results)} unique results")
        
        top_candidates = fused_results[:fusion_k]
    else:
        # Semantic only
        top_candidates = semantic_results[:fusion_k]
    
    # Step 4: Optional reranking
    if use_reranking and len(top_candidates) > top_n:
        logger.debug(f"Reranking {len(top_candidates)} candidates...")
        candidates_for_rerank = [(chunk, score, score) for chunk, score in top_candidates]
        reranked = rerank_with_llm(
            query,
            candidates_for_rerank,
            language_model,
            rerank_weight,
            original_score_weight
        )
        return reranked[:top_n]
    
    return top_candidates[:top_n]
