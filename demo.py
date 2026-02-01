dataset = []
with open('cat-facts.txt', 'r') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')

import ollama
import re
from collections import Counter
from pprint import pprint

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
VECTOR_DB = []

def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  print(f'Added chunk {i+1}/{len(dataset)} to the database')


def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)


def expand_query(query):
  """Expand query with related terms using the LLM"""
  expansion_prompt = f"""Given the following question, generate 2-3 alternative phrasings or related questions that would help find relevant information. 
Return only the alternative questions, one per line, without numbering or bullets.

Original question: {query}

Alternative questions:"""
  
  try:
    response = ollama.chat(
      model=LANGUAGE_MODEL,
      messages=[{'role': 'user', 'content': expansion_prompt}],
    )
    alternatives = [line.strip() for line in response['message']['content'].strip().split('\n') if line.strip()]
    return [query] + alternatives[:2]  # Return original + up to 2 alternatives
  except:
    return [query]  # Fallback to original query if expansion fails


def keyword_similarity(query, chunk):
  """Calculate keyword-based similarity using TF-IDF-like scoring"""
  query_words = set(re.findall(r'\b\w+\b', query.lower()))
  chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))
  
  if not query_words:
    return 0.0
  
  # Jaccard similarity + word overlap
  intersection = len(query_words & chunk_words)
  union = len(query_words | chunk_words)
  jaccard = intersection / union if union > 0 else 0.0
  
  # Word overlap ratio
  overlap_ratio = intersection / len(query_words)
  
  # Combined score
  return (jaccard * 0.3 + overlap_ratio * 0.7)


def retrieve(query, top_n=3, use_reranking=True, use_hybrid=True, retrieve_k=20):
  """
  Improved retrieval with optional reranking and hybrid search
  
  Args:
    query: User query
    top_n: Final number of results to return
    use_reranking: Whether to use LLM-based reranking
    use_hybrid: Whether to combine semantic and keyword search
    retrieve_k: Number of candidates to retrieve before reranking
  """
  # Step 1: Initial retrieval - get more candidates
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  candidates = []
  
  for chunk, embedding in VECTOR_DB:
    semantic_score = cosine_similarity(query_embedding, embedding)
    
    if use_hybrid:
      keyword_score = keyword_similarity(query, chunk)
      # Combine scores (weighted average)
      combined_score = semantic_score * 0.7 + keyword_score * 0.3
    else:
      combined_score = semantic_score
    
    candidates.append((chunk, combined_score, semantic_score))
  
  # Sort by combined score and take top retrieve_k
  candidates.sort(key=lambda x: x[1], reverse=True)
  top_candidates = candidates[:retrieve_k]
  
  # Step 2: Reranking using LLM (if enabled)
  if use_reranking and len(top_candidates) > top_n:
    reranked = rerank_with_llm(query, top_candidates)
    return reranked[:top_n]
  else:
    return [(chunk, score) for chunk, score, _ in top_candidates[:top_n]]


def rerank_with_llm(query, candidates):
  """
  Rerank candidates using LLM to score relevance
  This is more accurate than pure embedding similarity
  """
  # Format candidates for reranking
  candidate_texts = [chunk for chunk, _, _ in candidates]
  
  rerank_prompt = f"""You are a relevance scorer. Given a question and a list of facts, score each fact's relevance to the question on a scale of 0.0 to 1.0.

Question: {query}

Facts:
"""
  for i, fact in enumerate(candidate_texts):
    rerank_prompt += f"{i+1}. {fact.strip()}\n"
  
  rerank_prompt += """
Return ONLY a comma-separated list of scores (one per fact, in order), like: 0.9, 0.7, 0.5, ...
Do not include any other text."""
  
  try:
    response = ollama.chat(
      model=LANGUAGE_MODEL,
      messages=[{'role': 'user', 'content': rerank_prompt}],
    )
    
    # Parse scores
    scores_text = response['message']['content'].strip()

    # Extract numbers from the response
    scores = re.findall(r'0?\.\d+|1\.0|0', scores_text)
    scores = [float(s) for s in scores[:len(candidates)]]
    
    # Validate we got the expected number of scores
    if len(scores) != len(candidates):
      print(f"Warning: Expected {len(candidates)} scores, got {len(scores)}. "
            f"Using original scores for missing ones.")
    
    # Combine with candidates and sort by rerank score
    # If we got fewer scores, we assume they're in order and missing ones are at the end
    # (This is a reasonable assumption since the prompt asks for scores "in order")
    reranked = []
    original_scores = [score for _, score, _ in candidates]
    
    for i, (chunk, original_score, semantic_score) in enumerate(candidates):
      # Use LLM rerank score if available, otherwise fall back to original score
      if i < len(scores):
        rerank_score = scores[i]
        # Validate score is in reasonable range
        rerank_score = max(0.0, min(1.0, rerank_score))
      else:
        # Missing score - use original semantic score as fallback
        rerank_score = original_score
        print(f"  Using original score ({original_score:.3f}) for candidate {i+1}")
      
      # Combine original semantic score with rerank score
      final_score = rerank_score * 0.8 + original_score * 0.2
      reranked.append((chunk, final_score))
    
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked
    
  except Exception as e:
    print(f"Reranking failed: {e}, using original scores")
    # Fallback to original scores
    return [(chunk, score) for chunk, score, _ in candidates]


# Configuration
USE_RERANKING = True
USE_HYBRID_SEARCH = True
RETRIEVE_K = 20  # Retrieve top 20, then rerank to top 3

input_query = input('Ask me a question: ')

# Optional: Query expansion (can be enabled for better recall)
USE_QUERY_EXPANSION = False
if USE_QUERY_EXPANSION:
  expanded_queries = expand_query(input_query)
  print(f"Expanded queries: {expanded_queries}")
  # Use the best results from all expanded queries
  all_results = []
  for q in expanded_queries:
    results = retrieve(q, top_n=5, use_reranking=USE_RERANKING, 
                       use_hybrid=USE_HYBRID_SEARCH, retrieve_k=RETRIEVE_K)
    all_results.extend(results)
  # Deduplicate and take top results
  seen = set()
  unique_results = []
  for chunk, score in all_results:
    if chunk not in seen:
      seen.add(chunk)
      unique_results.append((chunk, score))
  unique_results.sort(key=lambda x: x[1], reverse=True)
  retrieved_knowledge = unique_results[:3]
else:
  retrieved_knowledge = retrieve(input_query, top_n=3, 
                                 use_reranking=USE_RERANKING,
                                 use_hybrid=USE_HYBRID_SEARCH,
                                 retrieve_k=RETRIEVE_K)

print('\nRetrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
  print(f' - (score: {similarity:.3f}) {chunk.strip()}')

context_lines = [chunk.strip() for chunk, _ in retrieved_knowledge]
context_text = '\n'.join([f'{i+1}. {chunk}' for i, chunk in enumerate(context_lines)])

# Improved prompt with better structure
instruction_prompt = f'''You are a helpful and accurate chatbot that answers questions based on provided context.

Context information:
{context_text}

Instructions:
- Answer the question using ONLY the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so clearly
- Do not make up or infer information that isn't in the context
- Be concise but complete in your answer
- Cite which fact(s) you're using when relevant

Question: {input_query}

Answer:'''


stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': 'You are a helpful assistant that provides accurate answers based on given context.'},
    {'role': 'user', 'content': instruction_prompt},
  ],
  stream=True,
)

# print the response from the chatbot in real-time
print('\nChatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
print()  # New line at the end

