# Parameter Tuning Report

Benchmark results on RAGQArena Tech dataset (28k documents, 300 eval examples).

## Summary

**Recommendation: Use semantic search only (no reranking)**

Reranking provides marginal quality improvement (~2% R@3) at significant latency cost (35-90x slower).

## Benchmark Results

### Semantic Search vs Reranking

| Config | R@1 | R@3 | R@5 | MRR | Latency |
|--------|-----|-----|-----|-----|---------|
| semantic_only | 0.260 | 0.477 | 0.567 | 0.391 | 15ms |
| semantic+bge (k=20) | 0.290 | 0.493 | 0.560 | 0.407 | 536ms |

**Finding**: BGE reranker improves R@3 by +1.6% but adds 35x latency.

### Reranker Pool Size (fusion_k)

| fusion_k | R@3 | R@5 | Latency | vs Baseline |
|----------|-----|-----|---------|-------------|
| 10 | 0.477 | 0.567 | 16ms | (no rerank) |
| 20 | 0.493 | 0.560 | 544ms | +1.6% R@3 |
| 30 | 0.497 | 0.560 | 825ms | +2.0% R@3 |
| 50 | 0.503 | 0.560 | 1415ms | +2.6% R@3 |

**Finding**: Larger pools improve quality marginally but latency scales linearly.

## Configuration Defaults

Based on benchmarks, defaults are set to:

```python
use_reranking = False      # Disabled - marginal benefit, high latency cost
use_hybrid_search = True   # Enabled - BM25+semantic fusion helps
fusion_k = 20              # If reranking enabled, 20 is the sweet spot
```

## When to Enable Reranking

Enable `--rerank` only when:
1. Quality is critical (every 1-2% R@3 matters)
2. Latency budget allows 500ms+ per query
3. Use fusion_k=20 (best quality/latency tradeoff)

## Hardware Notes

- **MPS (Apple Silicon)**: No significant speedup observed for small batch sizes
- **CPU**: Primary bottleneck; inference time ~27ms per candidate
- **GPU (CUDA)**: Would likely provide 2-3x speedup for larger pools
