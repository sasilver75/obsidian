---
aliases:
  - ANN
  - Approximate Nearest Neighbor
---
To retrieve documents with low latency at scale, we use ANN methods that optimize for retrieval speeds, returning the approximate (instead of exact) top k most similar neighbors, trading off a little bit of accuracy for a large speedup.

ANN embedding indices are data structures that let us do ANN searches efficiently; at a high level, they build partitions over embedding space so that we can quickly zoom in on the specific space where the query vector is. Popular ones include:
- [[Locality Sensitive Hashing]] (LSH): The core idea being to create hash functions so that similar items are likely to end up in the same hash bucket. By only needing to check relevant buckets, we can perform ANN queries efficiently.
- [[FAISS]]: Uses a combination of both quantization and indexing for efficient retrieval, supporting both CPU and GPU. Can handle billions of vectors due to efficient use of memory.
- [[Hierarchical Navigable Small Worlds]] (HNSW): Inspired by the "six degrees of separation" idea, builds a hierarchical graph structure that embodies the small world phenomenon. Most nodes can be reached from any other node via a minimum number of hops. This allows HNSW to initiate queries from broader, coarse approximations, and progressively narrow the search at lower levels.
- ==Scalable Nearest Neighbors (ScaNN):== A two-step process. First, coarse quantization reduces the search space, then fine-grained search is done in the reduced set. Eugene Yan said it's some of the best recall/latency trade-off he's seen.

When evaluating ANN indices, some factors to consider include:
- Recall: How does it fare against *exact* nearest neighbors?
- Latency/throughput: How many queries can it handle per second?
- Memory footprint: How much RAM is required to serve an index?
- Ease of adding new items: Can new items be added without having to reindex all documents, or does the index need to be rebuilt?

