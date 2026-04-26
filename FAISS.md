---
aliases:
  - Facebook AI Similarity Search
---
Resources:
- [Video: James Briggs FAISS Introduction to Similarity Search](https://youtu.be/sKyvsdEv6rk)
- [[Faiss - The Missing Manual (?) {James Briggs, Pinecone}]]

Pronounced "Fice"

A library from Meta for ==efficient similarity searching over dense vectors==. 
- Given a query vector, find the most similar vectors in the dataset.

Supports ==many index types,== with the choice being a tradeoff between speed, memory, and [[Recall (Information Retrieval)]] accuracy. Supports ==both exact and [[Approximate Nearest Neighbor Search|Approximate Nearest Neighbor]]== indexes/searches!

==FAISS IS NOT A VECTOR DATABASE==. It has no persistence layer, no metadata, no filtering by attributes, no updates. For production vector search with metadata filtering, teams typically use things like [[Weaviate]], [[Pinecone]], or [[pgvector]], which use FAISS-style algorithms internally, but wrap them in a proper database.

Example
```python
import faiss
  import numpy as np

  d = 128          # vector dimension
  n = 1_000_000    # dataset size

  vectors = np.random.rand(n, d).astype('float32')
  query = np.random.rand(1, d).astype('float32')

  index = faiss.IndexFlatL2(d)   # exact L2 search
  index.add(vectors)
  distances, indices = index.search(query, k=10)  # 10 nearest neighbors
```

Typical pipeline at scale:
  - Build time:
	  - vectors → train IVF centroids (k-means) → assign vectors to cells → compress with PQ → save index
  - Query time:
	  - query → find nearest nprobe cells → compute approximate distances in those cells → return top-k

# The FAISS Index Zoo
- FAISS supports many indexes, each with tradeoffs around speed, memory, and recall accuracy.
- ==Flat==:  (==Exact==)
	- `IndexFlatL2` and `IndexFlatIP`: Brute force, exact, slow at scale. Good baseline.
	- GPU Supported
- ==[[Inverted File Index]] (IVF)==: 
	- Clusters dataset into `nlist` [[Voronoi Diagram|Voronoi Cell]]s at build time, and at query time searches only `nprobe` nearest cells. `IndexIVFFlat` is ==Exact== within searched cells. Higher `nprobd` = better recall, slower search.
	- GPU Supported
- ==[[Product Quantization]] (PQ)==: 
	- `IndexPQ`: Compresses vectors into compact codes via [[Product Quantization]]; huge memory reduction ==approximate== distances via lookup tables.
	- NO GPU Support! CPU only
- ==IVFPQ==: 
	- ==The Workhorse.== Combines IVF partitioning and PQ compression, the ==standard choice for billion-scale search.== `IndexIVFPQ(d, nlist, m, nbits)`
	- GPU Support
- ==[[Hierarchical Navigable Small Worlds]] (HNSW)==: 
	- `IndexHNSW` for very fast ==approximate== queries, good recall, but high memory usage and slow build time.
	- ==NO GPU support!==
		- GPU excels at data-parallel operations; the same operation on many elements simultaneously. HNSW graph traversal is serial, each step depends on the previous step's result. You can't parallelize across steps.

Speed/Recall Tradeoff:
```
 Exact (Flat) ←————————————————→ Approximate (IVFPQ)
  100% recall                      ~95% recall
  Slow at scale                    Fast at scale
  High memory                      Low memory
```


Aside on [[Product Quantization]]:
- The key compression trick used in FAISS:
	1. Split each 128-dim vector into `m` subvectors (e.g. 8x16-dim)
	2. For each subvector position, cluster all vectors' subvectors into 256 centroids.
	3. Store each subvector as its centroid ID (1 byte)
	4. A  128-dim float32 vector (512 bytes) -> 8 bytes








