
A [[PostgreSQL]] extension that adds first-class vector data types and similarity search to Postgres.
- It's the de-facto way to do embeddings/AI search without leaving your existing database.

Core data types:
- `vector(n)`: dense float32 vector of dimension n (up to 16,000; indexable up to 2,000)
- `halfvec()`: float16 dense vector. Half the storage, indexable up to 4,000 dimensions.
- `bit(n)`: Binary vector for [[Hamming Distance]]
- `sparsevec(n)`: Sparse vector (stores only non-zero indexes), up to 1,000 non-zero elements indexable.

Adds the following distance operators:
![[Pasted image 20260501155757.png]]
Above: [[Euclidean Distance]], [[Cosine Distance]], [[Manhattan Distance]], [[Hamming Distance]], [[Jaccard Index|Jaccard Similarity]]
- They use "Negative [[Dot Product|Inner Product]]"; Postgres index operators are defined as *distance operators*, so smaller = closer, so `ORDER BY .... LIMMIT k` returns nearest neighbors. 
	- Because inner product is a *similarity*m not a distance, bigger is better, which is the wrong direction for ORDER BY ASC, so they negate it. Now smaller is more similar.

Index Types:
- IVFFlat: Partitions vectors into `lists` clusters ([[K-Means Clustering|K-Means]]), and search `probes` them at query time. Must be built *after* you have representative data.
- [[Hierarchical Navigable Small Worlds|HNSW]]: Graph-based, higher recall, slower build, more memory, but no need to retrain when data changes much. Tunable with `m` and `ef_construction` at build time, `hnsw.ef_search` at query time.


Both [[Neon]] and [[Supbase]] support `pgvector` out of the box.
- `CREATE EXTENSION vector`, and you're done. Supabase has it preinstalled on each project.


Worse than dedicated vector DBs ([[Pinecone]], Qdrant, [[Weaviate]], Milvus) when you have hundreds of millions of vectors. You need very high QPS on a very single index, or you need sophisticated metadata filtering at scale. The gap has narrowed a lot with HNSW + iterative scans, though.








