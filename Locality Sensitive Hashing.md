---
aliases:
  - LSH
---


![[Pasted image 20240115124418.png]]


From [Pinecone](https://www.pinecone.io/learn/a-developers-guide-to-ann-algorithms/) in May 2042:
> Main types of vector indexing algorithms:
> - Spatial Partitioning (cluster-based indexing, clustering indexes)
> - Graph-based Indexing (eg [[Hierarchical Navigable Small Worlds|HNSW]])
> - Hash-based Indexing (eg [[Locality Sensitive Hashing|LSH]])
> 
> We'll skip hash-based indexing because its performance on all aspects (reads, writes, storage) is currently worse than that of both graph-based and spatial partioning-based indexing.
> - ==Almost no vector databases us hash-based indexing nowadays.==