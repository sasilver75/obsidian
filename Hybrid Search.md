---
aliases:
  - Hybrid Retrieval
---
Using a traditional search index plus embedding-based search.

While embedding-based search is great in many instances, there are situations in which it falls short, like:
- Searching for a person or object's name (eg Eugene)
- Searching for an acronym or phrase (eg RAG, RLHF)
- Searching for an ID (eg gpt-3.5-turbo, titan-xlarge,v1.01)

And keyword search has limitations too
- Models simple word frequencies, but doesn't capture semantic or correlation information, so it doesn't deal well with synonyms or hypernyms.
One benefit of conventional search indices
- We can use metadata to refine results. We can use data filters to prioritize newer documents or narrow our search to a specific time period, or filtering on average rating of e-commerce products. This is handy for downstream ranking, like prioritizing documents that are cited more.

Thus combining traditional search with semantic search is complementary!

With regard to embeddings:
- The seemingly popular approach is to use `text-embedding-ada-002`
- The OG embedding approaches include [[Word2Vec]] and ==[[fastText]].== The latter is an open-source lightweight library that enables users to leverage pre-trained embeddings or train new embedding models. ==It comes with pre-trained embeddings, and is extremely fast, even without a GPU -- it's Eugene's go-to for early-stage PoCs.==

Another good baseline is [[Sentence Transformers]], which makes it simple to compute embeddings for sentences, paragraphs, and even images. It's a workhorse based on transformers like [[RoBERTa]], and supports 100+ languages.

Consider looking at the [[MTEB]] leaderboard

### How do we retrieve documents with low latency at scale?
- We use [[Approximate Nearest Neighbor Search|Approximate Nearest Neighber]] search, optimizing for retrieval speed and returns the approximate top-k most similar neighbors.
- ANN embedding indices are data structures that let us do ANN searches efficiently. Popular techniques include:
	- [[Locality Sensitive Hashing]]
	- [[FAISS]]
	- [[Hierarchical Navigable Small Worlds|HNSW]]
	- Scalable Nearest Neighbors (ScaNN)

When evaluating an ANN index, consider:
- Recall: How does it fare against exact nearest neighbors?
- Latency/throughput: How many queries can it handle per second?
- Memory footprint: How much RAM is required to serve an index?
- Ease of adding new items: Can new items be added without having to reindex all documents (LSH) or does the index need to be rebuilt (ScaNN)?



## (3/7) Fine-tuning: To get better at specific tasks
- The process of taking a pre-trained model and further refining it on a specific task. The intent is to harness the knowledge the model already has, and apply it to a specific task, usually involving a smaller, task-specific dataset.