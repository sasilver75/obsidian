Jun 10 Conference Talk with Ben Clavié, cracked researchers @ AnswerAI, creator of the RAGatoullie framework, maintains the `rerankers` lib, and comes from a deep background in IR

----

Topics Today (Only half an hour):
- Rant: Retrieval was not invented in December 2022
- The "compact MVP": Bi-encoder single-vector embedding and cosine similarities are all you need
- What's a cross-encoder, and why do I need it?
- TF-IDF and full text search is so 1970s -- surely it's not relevant, right?
- Metadata Filtering: When not all content is potentially useful, don't make it harder than it needs to be!
- Compact MVP++: All of the above, in 30 lines of code!
- Bonus: One vector is good, but what about many vectors (ColBERT)

Counter-Agenda: What we won't be talking about:
- How to systematically monitor and improve RAG systems
- Evaluations: These are too important to be covered quickly!
- Benchmarks/Paper references: We'll make claims, and you'll just trust Ben on them!
- An overview of all of the best performing models
- Synthetic data and training
- Approaches that go beyond the basics (SPLADE, ColBERT)

---

First, a rant!
- RAG is NOT:
	- A new paradigm
	- A framework
	- An end-to-end system
	- Something created by Jason Liu (LlamaIndex) in his endless quest for a Porsche
- It IS:
	- The act of stitching together Retrieval and Generation to ground the latter.
	- Good RAG is made up of good components:
		- Good retrieval pipeline
		- Good generative model
		- Good way of linking them up

"My RAG doesn't work" isn't enough -- when a Car doesnt work, *something specific* doesn't work! If you know enough about RAG, you can diagnose it.

### The Compact MVP
![[Pasted image 20240617192610.png]]
Documents and Queries get embedded into single vectors, and we do some similarity comparison (eg Cosine Similarity) to find relevant results.


![[Pasted image 20240617192819.png]]
- Get your model, get your data, embed your documents. Get a query, embed the query, do similarities (dot product, here), select the top 3, and return those paragraphs.
	- This is all numpy! The point of using a VectorDB is efficiently letting you search through a large number of documents (eg using [[Hierarchical Navigable Small Worlds|HNSW]]) without having to compute cosine similarity against every document; we use an approximate search ([[Approximate Nearest Neighbor Search]]). For many cases, you don't need!

Why are you calling embeddings [[Bi-Encoder]], so far?
- The representation method from the previous slide is commonly referred to as Bi-Encoders
- They're generally used to create single-vector representations; They allow us to pre-compute document representations, because documents and queries are encoded entirely separately (aren't aware of eachother).
	- So at inference time all we need to do is compute the embedding of our query, and start doing comparisons to precomputed document embeddings.


So if Bi-Encoders are really computationally efficient, there must be a tradeoff!
- Your query embedding is unaware of your document(s), and vice versa! There's no rich interaction between the terms within the query and document.

This rich interaction is usually done via [[Cross-Encoder]]s, which are usually also BERT-based models that encode the concatenation of the query and document.
![[Pasted image 20240617195428.png]]
This allows for rich interactions between the terms in the query and document.

![[Pasted image 20240617200943.png]]
For people really into retrieval, there are a bunch of rerankers that are not cross-encoders (eg using LLMs); The core idea though is just to use a powerful, expensive model to rank a *subset* of relevant documents (retrieved by a first step; for us here, via  bi-encoder).
- There are also many models to use that are API-based, like [[Cohere]] rerankers, and others you can run locally.

![[Pasted image 20240617201053.png]]
Here's our pipeline now -- but there's something else that we're still missing...

Semantic search by embeddings is powerful -- we love vectors. But it's very hard for some things; you're asking your model to boil a long passage into an (eg) 512, 1024-dimensioned vector.
- When we train an embedding, we train the model to retain information that's *useful for the training objective/queries*; not to retain *all information* in the passage!

When you then use that embedding model on your *own* (slightly)











