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

Semantic search by embeddings is powerful -- we love vectors. But it's very hard if you think about it; you're asking your model to boil a long passage into a single (eg) 512 or 1024-dimensioned vector, which we hope contains all of the *relevant* information in the document.
- When we train an embedding, we train the model to retain information that's ***useful for the training objective/queries***; not to retain ***all** information* in the passage!
	- (Retaining all of the information would be impossible, since embedding is basically compression)

When you then use that embedding model on your *own* (slightly different, out-of-distribution) data, it's likely that you're going to be missing some *actually relevant* information.
- Especially if you're trying to transfer to some specific domain like law or medicine, with a lot of jargon.

These are reasons why you should also **ALWAYS** have keyword search/full-text search ([[TF-IDF]], [[BM25]]) in your pipeline!
- TF-IDF assigns every word in a document a weight based on how rare they are.
- BM25 was invented in the 70s; "The reason IR hasn't taken off like NLP has is because the baseline is so good." -- and it's just word-counting with a weighting formula!

![[Pasted image 20240618123520.png]]
Comparing 
- Unless you go into very overtrained embeddings, BM25 is competitive with virtually all deep-learning-based approaches (at the time of [[BEIR]] 3 years ago)

Tf-IDF MVP++
![[Pasted image 20240618123557.png]]
Now, we're creating [[Hybrid Search]]
- There are many ways to combine the scores; one way that people often do (lol) is .7 to the vector score and .3 to the full-text search score.

Metadata Filtering
- This really completes your pipeline.
- Outside of academic benchmarks, documents don't exist in a vacuum -- there's a lot of metadata around them, some of which can be very informative!

Given query
> "Can you get me the Cruise division financial report for Q4 2022?"
- There's a lot of ways that semantic search can *fail* here:
	- It might fetch documents that don't meet one or more of the criteria above (Cruise division, financial report, Q4, 2022)
	- If the number of documents you search for (k) is too high, you might be passing *irrelevant reports* back to your LLM, hoping that it manages to figure out which set of numbers is correct!

![[Pasted image 20240618124125.png]]

![[Pasted image 20240618124223.png]]

The final compact MVP++ isn't so scary though -- only about 25 lines of code if you just 


![[Pasted image 20240618124251.png]]
He likes LanceDB, but he tries not to take side in the VectorDB wars, he thinks they all have their place.
- But he thinks that LanceDB is great for MVPs
Steps
- We create our Bi-Encoder
- Define our Document Structure
- Add our documents to a table in LanceDB and turn it into a full full text search index (TF-IDF)
- We're using rerankers from Cohere 
- Given a query, we filter using metadata (category=film), get the top 10 results, and then rerank using our Cohere re-rankers.

At the end
- It's definitely worth learning about Sparse (like [[SPLADE]], great for in-domain) and multi-vector methods (like [[ColBERT]], very good out-of-domain) if you're interested.

"You want your bi-encoder to be a little 'looser', which high recall, and then trust your reranker to select the relevant documents from the top k."

"I don't spend as much time thinking about chunking as I do. I'm hopeful about LM-driven chunking, but it doesn't work very well so far."

"I think with ColBERT you can get away without finetuning for a bit longer, but if you company says you have $500, spend $480 on OpenAI to generate synthetic questions to finetune your {encoders?} to get better results. Always finetune if you can."

What are some libraries that you recommend for finetuning embeddings?
"I'd recommend [[Sentence Transformers]]; no need to reimplement the wheel, they've got the basics implemented really well, especially with their recent 3.0 release."

Can you give an pointers to the flow of finetuning embedding models?
- You need queries and documents and you need to tell your model "for X query, Y document is relevant, and Z is not relevant" (triplet loss). I'm not going to go too much into this rabbit hole, but when you have triplets, you also want to use hard negatives, which is where you use retrieval to generate hard negatives that look almost like positives but aren't; this is useful to teach your model with. If you don't have user queries from production, write your own queries.

Please share your thoughts on Graph RAG?
- I see it mentioned all the time, but it's not something that's come up for me at all. I think it's cool, but that's pretty much the full extent of my knowledge.

When you have long context windows, does that allow you to do anything different with RAG than you were able to before? Longer documents? etc?
- Yeah two main things: I can use longer documents... 
- There are a lot of diminishing returns in IR where getting Recall@10, Recall@100 is very very easy, Recall@5 is hard, and Recall@3 or Recall@1 are very hard.

Is there an open-source project to look at to help us evaluate a difference in result quality from different IR strategies?
- It's unfortunate that there's probably not one that systematically goes through every step.
- Like many things in retrieval, many things are conventional wisdom... unless you dig deep into the papers, it's quite rare to find good resources ((?)).

Tutorial on Fine-Tuning Encoders?
- I recommend the Sentence Transformers documentation!

Do you have go-to embedding models?
- I think my go-to- are the Cohere ones; it's nice to work with APIs, it works really well, and it's cheap -- I'd use ColBERT if I'm using something in my own pipeline... but it depends on your use case.
- I do have strong opinions on... if you go to [[MTEB]], you'll see a lot of LLMs as encoders, and I'd advise against that, because the latency for these are very high, and many of these don't generalize well.
	- Stick to the small ones between 100M-1B parameters... Try {} {}, T5.

Anyone have experience building RAG with just keyword BM25 as retriever? Our work uses Elasticsearch... is there a way to keep using Elasticsearch with RAG, or do you mainly use vector databases?
- Yeah, I've used Elasticsearch a bit -- it's perfectly possible, but you lose the semantic search aspect, though I think Elasticsearch has a vectorDB offering now too...
- You can alway use BM25 and then plug in a reranker.

Is it worth incorporating the BM25 scores during the reranking step too?
- Probably not, no.

Have you used approaches where you try (in some loss function somewhere) encourage diversity in terms of pulling passages from *many documents*, instead of just a few?
- I don't think I can give you a clear answer, I don't know much about that.

Any thoughts on Hierarchical RAG?
- I have not used it, and I don't think that it's needed for traditional pipelines -- there are a lot of other things you can improve.

I'm eager to learn with answer.ai will come up with any books on LM applications in the future?
- I don't think so, but never say never.









