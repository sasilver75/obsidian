# Topic: Retrieval and RAG
https://www.youtube.com/watch?v=WQYi-1mvGDM&list=PL8PYTP1V4I8DZprnWryM4nR8IZl1ZXDjg&index=10

---

Given the prompt:
> I think Vin Diesel has been a voice actor for several characters in TV series, do you know what their names are?

There would be some problems with using a raw language model!
- It might hallucinate names
- It might not have the most recent characters, if the movies occurred after the training/knowledge cutoff of the model.
- If it *does* hallucinate, it's not easy for us to edit the model to be able to *not* hallucinate.

![[Pasted image 20240615205046.png]]
Above: (Black-box retrieval is just asking Google/Bing; this is a reasonable method if you want to search over internet data, or recent data. This is in fact what ChatGPT does -- searching over Bing)


## Sparse Retrieval
- Express the query as a sparse word frequency vector, usually normalized by length.
- Then we have a whole bunch of documents that we similarly encode using a sparse vector.
![[Pasted image 20240615205419.png|250]]
Above: Notice that even though the second document seems to us to be the one that has the "answer" to our query we actually retrieved the first document (irrelevant) because of the high similarity on the "is" dimension. Yikes!
- Common words like "like" and "is" can result in a high lexical similarity 🥲
	- 💡 We can fix this through something called ==Term Weighting==; in addition to having this vector that calculates the word frequency in a particular document, we also have an upweighting term that gives higher weight to low-frequency words (like 'NLP'), which tend to be more informative of what a document is about.

This introduces [[TF-IDF|Term Frequency Inverse Document Frequency]]
![[Pasted image 20240615205947.png]]
- If a term appears many times in a document, it will have a low IDF score. If a term rarely appears, it will have a high IDF score. You can basically think of it as "rarity" in your set of documents.
- There's a variant of TF-IDF called [[BM25]]
	- The details of this aren't important; basically, what this is doing is some smoothing on the term frequencies.

## Inverted Index
- Now that we have a term-based sparse vector, how do we use it to look up relevant documents in the collection very quickly?
	- Because we might have a collection that's extremely large (like, the size of the entire internet). We need a structure that allows for efficient lookup of our sparse vectors! This is an [[Inverted Index]]

![[Pasted image 20240615210356.png]]
On the right side, the values in the lists are the indices of documents that contain that term. This is called an Inverted Index.
- An example of software for this is ==Apache Lucene==.


## Dense Retrieval
- The way dense retrieval works is we encode the document and query into dense vectors, and find the nearest neighbor(s).
- To do these encodings, we can use 
	- out-of-the-box embeddings
	- learned embeddings, specifically for retrieval
		- If you're very serious about retrieval, it's a good idea to use these types of embeddings. If we think about the things that TF-IDF does, it gives high weight to  contentful words and rare words; we aren't guaranteed that any random embedding is going to do that.

## Learning Retrieval-oriented Embeddigns
- To learn these, we select positive and negative documents for a query, and train using a contrastive loss (eg Hinge Loss) and train our embeddings.
	- If you have gold-standard positive documents, then this is relatively easy to train, because you can get negative documents in a number of ways:
		- Popular: When we form a batch of data, you take the other documents in the batch that are positive for some *other* query, and use those as the negative documents for your current query.
	- Another common thing is to get ==Hard Negatives==, which is examples that *look positive* (relevant), but are actually negatives.
		- The [[Dense Passage Retrieval|DPR]] paper used [[BM25]] to mine these hard negatives.
- There are also methods to learn these retrievers based on non-supervised data!
	- If we're usually taking positive documents from human annotations of whether a document is correct or not... we might not always have that information available (eg from click logs)!
	- [[Contriever]] uses two random spans within a document as a positive pair, and two random spans from across documents as negative pairs. After you've done that large-scale initial pretraining in an unsupervised manner, you could then go in and fine-tune.

## Approximate Nearest Neighbor Search
- Given an embedding query and a large pool of embedded documents, how do we calculate our nearest neighbor documents?
	- Simply taking a product between the query and all documents in the document database is extremely costly.
	- There exist methods for [[Approximate Nearest Neighbor Search]], which allow us to retrieve embeddings with maximum inner product (MIPS) with sub-linear time.
![[Pasted image 20240615221641.png]]
There are two popular methods:
- [[Locality Sensitive Hashing]]: You make partitions in continuous space, and then use it like an [[Inverted Index]].
	- Let's say we have a bunch of embeddings (say they're shown in 2d space). We define a bunch of planes that separate these points into two spaces. 
	- A given point can then be described by whether it's on the right/wrong side of every plane, which creates a sparse vector.
	- We know that sparse vectors can be looked up in an inverted index table!
- Graph-based Search (eg [[Hierarchical Navigable Small Worlds|HNSW]]): We create "hubs", and search from there. These hubs are points in dense neighborhoods, like cluster centroids. This reduces the number of points we should be looking at; then we search through a specific neighborhood of points. You can do this in a hierarchical manner too!
Common software for these are things like [[FAISS]] or ChromaDB.


## Cross-Encoder Reranking
- There's still a problem once we've retrieved our relevant k vectors. One of the problems with forming a dense embedding is that you kind of need to know what you're looking for in advance, when you compress a long passage into a single embedding.
	- There's actually other information that could be relevant to the query that we have to throw out because of the limited embedding capacity. ((IMO this isn't really the major motivation for cross encoders/bi-encoder split (scalability), but it is an interesting idea))
![[Pasted image 20240615222126.png]]

There are some other approaches that are kind of in the middle of BiEncoders and CrossEncoders, with the most famous being [[ColBERT]]
![[Pasted image 20240615222435.png|450]]
- We use contextualized versions of tokens in the document and query, and do a rich interaction between these tokens at query time, but we're also still allowed to pre-create our document embeddings, which is a benefit over cross-encoders. The interaction between the query and document terms is lightweight, using a maximum similarity document
- The downside to this method might be obvious; Here, we have one vector for each token in the document, so our vector database gets N times larger; we can compress documents to smaller numbers of N, but...

![[Pasted image 20240615224303.png]]
[[HyDE]] (made at CMU!)
- Idea: It's easier to match a document and a document than it is to match a query and a document
	- But we're just given a query, so what do we do?
	- We take a LM, feed it in a query and a prompt that says "Generate a document that looks like it should be an answer to this query"
	- The LM generates a document, which hopefully looks similar to the documents that we want to retrieve -- ideally *more similar* than our query ((This isn't the reason why we use HyDE! We use it in situations where we don't have supervision to bootstrap our retriever with, so we basically rely on parametric knowledge of LMs!))

![[Pasted image 20240615225031.png]]
![[Pasted image 20240615231603.png]]
Above: [[Retrieval-Augmented Generation (Model)|RAG]]
P_n(z|x) is the probability of picking a document, given x
The next term is basically the probability of the next token, given that you have a document
This is basically linearly interpolating between the multiple documents.

The problem with training your embedding model is that it means that the documents that you've embedding now have stale embeddings, and it's very costly to recompute those embeddings.
- Solution: We only train the query embeddings, and keep the document embeddings fixed.
	- Alternative: [[REALM]] uses teh most recent document embedder to asynchronously update the search index during training, but this is quite an onerous process. It's more common to use a fixed document embedder and update the query embedder.


## When do we retrieve?
![[Pasted image 20240615231934.png]]

Triggering Retrieval with Token Embeddings was proposed by [[Toolformer]]; we generate tokens that trigger retrieval or other tools.
- This is trained with examples of tools being useful
![[Pasted image 20240615232048.png]]
This is how things are implemented in ChatGPT nowadays; not only for retrieval, but for other tool use, like generating code or images.

Another option is to trigger retrieval with uncertainty estimates.
In the FLARE paper (his student), they tried to generate content, and do retrieval if the language model certainty is low.
![[Pasted image 20240615232210.png]]
When we have low probability, we form a query where we *blank out* the low probability parts of this (?), and then do a search. This is a little bit like the HyDE method, where we create a document that's similar to the one we want to find. 
- Whenever we have a high confidence output, we don't do retrieval, but when we have low confidence outputs, we do the retrieval and base the output on this.
- The downside here is that we sometimes need to generate twice (generate the output once, find the low confidence parts, retrieve, generate again)

Token-by-token retrieval methods
- One of the methods that popularized this idea was something called kNN LM, which retrieved similar examples, and then used tokens from the examples.
- This is kind of like a very powerful count-based bigram model.
- ![[Pasted image 20240615232528.png]]
![[Pasted image 20240615232817.png]]
Glazing over


![[Pasted image 20240615233021.png]]
![[Pasted image 20240615233400.png]]

![[Pasted image 20240615233826.png]]
Speaker likes SCROLLS more than Long Range Areana

![[Pasted image 20240615233920.png]]
Even if we have long contexts, it seems that many models, even SoTA ones, pay less attentions to things in the middle of long context windows! "Lost in the Middle"


![[Pasted image 20240615234140.png]]
Filtering the context down to the most relevant content that we think is appropriate, before feeding it through the generator.




