https://blog.reachsumit.com/posts/2023/09/generative-retrieval/
----

The field of IR has seem a surge of interest in utilizing generative retrieval techniques to build search systems.
- These systems use autoregressive models to map each query to the relevant document, *eliminating the need for traditional search indexes*.

The resulting systems have been called ==Autoregressive Search Engines== and ==Model-based Retrieval Frameworks==. 
- We'll do a gentle introduction and literature review!

# Toward Index-free and Model-based Generative Retrieval

### Text Retrieval
- Text Retrieval is a fundamental task in IR where, given a query, finds relevant textual information from a  massive document corpus. Usually entails retrieving a single, most-relevant paragraph or document to the given query (eg no-hop retrieval) or multiple documents that together answer the query (multi-hop retrieval).

### Traditional IR Framework
- Most of the traditional IR systems follow a three-step paradigm of *index-retrieve-then-rank*. 
	- Index: Document representations are learned
	- Retrieval: Relevant documents to given queries are found 
	- Rank: Documents are sorted in the order of relevance
- For the Indexing and retrieval stage, the existing approach can be divided into the following categories with respect to the type of representation and mode of indexing.
![[Pasted image 20240528000700.png]]
### Sparse Retrieval with Inverted Indices
- This traditional retrieval and indexing method provides a great trade-off between accuracy and efficiency.
- First, an inverted index is built based on all documents in the corpus. This index encodes term-based features like term frequencies, positions, document structure information, and document length.
	- For retrieval, this method fetches documents based on matching between query terms and document terms.
- [[TF-IDF]], [[BM25]], and some graph-based approaches that employ term frequencies and precise word matching.

### Dense Retrieval with Vectorized Indexes
- Each of the documents in the corpus are encoded into a dense vector through a neural network. These low-dimensional dense representations are used to build a vectorized index.
- During retrieval, we embed the input query into the same latent space, and calculate embedding similarity between the query and documents through similarity functions like inner product, cosine similarity, etc.
- Dense retrieval models have been further enhanced with optimizations like
	- Negative sampling strategies, hard negative mining
	- Lightweight interaction layers
	- Knowledge distillation (often from Cross-Encoders)
	- Pre-training on pseudo-query and document pairs
	- Fine-tuning with large-scale pre-trained language models
- The most common dense retrieval methods use [[Bi-Encoder]] which are often based on a pretrained LM such as [[BERT|BERT]]. 
	- These bi-encoders separately encode documents into low-dimensional dense representations.

### Ranking
- The goal of the ranking model is to determine the relevance degree of each candidate document, given a query.
- There's been a wide variety of ranking algorithms proposed over the years (vector space models, probabilistic models, learning-to-rank models, and neural ranking models).

### Shortcomings of Traditional Paradigms
- Sparse retrieval techniques are known for their simplicity and effectiveness, but often fail to capture rich semantic connections between queries and documents.
	- BM25 relies on lexical overlap, term frequency heuristics, and inverse document frequencies, thereby failing to match documents that have minor world overlap, but are semantically related (synonyms, acronyms).
- Bi-Encoder models still suffer from issues like the embedding space bottleneck, information loss, and limited expressiveness due to fixed-size embeddings, as well as lack of fine-grained interactions between embeddings.
- Problems with the traditional "index-retrieve-then-rank" framework:
	1. During training, the heterogenous components in traditional pipeline are difficult to jointly optimize in an end-to-end way towards a global objective. Errors accumulate among the components.
	2. At the inference stage, it requires a large document index to search over the corpus, leading to significant memory consumption that increases linearly with corpus size.
	3. In a dual-encoder setup, the representation for queries and documents are obtained independently, allowing for only shallow interactions between them.
	4. By limiting query and document representations to a single fixed-size dense vector, dual encoders also potentially miss fine-grained information when capturing the similarity between two vector representations. This is even more critical in the case of multi-hop retrieval!
	5. An appropriately hard set of negative data has to be subsampled during training.

## Generative  Retrieval
- Fundamentally different from the long-standing "index-retrieve-then-rank" paradigm.
	- This model-based IR system collapses the indexing, retrieval, and ranking components to a single consolidated model used to directly return relevant information corresponding to a query, without utilizing explicit document indexes.
	- Uses an [[Encoder-Decoder Architecture]] like [[T5]] or [[BART]], and generates the output token by token in an autoregressive fashion conditioned on context.

![[Pasted image 20240528002716.png|300]]
Above: ((I don't really... get it. Language models aren't databases, we can't reliably "store" grounded information in them.))

Advantages
- Simplifies the retrieval pipeline by replacing a large external index with an internal index (model parameters)
- Knowledge of all documents in the corpus is encoded ((fuzzily!)) into model parameters, and can be optimized during model training towards a global objective.
- Improves generalization ability by forcing the model to explain every token in the question and document using cross-attention.


## Inputs and Outputs
- During training, the model learns to map the textual representation of documents to output targets, such as document identifiers, or an entire text sequence, like a tile, while also learning to fetch the same targets when it retrieves a relevant query as input.


... ==I'm going to stop reading at this point. I really don't see how this is a good idea, and it doesn't seem obvious that this gets any SoTA results, so... I'll just leave this article until I "need" to read it==



