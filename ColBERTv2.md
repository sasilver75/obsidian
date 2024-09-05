December 2, 2021 (20 months after [[ColBERT]])
Keshav Santhanam, [[Omar Khattab]], [[Christopher Potts|Chris Potts]], [[Matei Zaharia]]
Paper: [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488)
#zotero 
Takeaway: An optimization of [[ColBERT]]; ColBERT's decomposition of relevance modeling into token-level computations inflates the space footprint of the models, and this paper introduces improvements to improve the quality and space footprint of late interaction by 6-10x. ColBERTv2 introduces techniques to improve the quality and space-efficiency of multi-vector representations using a residual representation of vectors and distillation from cross-encoder systems.

----

Notes:
- Many neural IR vectors follow a *single-vector similarity paradigm* -- a pretrained LM is used to encode each query and each document into high-dimensional vectors, and relevance is modeled as a simple dot product between both vectors.
- An alternative is ==late interaction==, introduced by ColBERT, where queries and documents are encoded at a finer-granularity into multi-vector representations, and relevance is estimated using  rich yet scalable interactions between these two sets of vectors.
	- Specifically, ColBERT produces an embedding for every token in the query and document and computes similarity as the sum of maximum similarities (MaxSim) between each query vector and all vectors in the document.
	- Whereas single-vector models must capture complex query-document relationships with one single dot product, late interaction encodes meaning at the level of tokens, and delegates query-document matching to the interaction mechanism. 
	- However, this added expressivity comes at a cost -- existing late interaction systems impose an ==OOM larger *space footprint*== than single-vector models, since we need to store billions of small vectors for web-scale collections!
- ColBERTv2: A new late-interaction retriever that employs a simple combination of *==distillation from a cross encoder==* and *==hard-negative mining==* to boost quality beyond any existing method, and then uses a *==residual compression==* mechanism to reduce the space footprint of the late interaction by 6-10x while preserving quality.
	- As a result, ColBERTv2 establishes SoTA retrieval quality both *within* and *outside* its training domain.
- Authors introduce a new benchmark, dubbed [[LoTTE]] (Long-Tail Topic-stratified Evaluation) for IR that features 12 domain-specific search tests.
	- Focuses on relatively long-tail topics in its passages, unlike the Open-QA tests and many of the [[BEIR]] tasks. 
	- It evaluates models on their ability to answer natural search queries with a "practical intent," unlike many of BEIR's semantic-similarity tasks.
	- LoTTE is a new resource for out-of-domain evaluation of retrievers. It focuses on natural information-seeking queries over long-tail topics, an important yet understudied application space.
- Related Work: Token-Decomposed Scoring in Neural IR
	- Mentions [[SPLADEv2]] as the central point of comparison in the experiments that we report in in this paper. The SPLADE family produces a sparse vocabulary-level vector that retains the term-level decomposition of late interaction while simplifying the storage into one dimension per token. The SPLADE family piggybacks on the language modeling capacity acquired by BERT during pretraining, and is highly effective.
- Related Work: Vector Compression for Neural IR
	- In this work, we focus on late-interaction retrieval and investigate compression using residual compression approach that can be applied off-the-shelf to late interaction models, without special training. To the best of our knowledge, ColBERTv2 is the first approach to use residual compression for scalable neural IR.
- Related Work: Improving the quality of *single-vector* representations
	- Instead of compressing multi-vector representations as we do, much recent work has been on improving the quality of *single vector* models -- this can be decomposed into three directions:
		1. Distillation of more expressive architectures, including explicit denoising
		2. Hard negative sampling
		3. Improved pretraining
	- We adopt similar techniques to (1) and (2) above for ColBERTv2's multi-vector representations.
- Related Work: Out-of-Domain Evaluation in IR
	- In the [[MS MARCO]] and [[Natural Questions]], queries tend to reflect high-popularity topics like movies and athletes. But in practice, user-facing IR and QA applications often pertain to domain-specific corpora for which little-to-no training data is available, and whose topics are under-represented in large public collections.
	- The out-of-domain regime achieved recent attention with the [[BEIR]] benchmark, which combines several existing datasets into a heterogenous suite for "zero-shot IR" tasks, spanning bio-medical, financial, and scientific domains.
	- We introduce [[LoTTE]], a new dataset for out-of-domain retrieval, exhibiting natural search queries over long-tail topics.
- ==[[ColBERTv2]]==
	- Uses the late interaction architecture of ColBERT, where queries and passages are independently encoded with BERT, and the output embeddings encoding each token are projected to a lower dimension.
		- During offline indexing, every passage in the corpus is encoded into a set of vectors, and these vectors are stored; at search time, only the query $q$ is encoded into a multi-vector representation, and its similarity to a passage $d$ is computed as the summation of query-side "MaxSim" operations -- namely, the largest cosine similarity between each query token and all passage token embeddings. ![[Pasted image 20240518183405.png|150]]
		- The intuition is to align each query token with the most contextually relevant passage token, quantify these matches, and combine the partial scores across the query.
	- Supervision
		- Training a neural retriever typically requires both *positive* and *negative* passages for each query in the training set. ColBERT was trained using the official $<q, d^+, d^->$ triples of MS MARCO.
		- There are several weaknesses in this standard supervision approach -- our goal is to adopt a single, uniform supervision scheme that selects challenging negatives and avoids rewarding false positives or penalizing false negatives.
		- For each training query, we retrieve the top-k passages and feed each into a cross-encoder reranker. We then collect w-way tuples consisting of a query, a highly-ranked passage, and one or more lower-ranked passages. We use w=64 passages per example. We use a KL Divergence loss to distill the cross-encoder's scores into the ColBERT architecture. This Denoised training with hard negatives has been positioned in recent work as a way of bridging the gap between single-vector and interaction-based models, including late interaction architectures like ColBERT.
	- Representation
		- Evidence suggests that vectors in ColBERT corresponding to each sense of a word cluster closely, with only minor variation due to context. We exploit this regularity with a *residual* representation that dramatically reduces the space footprint of late interaction models, completely off the shelf without architectural or training changes.
		- Given a set of centroids C, ColBERTv2 ==encodes each vector *v* as the index of its closest centroid C_t *and* and a *quantized* vector r~ that approximates the residual r = v - C_t==. At search time, we use the centroid index t and the residual vector r~ to recover an approximate v~ = C_t  r~.
			- We quantize every dimension of r into one or two bits; in principle, our b-bit encoding of n-dimensional vectors needs log|c|+bn bits per vector; in practice, with n=128, we use four bytes to capture up to 2^32 centroids and 16 or 32 bytes (for b=1 or b=2) to encode the residual... ==This total of 20 or 36 bytes per vector contrasts with ColBERT's use of 256-byte vector encodings at 16-bit precision.==
	- Indexing
		- Given a large corpus of passages, the indexing stage precomputes all passage embeddings and organizes their representations to support fast nearest-neighbor search. There are three stages:
		1. **Centroid Selection**: In the first stage, ColBERTv2 selects a set of cluster centroids $C$. These are embeddings that ColBERTv2 uses to support ==residual encoding==, and for nearest neighbor search. Setting |C| proportionally to the square root of n_embeddings in the corpus works well, empirically.
		2. **Passage Encoding**: Having selected the centroids, we encode every passage in the corpus. This entails invoking the BERT encoder and compressing the output embeddings, assigning each embedding to the nearest centroid and computing a quantized residual. Once a chunk of passages is encoded, the *compressed* representations are saved to disk.
		3. **Index Inversion**: To support fast nearest-neighbor search, we group the embedding IDs that correspond to each centroid together, and save this *inverted list* to disk. At search time, this allows us to quickly find token-level embeddings similar to those in the query.
	- Retrieval
		- Given a query representation Q, retrieval starts with candidate generation. For every vector Q_i in the query, the nearest n_probe >= 1 centroids are found.
		- Using the inverted list, ColBERTv2 identifies the passage embeddings close to these centroids, decompresses them, and computes their cosine similarity with every query vector.
		- The scores are then grouped by passageID for each query vector, and scores corresponding to the same passage are max-reduced; this allows ColBERTv2 to conduct an approximate MaxSim operation per query vector, computing a lower-bound on the true MaxSim using the embedding identified via the inverted list.
		- These lower bounds are summed across query tokens, and the top-score n_candidate candidate passages based on these approximate scores are selected for ranking, which loads the complete set of embeddings of each passage, and conducts the same scoring function using all embeddings per document using the usual MaxSim operation.
- [[LoTTE]]: Long-Tail, Cross-Domain Retrieval Evaluation
	- LoTTE is a new dataset for "**Long-Tail Topic-stratified Evaluation**" for IR.
	- To complement the out-of-domain tests of [[BEIR]], LoTTE focuses on *natural user queries* that pertain to *long-tail topics*, ones that might not be covered by an entity-centric knowledge base like Wikipedia.
	- LoTTE has 12 test sets (divided by topic), each with 500-20000 queries and 100k-2M passages. Each test set is accompanied by a validation set of *related but disjoint queries and passages* (by making the passage texts disjoint, we encourage more realistic out-of-domain transfer tests).
	- 

Abstract
> Neural information retrieval (IR) has greatly advanced search and other knowledge-intensive language tasks. While many neural IR methods encode queries and documents into single-vector representations, ==late interaction== models produce ==multi-vector representations at the granularity of each token== and ==decompose relevance modeling into scalable token-level computations==. This decomposition has been shown to make late interaction more effective, but it inflates the space footprint of these models by an order of magnitude. In this work, we introduce ==ColBERTv2==, a retriever that ==couples an aggressive residual compression mechanism with a denoised supervision strategy to simultaneously improve the quality and space footprint of late interaction==. We evaluate ColBERTv2 across a wide range of benchmarks, establishing state-of-the-art quality within and outside the training domain while reducing the space footprint of late interaction models by 6--10x


> BEIR is practically useless nowadays; it was useful as a zero-shot test set. Among models trained openly on MS MARCO _without_ BEIR contamination or feedback, **ColBERTv2** is ahead of everything else. Nowadays, all the top models are build FOR BEIR/MTEB evals, and use (at best) the training splits of BEIR tasks (and more often than not, direct/indirect validation on BEIR test sets). I have a big text file with 50 papers that shows the ColBERT paradigm being 50-100x more data efficient and up to 15-30 points better in quality than single vector. Note: "Everything else" here refers to dense retrievers of the same size and budget/tricks; you CAN build expensive cross-encoders that do better than vanilla BERT-base ColBERTv2. SPLADE is a great competitor at mid-scale, but dense bi-encoders are generally not.
> - Omar Khattab, Jan 28 2024 (ColBERT): https://x.com/lateinteraction/status/1751661624539357550

# Paper Figures
![[Pasted image 20240518010344.png]]

![[Pasted image 20240519000618.png]]
Above: Breakdown of LoTTE
- Search Queries: Collected from GooAQ, a recent dataset of Google search-autocomplete queries and their answer boxes.
- Forum Queries: We collect forum queries by extracting post titles from StackExchange communities to use as queries, and collect their corresponding answer posts as targets.

![[Pasted image 20240519001403.png]]
Above: Examples of LoTTE queries
