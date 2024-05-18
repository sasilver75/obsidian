December 2, 2021 (20 months after [[ColBERT]])
Keshav Santhanam, [[Omar Khattab]], [[Christopher Potts|Chris Potts]], Matei Zaharia
Paper: [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488)
#zotero 
Takeaway: An optimization of [[ColBERT]]; The decomposition of relevance modeling into token-level computations inflates the space footprint of the models, and this paper introduces improvements to improve the quality and space footprint of late interaction by 6-10x.

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
	- 

Abstract
> Neural information retrieval (IR) has greatly advanced search and other knowledge-intensive language tasks. While many neural IR methods encode queries and documents into single-vector representations, ==late interaction== models produce ==multi-vector representations at the granularity of each token== and ==decompose relevance modeling into scalable token-level computations==. This decomposition has been shown to make late interaction more effective, but it inflates the space footprint of these models by an order of magnitude. In this work, we introduce ==ColBERTv2==, a retriever that ==couples an aggressive residual compression mechanism with a denoised supervision strategy to simultaneously improve the quality and space footprint of late interaction==. We evaluate ColBERTv2 across a wide range of benchmarks, establishing state-of-the-art quality within and outside the training domain while reducing the space footprint of late interaction models by 6--10x


> BEIR is practically useless nowadays; it was useful as a zero-shot test set. Among models trained openly on MS MARCO _without_ BEIR contamination or feedback, **ColBERTv2** is ahead of everything else. Nowadays, all the top models are build FOR BEIR/MTEB evals, and use (at best) the training splits of BEIR tasks (and more often than not, direct/indirect validation on BEIR test sets). I have a big text file with 50 papers that shows the ColBERT paradigm being 50-100x more data efficient and up to 15-30 points better in quality than single vector. Note: "Everything else" here refers to dense retrievers of the same size and budget/tricks; you CAN build expensive cross-encoders that do better than vanilla BERT-base ColBERTv2. SPLADE is a great competitor at mid-scale, but dense bi-encoders are generally not.
> - Omar Khattab, Jan 28 2024 (ColBERT): https://x.com/lateinteraction/status/1751661624539357550

# Paper Figures
![[Pasted image 20240518010344.png]]

