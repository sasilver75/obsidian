April 27, 2020 -- [[Omar Khattab]] and [[Matei Zaharia]]
Paper: [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
See also: [[ColBERTv2]], [[ColBERT-QA]]
#zotero 
Takeaway: Introduced the ==late interaction== paradigm for efficient neural ranking in the form of ColBERT. Involves representing each token in the query and document as its own (small) vector using BERT, and then we compute similarity scores between these bags of vectors. Has the precomputability benefits of representation-based similarity (eg Bi-Encoders), but keeps the fine-grained matching of interaction-based methods.

Significance: It's said that ColBERT generalizes better out-of-domain than dense single-vector alternatives, like Bi-Encoders. This is probably because of the more granular token-level representation of data, as opposed to a document-level representation. It performs similarly to BERT-based models, but executes two OOMs faster, requiring 4 OOMs fewer FLOPs per query.

See some of the bonus pieces of information below the abstract from interviews with Omar.

References:
- [Video: Zeta Alpha's "ColBERT and ColBERTv2: Late interaction at a reasonable inference cost"](https://youtu.be/1hDK7gZbJqQ?si=iWtkZxCugM9WA05U)
- [Video: Vertex Venture's Neural Notes: ColBERT and ColBERTv2 with Omar Khattab](https://youtu.be/8e3x5D_F-7c?si=l-JaHCC63j4vgusp)

----

Notes:
- In IR, recent approaches involved fine-tuning deep pre-trained language models like [[ELMo]] and [[BERT|BERT]] for estimating relevant relevance -- these LMs help bridge the pervasive vocabulary mismatch between document and queries by computing and comparing deeply-contextualized semantic representations of each!
	- But these came at a steep computational cost of ~100-1000x more expensive than prior models.
- ColBERT propose a novel ==late interaction== paradigm for estimating relevance between a query $q$ and document $d$. 
	- The $q$ and $d$ are separately encoded into two sets of contextual embeddings, and relevance is evaluated using cheap and *==pruning-friendly==* computations between both sets -- meaning fast computations that enable ranking without exhaustively evaluating every possible candidate.
- (Lots of really good notes down below Figure 2 in the Paper Figures section, comparing and contrasting ==Representation-based Similarity==, ==Query-Document Interaction==, ==All-to-all Interaction==, and ==Late Interaction==)
- ColBERT can serve queries in tens or few hundreds of milliseconds, delivering over 170x speedup relative to existing BERT-based models, and being more effective performance-wise than every non-BERT baseline.
- Paper contributions
	- Proposal of ==late interaction== as a paradigm for efficient neural ranking
	- Presentation of ==ColBERT==, an effective model that employs novel BERT-based query and document encoders within the late interaction paradigm.
	- We show how to leverage ColBERT both for re-ranking on top of a term-based retrieval model (eg BM25) *and* for searching a full collection using vector similarity indices.
- Related Work
	- Neural Matching Models
		- IR researchers have introduced numerous neural architectures for ranking, including KNRM, ConvKNRM, SNRM.
	- Language Model Pretraining for IR
		- Recent work has emphasized the importance of pre-training language representation models in a unsupervised/self-supervised fashion before subsequently fine-tuning them on downstream tasks -- Notably BERT, but also ELMo.
		- The common approach is to feed the query-document pair through BERT together, and use an MLP on top of bERT's \[CLS\] output token to produce a relevance score.
	- BERT Optimization
		- LM-based rankers can be highly expensive in practice! Ongoing efforts to distill, compress, and prune BERT are beginning to narrow this gap.
	- Efficient NLU-based Models
		- Attempting to push expensive NLU computation *offline*. doc2query and DeepCT are the references used. ColBERT is also in this category!
- ==[[ColBERT]]==
	- Prescribes a simple framework for balancing the *quality* and *cost* of Neural IR. We delay the query-document interaction to facilitate cheap neural re-ranking through pre-computation, and even support practical end-to-end neural retrieval (through pruning via vector-similarity search).
	- Even though ColBERT's late-interaction framework can be applied to a wide variety of architectures, we choose to focus on using BERT to create query/document term encodings because of its SoTA performance.
	- Architecture
		- Query Encoder
		- Document Encoder
		- Late interaction mechanism
	- Process
		- Given a query $q$ and document $d$, our query encoder encodes $q$ into a bag of fixed-size contextual embedding vectors, and our document encoder encodes $d$ into another bag of fixed-size contextual embeddings.
		- Using our two bags of vectors, we compute relevance scores between $q$ and $d$ via late interaction, which we define as a summation of maximum similarity (==MaxSim==)  operators. In particular, we find the maximum [[Cosine Similarity]] of each query term vector with vectors in the bag of document term vectors, and combine the outputs via summation.
			- More sophisticated matching is possible with other choices like deep convolution and attention layers, but our summation of MaxSim computations has two nice characteristics: It's *cheap*, and it's amenable *to highly-efficient pruning* for top-k retrieval.
	- Query Encoder
		- (We share a *single* BERT model among our query and document encoders, but distinguish input sequences that correspond to queries and documents by prepending a special token \[Q\] to queries and \[D\] to documents)
		- Given a textual query $q$, we tokenize it into its BERT-based WordPiece tokens q1,q2...qn. We prepend the token \[Q\] to the query, placing it right after BERT's sequence-start token \[CLS\]. If the query has fewer than a pre-defined number of tokens, we pad it with BERT's special \[mask\] tokens.
		- BERT then computes a contextualized representation of each token. Our encoder passes the contextualized output representations through a linear layer with no activations, which serves to control the dimension of ColBERT's embeddings, producing m-dimensional embeddings for the layer's output size m. The dimensionality of these resulting token vectors is much smaller than BERT's fixed hidden dimension.
			- This reduced dimensionality has only limited impact on the efficiency of query encoding -- it's actually for controlling the *space footprint* of documents -- it has a significant impact on query execution time, particularly the time taken for transferring the document representations onto the GPU from system memory (which can be the most expensive step in re-ranking with ColBERT!)
	- Document Encoder
		- (We share a *single* BERT model among our query and document encoders, but distinguish input sequences that correspond to queries and documents by prepending a special token \[Q\] to queries and \[D\] to documents)
		- The document encoder has a very similar process; we segment a document d into its constituent tokens, to which we prepend BERT's start token \[CLS\] followed by \[D\], indicating *document*. 
		- Unlike queries, we do not apply \[mask\] tokens to documents.
		- After feeding this sequence through BERT and the subsequent linear layer, the document encoder filters out the embeddings corresponding to punctuation symbols, determined by a pre-defined list. This is meant to reduce the number of embeddings per document, as we hypothesize that (even contextualized) embeddings of punctuations aren't necessary for effectiveness.
	- Late Interaction
		- Given our representations of query $q$ and document $d$, we compute a relevance score via late interaction between their bags of contextualized embeddings. This is conducted as a sum of maximum similarity computations, namely cosine similarity (implemented as dot-products due to the embedding normalization).
		- ColBERT is differentiable end-to-end; we fine-tune the BERT encoders and train from scratch the additional parameters (the linear layer and the Q/D markers' embeddings) using the Adam optimizer.
		- Given a triple $<q,d^+,d^->$ with query *q*, positive document $d^+$, and negative document $d^-$, ColBERT is used to produce a score for each document individually and is optimized via pairwise softmax cross-entropy loss over the computed scores of d+ and d-.
	- Offline Indexing: Computing and Storing Document Embeddings
		- By design, ColBERT isolates almost all of the computations between queries and documents, largely to enable pre-computation of document representations offline.
		- Indexing process: We proceed over documents in the collection in batches, running our document encoder on each batch and storing the output embeddings per document.
		- Although indexing a set of documents is an offline process, we still implement a few optimizations:
			- We exploit multiple GPUs for faster encoding of batches of documents in parallel.
			- When batching, we pad all documents to the maximum length of a document *within* the batch. To make this more effective, our indexer proceeds through groups of B documents (eg 10,000), sorts these documents by length, and feeds batches of b (eg 128) documents of comparable length through our encoder. This length based bucketing in sometimes referred to as `BucketIterator` in some libraries.
			- We found that a non-trivial portion of indexing time is spent on pre-processing the text sequences, primarily BERT's WordPiece tokenization; we parallelize these across the available CPU cores.
		- We save our document representations to disk using 32bit or 16bit values.
	- Top-K *Re-Ranking* with ColBERT
		- Recall that ColBERT can be used for *reranking* the output of *another* retrieval model (typically a fast term-based model) -- or it can directly be used for end-to-end retrieval from a document collection (as we'll discuss in the next section)
		- Given a query $q$, we compute its bag of contextualized embeddings, and concurrently gather the document representations into a 3-dimensional tensor $D$ consisting of $k$ (eg 1,000) document matrices. We pad the k documents to their maximum length to facilitate batched operations. On our GPU, we compute a batch dot-product of our query embedding and our $D$ tensor.
		- The output materializes a 3-dimensional tensor that is a collection of cross-match matrices between $q$ and each document.
		- To compute the score of each document, we reduce its matrix across document terms via  max-pool and reduce across query terms via a summation.
		- Finally, we sort the k documents by their total scores.
		- Relative to existing neural rankers, this computation is very cheap that (in fact) its cost is dominated by the cost of gathering and transferring the pre-computed embeddings
	- End-to-End Top-k Retrieval with ColBERT
		- As mentioned before, ColBERT's late-interaction operator is specifically designed to enable end-to-end retrieval from a large collection (in addition to the reranking from the output of another retrieval model, as mentioned in the previous section).
		- In this section, the number of documents to be ranked is too large for exhaustive evaluation of *each* possible document, especially when N=10,000,000 and k << N.
		- To do so, we leverage the ***pruning***-friendly nature of the MaxSim operations at the backbone of late interaction!
			- Instead of applying MaxSim between one of the query embeddings and all of one document's embeddings, we can use fast vector-similarity data structures to efficiently conduct this search between the query embedding and *all* document embeddings across the full collection. We use [[FAISS]], an off-the-shelf library, to do this. (Note that at the end of offline indexing, we maintain a mapping from each embedding to its document of origin, then index all document embeddings into `faiss`).
		- When serving queries, we use a two-stage procedure to retrieve the top-k documents from the entire collection:
			1. An approximate stage aimed at filtering: We concurrently issue $N_q$ vector-similarity queries (corresponding to each of the embeddings in our query bag) onto our `faiss` index. This retrieves the top k' (eg k' = k/2) matches for that vector over all document embeddings. We map each of those to its document of origin, producing $N_q \times k'$ document IDs, not all of which are necessarily unique.
			2. A refinement stage: Now, we refine this set by exhaustively re-ranking *only* those K documents in the usual manner we described in the previous section.
- Conclusion
	- We introduced ColBERT, a novel ranking model that employs contextualized late interaction over deep LMs for efficient retrieval.
	- By independently encoding queries and documents into fine-grained representations that can interact via cheap and pruning-friendly computations, we can leverage the expressiveness of deep LMs which greatly speeding up query processing -- this also allows us to use ColBERT for end-to-end neural retrieval directly from a large document collection!
	- ColBERT is >170x faster and requires 14,000x fewer FLOPs/query than existing BERT-based models, all while only minimally impacting quality and while outperforming every non-BERT baseline!



Abstract
> Recent progress in Natural Language Understanding (NLU) is driving fast-paced advances in Information Retrieval (IR), largely owed to fine-tuning deep language models (LMs) for document ranking. While remarkably effective, the ==ranking models based on these LMs increase computational cost by orders of magnitude over prior approaches==, particularly as they must feed each query-document pair through a massive neural network to compute a single relevance score. To tackle this, we present ==ColBERT==, a ==novel ranking model that adapts deep LMs (in particular, BERT) for efficient retrieval==. ColBERT introduces a ==late interaction== architecture that ==independently encodes the query and the document using BERT and then employs a cheap yet powerful interaction step that models their fine-grained similarity==. By delaying and yet retaining this fine-granular interaction, ColBERT can ==leverage the expressiveness of deep LMs while simultaneously gaining the ability to pre-compute document representations offline==, considerably speeding up query processing. Beyond reducing the cost of re-ranking the documents retrieved by a traditional model, ColBERT's pruning-friendly interaction mechanism enables leveraging vector-similarity indexes for end-to-end retrieval directly from a large document collection. We extensively evaluate ColBERT using two recent passage search datasets. Results show that ColBERT's effectiveness is competitive with existing BERT-based models (and outperforms every non-BERT baseline), while executing two orders-of-magnitude faster and requiring four orders-of-magnitude fewer FLOPs per query.

Bonus Content
> Q: Why is ColBERT superior to traditional embedding models?
> A: The idea that you can accurately boil down the nuances of ~256 tokens (2/3s of a page, e.g.) into a single vector is a pretty wild proposition. No matter how good the model, semantic nuance and details will inevitably be lost. Instead, ColBERT's approach is to allocate a small, efficient representation to ==EACH TOKEN== within the passage; this way, you're not crossing your fingers that your compression strategy isn't crushing a lot of semantic value. 
> ==Key==: At a high level, you embed the query and the passage, getting vector representations for every token in both. Then, for each query token, you find the token in the passage with the largest dot product similarity, which is called the "MaxSim" for each token. Finally, the similarity scores between the query and passage is just the summation of all the MaxSims you found. So while you might have to compute many such dot products, each vector is much smaller than usual (eg dimensionality of 4), so it scales much better than you'd think. The "RAGatouille" library is much easier to grok than the original library. ((Sam question: Should we really be computing vectors and maxSims for "stopwords" in the query and passage? Could we have compute by not doing that?))
> Source: https://x.com/marktenenholtz/status/1751406680535883869

> The ==MaxSim== operator's introduction in the original ColBERT paper is perhaps the deepest insight in the paper; inspired by the things that work well in traditional IR, where, event though you're working with a bag of words representation, you don't want to score every document that has one or more of the terms. Instead, you do some pruning. In Search, if you can prove that some document's cant possibly have a high enough score, you save a lot of work. For our ColBERT scoring function, we have two bags of vectors, one from each the query and document. A document is relevant to a query (a bunch of words), IFF, for most terms in the query, there's a contextual match on the document end. So for each term in the query, we find the closest vector on the document side, repeat for each word in the query, and sum up these partial scores to get an average of how well is the query contextually captured in the document. ((Wait, I don't see how we're avoiding checking documents/passages, though))
> You can think of ColBERT as two things: You can use it out of the box as search, or you can think of it as the key idea of doing late-interaction between fine-grained representations. Along that spectrum, there are a bunch of modular components: The encoder (a BERT model or other language encoder that takes in text and spits out a bag of vectors), the search stack (Here, you could be very modular, but the more modular you are, you might be leaving some e2e optimizations. For a long time, ColBERT was a modular wrapper around FAISS).
> - Omar Khattab (https://youtu.be/8e3x5D_F-7c?si=eMh4Z0FdEhUwZJy_)

# Paper Figures
![[Pasted image 20240517180611.png]]
Above: Showing similar performance (as measured by [[Mean Reciprocal Rank|MRR]]) to [[BERT|BERT]]-large, but with 2 OOM less query latency.

![[Pasted image 20240517181612.png]]
Above: Comparison between various paradigms in Neural IR
- a: ==Representation-based Similarity==: These independently compute an embedding for $q$ and an embedding for $d$, then estimate relevance a single similarity score between two vectors. Makes it possible to precompute document representations ahead of time, greatly reducing the computational load per query.
- b: ==Query-Document Interaction==: Interaction-focused rankers model word- and phrase-level relationships across $q$ and $d$, and match them using a deep neural network (eg with CNNs/MLPs). In the simplest case, they feed the NN an *interaction matrix* that reflects the similarity between every pair of words across $q$ and $d$. Cannot precompute document representations ahead of time, but good performance.
- c: ==All-to-all Interaction==: Models the interactions between words *within* as well *across* $q$ and $d$, like how we use [[BERT|BERT]]'s transformer architecture to do query-document similarity. Cannot precompute document representations ahead of time, but good performance.
- d: ==Late Interaction== (eg the proposed ColBERT): While b and c above have superior performance, they can't precompute document representations like a can. Imagine having 1,000,000 documents in the database and having to do 1,000,000 BERT forward passes -- it's unworkable! Late Interaction enables both the fine-grained matching of interaction-based models and the precomputability of document representations from representation-based models by retaining, yet judiciously *delaying* the query-document interaction! Every query embedding interacts with all document embeddings via a MaxSim operator, which compute compute maximum similarity, and the scalar outputs of these operators are summed across query terms. This allows ColBERT to exploit deep LM-based representations while shifting the cost of encoding documents offline and amortizing the cost of encoding the query *once* across all ranked documents. Also enables ColBERT to leverage vector similarity search indices to retrieve the top-k results directly from a large document collection, substantially improving *recall* over models that only re-rank the output of term-based retrieval.

![[Pasted image 20240517213218.png]]
Above: See that we use BERT because of its SoTA effectiveness (though ColBERT's late interaction framework could be applied to a wide variety of architectures).
# Additional Figures
![[Pasted image 20240414150952.png]]

