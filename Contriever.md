December 16, 2021
[[Meta AI Research]], PSL University, University of Grenoble, UCL
Paper: [Unsupervised Dense Information Recall with Contrastive Learning](https://arxiv.org/abs/2112.09118)
#zotero 
Takeaway: Uses unsupervised training as a method of bootstrapping large labeled datasets for training retrievers. Uses a contrastive learning objective (with positive and negative documents), building positive pairs through the Inverse Cloze Task and Independent Cropping, and negative pairs through in-batch negative sampling and MoCo.

---

Notes:
- Traditionally, retrieval systems leverage lexical similarities to match queries and documents, using (eg) [[TF-IDF]] or [[BM25]] weighting. These approaches are based on near-exact matches between tokens of the queries and documents, and don't generalize well (from queries to documents).
	- In contrast, neural retrievers allow learning beyond lexical similarities, enabling SoTA performance across many QA benchmarks. But training these models requires datasets that manually match queries to relevant documents in the collection, which is hardly possible when collections contain millions or billions of elements.
	- A potential solution is to train a dense retriever on some large retrieval dataset like [[MS MARCO]], and *hope* that you can zero-shot transfer to new domains - the truth is that neural retrieval models are still often outperformed by classical methods based on term frequency. These large datasets often aren't available in languages other than english.
- To improve the generalization ability of neural retrievers, a natural alternative is [[Unsupervised Learning]], where we train dense retrievers without supervision by using an auxiliary task that *approximates* retrieval.
	- ==Given a document, we can generate a *synthetic query*, and then train the network to retrieve the original document, among many others, given the query.==
	- The Inverse Cloze Task (ICT) to pre-train retrievers uses a given sentence as a query, and predicts the context surrounding it. Strongly related to contrastive learning.
- Related Work
	- Term-frequency based information retrieval
		- Historically, documents and queries are represented as sparse vectors, where each element of the vectors corresponds to a term of the vocabulary ([[One-Hot Encoding]]).
		- Different weighing schemes have been proposed, but one of the most popular is [[TF-IDF]], and is based on inverse document frequency, or term specificity. [[BM25]] extends TF-IDF.
		- A well-known limitation is that they rely on near-exact match to retrieve documents (which led to the introduction of dense vectors).
	- Neural network based information retrieval
		- A limitation of [[Bi-Encoder]] is that queries and documents are represented by a single vector, preventing the model from capturing *fine-grained* interactions between terms.
		- [[Cross-Encoder]] models, based on the BERT model, jointly encodes queries and documents. The application of a strong-pre-trained model plus the cross-encoder architecture led to important improvements on the MS MARCO benchmark. 
			- ((Note that Cross-Encoders have a downside, in that you can't precompute document representations. At query time (assuming you're using cross-encoders as the first stage of retrieval), you have to do a forward pass with (q,d) for every document in your collection, which isn't tenable. Cross-Encoders are much more reasonable to use in a re-ranking step on a curated subset of documents retrieved using (eg) BM25))
		- The ColBERT model keeps a vector representation for each term fo the query and documents. To make retrieval tractable, the term-level function is approximated to first retrieve an initial set of candidates, which are then reranked with the true score.
		- Knowledge distillation has been used to train retrievers, either using the attention scores of the reader on the downstream task as synthetic labels, or the relevance score from a cross encoder.
	- Self-supervised learning for NLP
		- Word2Vec's success in 2013 prompted many SSL techniques. 
		- 2018's BERT used earlier objective functions to learn sentence representations, including next sentence prediction and sentence order prediction.
		- Lee at al (2019) introduced the Inverse Cloze Task (ICT) whose purpose is to predict the context surrounding a span of text.
		- Guu et al (2020) integrated a bi-encoder retriever model in a BERT pre-training scheme; retrieved documents are used as additional context in the BERT task.
		- MoCo (2020) obtained positive pairs of sentences using back-translation.
		- SBERT (Reimers, 2019) uses a siamese network similar to contrastive learning to learn a BERT-like model that is adapted to matching sentence embeddings. Similar to our work, but theiers requires aligned pairs of sentences to form positive pairs, whereas we propose to use data augmentation to leverage large unaligned text corpora.
- Training Method for Contriever
	- The objective of a retriever is to find relevant documents in a large collection for a given query by outputting a query relevance score for each document.
		- Because Cross-Encoders don't scale well to large collections of documents, we use a Bi-Encoder architecture where we independently encode queries and documents; relevance scores are given by the dot product between their representations.
	- [[Contrastive Loss|Contrastive Learning]] is an approach that relies on the fact that every document is, in some way, unique. This signal is the only information available in the absence of manual supervision; contrastive loss is used to learn by discriminating between documents. This loss compares either positive (from the same document) or negative (from different documents) pairs of document representations. ![[Pasted image 20240527133148.png]]
	- Above: Given a query $q$ with an associated positive document $k_+$, and a pool of negative documents $k_{i...K}$ , we define contrastive InfoNCE loss, with $\tau$ as a temperature parameter. This loss encourages positive pairs to have high scores and negative pairs to have low scores.
	- Another interpretation: Given the query representation $q$, the goal is to recover (or retrieve) the representation $k_+$ , corresponding to the positive document, among the negative negatives $k_i$.
	- We consider ways to build [[Positive Pair]]s
		- ==Inverse Cloze Task== (ICT), a data augmentation generating two mutually-exclusive views of a document. 
			- The first view is obtained by randomly sampling a span of tokens from a segment of text, while the complement of the span forms the second view. ((So if we have a sequence `I love to pet dogs on a sunny day`, we might randomly sample `to pet dogs`, and have the complement `I dogs on a sunny day`)).
		- ==Independent Cropping== is a common independent data augmentation for images where views are generated independently by cropping the input. In the context of text, cropping is equivalent to sampling a span of tokens. This strategy thus samples independently two spans from a document to form a positive pair. This is different from ICT, because in cropping, both views correspond to contiguous subsequences of the original data -- Also, random cropping is symmetric, with both queries and documents following the same distribution. Can lead to overlap between two views of the data, encouraging t the network to learn exact matches between query and document.
		- We also consider additional augmentations like random word deletion, replacement, or masking.
	- We consider ways to build [[Negative Pair]]s
		- In-batch Negative Sampling
			- Generate the negatives by using the other examples from the same batch; each example in a batch is transformed twice to generate positive pairs, and we generate negatives by using the views from the other examples in the batch. Requires extremely large batch sizes to work well.
		- MoCo/"Negative pairs across batches"
			- An alternative approach is to store representations from previous batches in a queue, and use them as negative examples in the loss.
			- This lets us use slightly smaller batch sizes, but slightly changes the loss by making it asymmetric between "queries" (one of the views generated from the elements of the current batch), and "keys" (the elements stored in the queue).
			- In practice, features stored in the queue from previous batches come from previous iterations of the network; this leads to a drop in performance when the network rapidly changes during training. Instead, we generate (a la Het et al., 2020) representations of keys from a *second network* that is updated more slowly.
			- This approach, called MoCo, considers two networks
				- One for keys, parameterized by $\theta_k$
				- One of the query, parametrized by $\theta_q$
			- The parameters of the query network are updated with backprop and SGD, while the parameter of the key network (or momentum encoder) is updated from the parameters of the query network, using an exponential moving average.
- 

Abstract
> Recently, information retrieval has seen the emergence of ==dense retrievers==, using neural networks, as an alternative to classical sparse methods based on term-frequency. These models have obtained state-of-the-art results on datasets and tasks where large training sets are available. However, they ==do not transfer well to new applications with no training data, and are outperformed by unsupervised term-frequency methods such as BM25.== In this work, ==we explore the limits of contrastive learning as a way to train unsupervised dense retrievers== and show that it leads to strong performance in various retrieval settings. On the BEIR benchmark our unsupervised model outperforms BM25 on 11 out of 15 datasets for the Recall@100. ==When used as pre-training before fine-tuning==, either on a few thousands in-domain examples or on the large MS~MARCO dataset, ==our contrastive model leads to improvements on the BEIR benchmark==. Finally, we evaluate our approach for multi-lingual retrieval, where training data is even scarcer than for English, and show that our approach leads to strong unsupervised performance. Our model also exhibits strong cross-lingual transfer when fine-tuned on supervised English data only and evaluated on low resources language such as Swahili. We show that our unsupervised models can perform cross-lingual retrieval between different scripts, such as retrieving English documents from Arabic queries, which would not be possible with term matching methods.

# Paper Figures
![[Pasted image 20240527135340.png]]
![[Pasted image 20240527135357.png]]
![[Pasted image 20240527135414.png]]
![[Pasted image 20240527135515.png]]
