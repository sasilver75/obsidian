January 2, 2021 (8 months after [[ColBERT]])
[[Omar Khattab]], [[Christopher Potts|Chris Potts]], [[Matei Zaharia]]
Paper: [Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval](https://arxiv.org/abs/2101.00436)
#zotero 
Takeaway: Baleen is a system for [[Multi-Hop]] question-answering, using [[ColBERT]] as a retriever. It introduces:
- **==condensed retrieval==** (where we summarize  retrieved passages after each hop into a single compact context)
- A ==focused late interaction== passage retriever (==FLIPR==) that allows different parts of the same query representation to match disparate relevant passages.
- **==Latent hop ordering==**, a strategy where the retriever itself selects the series of hops.

Note: "Baleen" in marine biology refers to the filter-feeding system that baleen whales use to capture small organisms, filtering out seawater. Similarly, our system seeks to capture relevant facts from a sea of documents.

---

Notes: 
- In [[Open-Domain]] reasoning, models are tasked with retrieving evidence from large corpuses to answer questions, verify claims, or exhibit broad knowledge; in practice, such open-domain tasks often require [[Multi-Hop]] reasoning, where evidence must be extracted from two or more documents.
	- ==The model must use *partial* evidence it retrieves to bridge its way to additional documents leading to an answer==.
- Challenges:
	- Multi-hop queries encompass multiple information needs.
	- Retrieval errors in each hop propagate to subsequent hops
	- Because of the dependency between hops, retrievers must learn an effective sequence of hops, where previously-retrieved clues lead to other relevant passages. These ==inter-passage dependencies can be non-obvious for many-hop problems==, and it can be expensive to annotate one (or every) sequence in which facts could be retrieved.
- These challenges call for
	- Highly expressive query representations
	- Robustness to retrieval errors
	- Scalability to many hops over massive document collections
- 📉 Existing SoTA systems ...
	- Rely on bag-of-words or single-vector dot-product retrievers, whose capacity to model and open-domain question is limited.
	- Embed trade-offs when it comes to *hopping*, either employing ==brittle greedy search== (limiting recall per hop) or ==beam search over an exponential space== (reducing the scalability to many hops) or ==assuming explicit links connecting every passage with related entities== (which ties them to link-structured corpora)
- ==Baleen== Overview
	- Scalable multi-hop reasoning system that improves accuracy and robustness.
	- We introduce a ==condensed retrieval== architecture, where the retrieved facts from each hop are summarized into a short context that becomes a part of the query for subsequent hops. This allows for effective scaling to many hops (unlike beam search), and we find that it complements greedy search (i.e. taking the best passage per hop) to improve recall considerably.
	- We tackle the complexity of queries by proposing a ==focused late interaction== passage retriever (==FLIPR==), a robust learned search model that allows different parts of the same query representation to match disparate relevant passages. Inherits the scalability of vanilla late interaction paradigm of [[ColBERT]] but uniquely allows the same query to exhibit tailored matching patterns against each target passage.
	- We device ==latent hop ordering==, a weak supervision strategy that uses the retriever itself to select effective hop paths.
- Related Work
	- **Open-domain reasoning**:
		- The most popular such open-domain task is [[Question Answering]], but other open-domain tasks include claim verification, question generation, and open dialogue. Many of these are *knowledge-intensive tasks*.
		- Most relevant models to our work are OpenQA models that include *learned* retrieval components, like [[ORQA]], [[REALM]], [[Dense Passage Retrieval|DPR]], and [[Retrieval-Augmented Generation|RAG]].
	- **Multi-hop open-domain reasoning**:
		- Many open-domain tasks can be solved by finding *one* relevant passage in the corpus (often by design). In contrast, a number of recent works explore multi-hop reasoning over multiple passages (eg [[HotpotQA]])
	- **Multi-hop open-domain models**:
		- To conduct the "hops", many prior multi-hop systems assume *explicit links* connecting every passage with related entities; we argue that this risks tying systems to link-structured knowledge bases (like Wikipedia) or producing brittle architectures tailored for datasets constructed by following hyperlinks.
		- Recently, MDR and IRRR were introduced, both state-of-the-art systems that assume no explicit link structure. Instead, they use an *iterative retrieval* paradigm that retrieves passages relevant to the question, reads the passages, then formulates a new query for another hop if necessary.
	- [[ColBERT]]: late interaction paradigm
		- Most learned-retrieval systems for OpenQA encode every query and passage into a *single* dense vector; ColBERT argues that these aren't sufficiently expressive for retrieval in many scenarios, introducing ==late interaction==, a paradigm representing each query/document at finer granularity; it uses a vector for *each token* in the query/documents.
		- ColBERT uses a BERT encoder to embed the query into a matrix of N vectors, and every passage into a matrix of M vectors (for M tokens per passage). The passage representations are query-independent and are thus computed offline, and indexed or fast retrieval. At query time, we compute a passage's similarity score by finding the maximum similarity (MaxSim) score between each vector in q's representation and *all* the vectors of *d*, then summing these MaxSim scores.
		- While existing retrievers seek passages that match *all* of the query, multi-hop queries can be long and noisy and need to match disparate contexts. ==FLIPR== handles this explicitly with *focused* late interaction (more on this later).
	- Relevance-guided supervision
		- Khattab et all have a 2021 paper and an iterative strategy for [[Weak Supervision]] of retrievers called [[Relevance-Guided Supervision]] (RGS).
		- It assumes that no labeled passages are supplied for training the retriever; instead, every training question is associated with a short answer string whose presence in a passage is taken as a weak signal for relevance.
		- Starts with an off-the-shelf retriever and uses it to collect the top-k passages for each training question, dividing these passages into positive and negative examples based on inclusion of the short answer string.
		- These examples are then used to train a stronger retriever, which is then used to repeat this process one or two times. 
		- Our latent hop ordering in Baleen is inspired by RGS, but unlike RGS, we *do* have gold-labeled passages for training, but we crucially have multiple hops, whose order is not given. 
- Baleen
	- Uses an iterative retrieval paradigm to find relevant facts in >= 1 hops.
	- In every hop, FLIPR uses $Q_{t-1}$ to retrieve K passages from the full corpus. These passages are fed into a two-stage *condenser*, which reads these passages and extracts the most relevant *facts*. These facts are collected into a single sequence, and added to $Q_t$ for the subsequent hop, if needed.
	- Once all hops are complete, Baleen's task-specific *reader* process $Q_T$, which now contains the query and all condensed facts, to solve a downstream task. 
	- Baleen's retriever, condenser, and reader are all implemented as Transformer encoders and trained individually (BERT-base for FLIPR, ELECTRA-large for other components).
	- ==FLIPR: focused late interaction passage retreiver==
		- Our query encoder reads $Q_{t-1}$ and outputs a vector representation of every token in the input. Each query embedding interacts with all passage embeddings via a MaxSim operator, permitting us to inherit the efficiency/scalability of ColBERT.
		- While ColBERT sums ALL MaxSim outputs indiscriminately, FLIPR considers only the strongest-matching query embeddings for evaluating each passage -- it sums only the top-$\hat{N}$ partial scores from N scores, with $\hat{N} < N$ .
		- We refer to this top-k filter as a =="focused" interaction.==
			- It allows the same query to match multiple relevant passages that are contextually unrelated, by aligning -- during training and inference -- a different subset of the query embeddings with different relevant passages. ((This seems critical but I don't quite understand it))
		- (Tapping out on the rest of this section, it's difficult)
	- Supervision: ==Latent Hop Ordering==
		- For every training example, our datasets supply *unordered* gold passages -- but dependencies often exist in retrieval! Such dependencies can be complex for 3 and 4-hop examples.
		- We propose a generic mechanism for identifying the best gold passage(s) to learn to retrieve next!
			- The ==key insight==  is that, among the gold passages, a weakly-supervised retriever with strong inductive biases would reliably "prefer" those passages that it can more naturally retrieve, given the hops so far.
			- So given $Q_{t-1}$, we can train the retriever with *every* remaining gold passage, and, among those, label as positives for $Q_{t-1}$ only those that are highly-ranked by the model. We subsequently use these weakly-labeled positives to train another retriever for all the hops.
			- (See the Paper Figures algorithm, where I copy some of the instructions in more detail)
	- Condenser (ie ==Condensed Retrieval==): Per-Hop Fact Summarization
		- After each hop, the condenser proceeds in two stages
			1. In the first stage, it reads $Q_{t-1}$ and each of the top-K passages retrieved by FLIPR (the input passages are divided into their constituent sentences and every sentence is prepended with a special token). The output embedding corresponding to these sentence markers are scored via a linear layer, assigning a score to each sentence in every retrieved passage... we train the first-stage condenser with a cross-entropy loss over the sentences of two passages, a positive and negative.
				- ((Given the previous-timestep's query and top-K retrieved passages from FLIPR, assign a score to each sentence in every retrieved passage. Goal: find the top-k facts))
			2. Across the K passages, the top-k facts are identified and concatenated into the second-stage condenser model. As above, we include special tokens between the sentences, and the second-stage condenser assigns a score to every fact while attending jointly over all of the top-k facts, allowing a direct comparison within the encoder. Every fact whose score is positive is selected to proceed, and the other facts are discarded; we train the second-hop condenser over sets of 7-9 facts, some positive and others (sampled) negative, using a linear combination of cross-entropy loss for each positive fact (against all negatives), and binary cross-entropy loss for each individual fact.
				- ((Across the passages, identify the top-k facts (where facts are sentences, I think), and concatenate them into the condenser. The second-stage condenser assigns a score to every *fact*, while attending jointly over all of the top-k facts. Every fact whose score is positive gets to proceed. Goal: Filter the top-k facts))
				- Seems to scale better to more hops, as it represents the facts from K long passages using only a few sentences. 
		- Authors seem to say that the reranking approach and condensed retrieval approaches are complementary, and that a single retriever combining both pipelines together seems to outperform standard rerank-only pipeline.

Abstract
> ==Multi-hop reasoning== (i.e., reasoning across two or more documents) is a key ingredient for NLP models that leverage large corpora to exhibit broad knowledge. To retrieve evidence passages, ==multi-hop models must contend with a fast-growing search space across the hops, represent complex queries that combine multiple information needs, and resolve ambiguity about the best order in which to hop between training passages==. We tackle these problems via ==Baleen==, a system that improves the accuracy of multi-hop retrieval while learning robustly from ==weak training signals== in the many-hop setting. To tame the search space, we propose ==condensed retrieval==, a ==pipeline that summarizes the retrieved passages after each hop into a single compact context==. To model complex queries, we introduce a ***==focused late interaction retriever==*** that allows different parts of the same query representation to match disparate relevant passages. Lastly, to infer the hopping dependencies among unordered training passages, we devise ==latent hop ordering==, **a weak-supervision strategy in which the trained retriever itself selects the sequence of hops**. We evaluate Baleen on retrieval for two-hop question answering and many-hop claim verification, establishing state-of-the-art performance.

# Paper Figures
![[Pasted image 20240519204441.png]]

![[Pasted image 20240519211939.png]]

![[Pasted image 20240519214128.png]]

![[Pasted image 20240519215956.png]]
Above: Description of ==latent hop ordering== algorithm
- Assumes that we've already trained a single-hop/first-hop retriever $R_1$ in the manner of relevance-guided supervision.
- We use R1 to retrieve the top-k (eg 1000) passages for each training question. 
- We then divide these passages into positives P1 and negatives N1; positives are *highly-ranked* gold passages and negatives are the non-gold passages.
- We expand first-hop queries with the oracle facts from P1 to obtain the queries for the second hop.
- We train a second-hop retriever R2 using the aforementioned queries and negatives -- as we don't have second-hop positives, as a form of weak supervision, we train the retriever with aLL gold passages (per query), besides those already in P1. Once trained, we use R2 to discover the second-hop positives P2, to collect negatives N2, and to expand the queries.
- We repeat this procedure for the third hop onward; once this iterative procedure is complete, we've bootstrapped positives (and negatives) corresponding to every retrieval hop for each query.
- We combine these sets to train a *single* multi-hop retriever, which takes a query $Q_{t-1}$ and learns to discriminate the positives in $P_t$ from the negatives in $N_t$ for every t.