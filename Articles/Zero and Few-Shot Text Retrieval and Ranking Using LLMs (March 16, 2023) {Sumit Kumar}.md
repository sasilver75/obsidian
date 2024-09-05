Link: https://blog.reachsumit.com/posts/2023/03/llm-for-text-ranking/

This was a great blog post about doing zero/few-shot domain transfer in RL using synthetic data from LLMs!

---

Text retrieval and ranking simply refers to the task of producing a ranked list of the most relevant documents or passages out of a large text collection, given a user query.

## Syntactic vs Semantic Approaches
- Some of the most common methods used are based on keyword matching, sophisticated dense representations, or a hybrid of the two.
- [[BM25]] is a simple term-matching-based scoring function produced decades ago, but is still immensely popular. It only uses the terms common to both the query and document, and ==isn't able to recognize synonyms and distinguish between ambiguous words==. Still, a lot of studies in the field have proven BM25 to be a ==really strong baseline==.
- Neural information retrieval, on the other hand, captures and compares the ==semantics== of queries and documents. Dense representation-based NN models usually take the form of a [[Bi-Encoder]] network or a [[Cross-Encoder]] network.
	- [[Bi-Encoder]]s *independently* learn latent representations for query and documents, and interacts them only the final layer to calculate some similarity function (dot-product, cosine, MaxSim, euclidean distance). An indexing solution like [[FAISS]] is often used to fetch document embeddings in real-time, during inference. The ability to encode and index passages ahead of time makes them a popular choice.
	- [[Cross-Encoder]]s take a query *and* document vector as input, and calculate a scalar (or sometimes, binary) relevance score. Achieve *higher performance* than Bi-Encoders due to the richer interactions between queries and documents, but they don't scale well because you can't separately encode documents ahead of time.
		- Metrics like Accuracy, Mean Rank, and [[Mean Reciprocal Rank]] (MRR) are used if the relevance score is binary, otherwise metrics like Discounted Cumulative Gain (DCG) and [[Normalized Discounted Cumulative Gain]] (nDCG) are used if a graded relevance score is used.

## Cascade Ranking Pipline
- In designing e2e retrieval systems, we often have to balance the tradeoffs between effectiveness and efficiency.
	- (eg) Cross-Encoders are highly effective, but their time complexity can be prohibitive for most real-time use cases operating over large document collections.
- To address this, a ==cascading ranking pipeline== is adopted, where increasingly complex ranking functions progressively prune and refine the set of candidate documents to minimize retrieval latency and maximize result set quality.s
	- A relatively simpler algorithm like Elasticsearch, BM25, Bi-Encoder, or some combination might be used to retrieve the top-n (eg 100, 1000) candidate documents.
	- Then, a more complex algorithm like a Cross-Encoder is used to *re-rank* these retrieved candidates.
		- Often, modern IR systems will use ==*multi-stage* reranking==, by using a bi-encoder followed by a *cheap* cross-encoder, followed by a *more expensive* cross-encoder for final reranking.

## Zero and Few-Shot Settings
- A significant challenge in developing neural retrievers is the lack of domain-specific training data.  Low-resource target domains often lack labeled training data, and it's difficult+expensive to scale human labeling of query-document relevance.
- There are a few "general-purpose" datasets like [[MS MARCO]] and [[Natural Questions]], but they don't always generalize well for out-of-domain uses,  are often not available under commercial licenses, and are usually English-only.
- To address the difficulty of zero/few-shot transfer to new domains, a recent line of research work has started using *generative LLMs* to do zero-shot/few-shot domain adaptations of retrieval and ranking models. 
	- A lot of research here focuses on prompting LLMs with instructions for the task and a few examples in natural language to generate *synthetic examples* that can be used to finetune task-specific models.

## Using LLMs for Zero or Few-Shot Domain Adaptation

### InPars 🦜 (Jul 10, 2023)
- The Inquisitive Parrots for Search ([[InPars]]; Jul 10, 2023) system used LLMs to generate synthetic data for IR tasks in a few-shot manner, under minimal supervision. This data is then used to finetune a neural reranker model that is used to rerank search results in a pipeline composed of a BM25 retriever and a neural monoT5 reranker. ![[Pasted image 20240527184145.png]]
- The training set has query, positive, and negative document triplets ($q,d_+,d_-$)
- Given a collection of documents, 100k documents randomly sampled, and GPT-3 Curie is used as an LLM generating one question corresponding to each of the sampled document based on greedy decoding (temperature=0)
- Experimented with two prompting strategies
	- Vanilla prompting, which uses 3 randomly-chosen parts of the document, and relevant questions from the MS MARCO dataset. 
	- Guided by Bad Questions (GBQ) uses a strategy similar to Vanilla, but the corresponding question from MS MARCO is marked as a *bad* question, while a manually-created example is marked as a *good* question. This was done to encourage the model to produce more contextual-aware questions than the one from MS MARCO.
- Negative examples were sampled from top candidates returned by the BM25 method that *weren't* relevant documents.
- ![[Pasted image 20240527184556.png]]
- The Vanilla prompting strategy worked better for two out of the five tested datasets, while GBQ performed better for the other three. 

### Promptagator 🐊 (Sep 23, 2022)
- In Prompt-based Query Generation for Retriever ([[Promptagator]]; Sep 23, 2022), ==authors argued that different retrieval tasks have different search intents== (like finding evidence, retrieving entities, etc.).
- They proposed a few-shot setting for dense retrievers where each task comes with a short description and a few annotated examples illustrate the search *intents*. 
	- The method relies on these few (2-8) in-domain relevant query-document examples from the target tasks, *without* using any query-document pairs from other tasks or datasets!
- They ran the prompt on all documents from the corpus, and created a large set of synthetic examples using a FLAN-137B LLM. 
- The quality of generated queries was further improved by ==round-trip consistency==, i.e. that the query should result in the retrieval of the source passage. They kept the generated pair only if the corresponding document occurs among top-K (they used K=1) passages returned in the prediction by the same retriever.![[Pasted image 20240527185114.png]]
- Finally, they trained a retriever (dual encoder) followed by a cross-attention reranker on the filtered data. Promptagator outperformed ColBERTv2 and SPLADEv2 on all tested retrieval tasks.


### UPR (Unsupervised Passage Re-Ranker) (Apr 15, 2022)
- In [[UPR]] (Apr 15, 2022), authors proposed a fully-unsupervised pipeline consisting of a retriever and a reranker that can outperform supervised dense retrieval models (like [[Dense Passage Retrieval|DPR]]) alone. ![[Pasted image 20240527190414.png]]
Above: ((My understanding is that UPR wants to compute a relevance score for each passage z_i given q, which we denote as p(z_i|q). The probability is calculated using a pre-trained language model. The intuition here is that if a passage can "explain" or "generate" the question well, then it's likely relevant to the question.))
- The retriever could be based on any supervised method like BM25, Contriever, or MSS; the only requirement is that the retriever provides the K most relevant passages.
- The rerankers are given the following prompt in a zero-shot manner:
	- `Passage: {p}. Please write a question based on this passage.`
- The reranking score is computed as $p(z_i|q)$ for each passage $z_i$ and query $q$. 
	- The paper shows that this relevancy score can be approximated by computing the average log-likelihood of the question, conditioned on the passage, i.e. $logp(q|z)$.
- UPR performs well, but due to LLM usage in the pipeline, UPR suffers from high-latency issues with a complexity directly proportional to the product of question and passage tokens and the number of layers in LLM.

### HyDE: Hypothetical Document Embeddings (Dec 20, 2022)
- In [[HyDE]] (Dec 20, 2022), Gao et al proposed a novel zero-shot dense retrieval method. 
- Given a query, they first zero-shot instruct [[InstructGPT]] to generate a synthetic ("hypothetical") document, basically asking "What sort of document would answer this question?". Next, we use an unsupervised contrastively learned encoder (like a Contriever) to encode this hypothetical document into an embedding vector. Finally, we use a nearest-neighbor approach to fetch similar real documents based on vector similarity in corpus embedding space.
- The assumption is that the bottleneck layer in the encoder filters out factual errors and incorrect details in the hypothetical document. In their experiments, HyDE outperformed the state-of-the-art unsupervised Contriever method and also performed comparably to finetuned retrievers on various tasks.
- ((HyDe is a little slower though, because it involves a LM forward pass. I think it's true that HyDE is useful to "bootstrap" retrieval in a new domain, where you don't already have a labeled dataset of matching query/documents.))![[Pasted image 20240527193154.png]]

### GenRead (Generate then Read) (Sep 21, 2022)
- In [[Generate-then-Read]] (Sep 21, 2022), authors replace the retriever in the traditional *retrieve-then-read* QA pipeline with an LLM generator model to create a *generate-then-read* QA pipeline.
- We generate a synthetic document using an InstructGPT LLM, given the input query, and then use the reader model on the generated document to produce the final answer.
	- ((This sounds sort of like HyDE, but instead of using the generated document to then go and retrieve the *real* document, we actually just... use the generated document, passing it to the reader. So there's sort of no REAL retrieval of documents going on. I'm curious as to what the prompts are, and vaguely skeptical about whether this would be better than simply (eg) CoT-prompting the model, rather than asking it to generate a supporting document))
- Using multiple datasets, they show that the LLM-generated document is more likely to contain correct answers than the top retriever document, which "justifies" the use of the generator in this context.
- Document-generator prompt
```
Generate a background document to ansewr the given question. {question placeholder}
```
- Reader zero-shot reading comprehension prompt
```
Refer to the passage below, and answer the following question. Passage: {background placeholder} Question: {question placeholder}
```
- A supervised setting with a reader model like [[Fusion-in-Decoder|FiD]] was shown to provide better performance than a zero-shot setting. ![[Pasted image 20240527194802.png]]
- To improve recall, authors proposed a clustering-based prompt method that introduces variance and diversity to generated documents; they do offline K-Means clustering on GPT-3 embeddings of a corpus of query-document pairs, and at inference a fixed number of documents are sampled from each of these clusters and given to the reader model.

## InPars-v2 (Jan 4, 2023)
- InPars-v2 was an update to the earlier InPars paper where they swapped GPT-3 with GPT-J. They only used the GB strategy proposed in InPars-v1.
- They sampled 100k documents from the corpus and generated one synthetic query per document.
- Instead of filtering them to the top-10k pairs with highest log-probabilities of generation like in v1, they used a relevancy score calculated by a monoT5 model finetuned on MS MARCO, to keep the top010k.

### Inpars-Light
- A reproducibility study of InPars that proposed some cost-effective improvements; instead of GPT-3, they used the open-source BLOOM and GPT-J. Instead of using MonoT5, they used MiniLM/ERNIEv2/DeBERTAv3 rerankers.
- For prompting, they used the 'vanilla' strategy in the InPars paper. For consistency checking, they used the same approach as used in the Promptagator model, but with K value set to 3.

### UDAPDR (Marc 1, 2023)
- Unsupervised Domain Adaptation via LLM Prompting and Distillation of Rerankers ([[UDAPDR]]) used a two-stage LLM pipeline (one powerful and expensive LLM, followed by a smaller and cheaper LLM) to generate synthetic queries in zero-shot settings. These queries are used to finetune a reranker model, which is then distilled into a single efficient retriever. ![[Pasted image 20240527204552.png]]
- This requires access to in-domain passages (But doesn't require in-domain queries or labels). 
	- These passages and LLM-prompting is used to generate a large number of synthetic queries.
	- These passages are first fed to a GPT-3 text-davinci-002 model, using five prompting strategies. Note that the first two prompts are the same as the InPars paper, and the other are three zero-shot strategies from another recent paper.
![[Pasted image 20240527204845.png]]
- The generated (document, synthetic query) pairs are then used to populate the following prompt template, which is used to generate a good query for a new passage through the *smaller* LLM.
	- While the first LLM was given a few (5 to 100) sampled passages, this smaller LLM is given a much larger sampled set (10k to 100k) of passages.
![[Pasted image 20240527205125.png]]
- Similar to earlier work, [[Consistency Filtering]] is applied. UDAPDR uses a zero-shot ColBERTv2 model for this purpose, keeping synthetic queries only if it returns its gold passage within the top-20 results.
- Finally, a [[DeBERTa]]v3-Large reranker is trained using this filtered synthetic data, and distilled into a [[ColBERTv2]] retriever model.
- These experiments show good zero-shot results in long-tail domains.

### DataGen
- In DataGen, Dua eta al. proposed a taxonomy for dataset shift, and showed that zero-shot adaptations don't work well in cases where the target domain distribution is very far from the source domain.
- To fix this, they prompt a [[PaLM]] in few-shot settings to generate queries given an article: `AFter reading the article, <<context>> the doctor said <<sentence>>` for PubMed articles.
	- They replace "doctor" with engineer, journalist, or poster for StackOverflow, DailyMail, and Reddit target corpora respectively.
	- They filter out questions that repeat the passage verbatim, or had 75%+ overlap with it.
- Used both supervised and synthetic data to train their retriever model.




