#article  #premium 
October 5, 2021 (5 months after GPT-3, 1 month before ChatGPT)
Link: https://ai.stanford.edu/blog/retrieval-based-NLP/


This is an article from 2021, so caveat emptor! But Chris recommended reading it in his XCS224U lecture.

----

NLP has witnessed impressive developments in [[Question Answering]], [[Summarization]], and [[Machine Translation]].
- Much of this progress is owed to training ever-larger LMs like [[T5]] and [[Bidirectional Encoder Representations from Transformers|BERT]].


During training, these models distill the facts they read into parametric knowledge they learn highly abstract knowledge representations of entities, events, and facts, as well as specific pieces of knowledge.

Despite the success of LM,s their black-box nature hinders key goals of NLP; they're currently:
1. ==Inefficient==: Requiring Billions or Trillions of Parameters, causing a large environmental impact and excluding many from access to these models.
2. ==Opaque==: Encode knowledge into model weights, but makes it difficult to know which sources the model is using to make a prediction.
3. ==Static==: Expensive to update; We can't efficiently adapt a GPT a to contain information that occurred yesterday -- adaptation requires expensive things like fine-tuning on new corpuses.

The post explores an emerging alternative, ==Retrieval-based NLP==, in which models directly "search" for information in a text corpus to exhibit knowledge, leveraging the representational strengths of language models while addressing the challenges above.

Such models, like [[REALM]], [[Retrieval-Augmented Generation|RAG]], [[ColBERT]], and Baleen, are already advancing the SoTA for talks like open-domain QA and verification of complex claims, ==all with architectures that back their predictions with checkable sources== while while being 100-1000x smaller than GPT-3.

![[Pasted image 20240425221326.png]]

Retrieval-based NLP methods view tasks as ==open book==, meaning that the model learns to search for pertinent passages and to then use the retrieved information for crafting knowledgable responses.
- In doing so, retrieval helps decouple the capacity that languages model have for *understanding* text from how they store *knowledge*.
- Said again, retrieval systems combined with LMs help us ==decouple knowledge (Documents) from understanding (LM)==

This leads to 3 key advantages:
1. ==Tackling Inefficiency==: Retrieval-based models can be much smaller and faster, and thus more environmentally friendly.
2. ==Tackling Opaqueness==: Retrieval-based NLP offers a transparent contract with users: When the model produces an answer, we can read the source it retrieved and judge their relevance/credibility for ourselves.
3. ==Tackling Static Knowledge==: With facts stored as text, the retrieval knowledge store can be efficiently updated or expanded by modifying the text corpus.

# ColBERT: Scalable yet expressive neural retrieval
- Traditionally in IR, search tasks were conducted using ==bag of words== models like [[BM25]], which seek documents containing the *same tokens* as the query!
- In 2019, search was revolutionized with [[Bidirectional Encoder Representations from Transformers|BERT]] for ranking.
![[Pasted image 20240425221743.png]]
- The standard approach is above in ==2a==; each document is concatenated with the query, and both are fed jointly into a BERT model, fine-tuned to estimate relevance.
	- BERT doubled the MRR@10 quality metric over BM25 on the popular [[MS MARCO]] passage ranking leaderboard, but also posed a ==fundamental limitation==: *scoring* each query-document pair required *billions* of operations!
	- As a result, ==BERT can only be used to *re-rank the top-k (eg 1,000)* documents that were already extracted (eg by simpler methods like BM-25)==.
	- The key limitation of this approach is that it encodes queries and documents *jointly*.

![[Pasted image 20240425230222.png|250]]
- Many representation-similarity systems have been proposed to tackle this, some of which repurpose BERT in the manner of ==2b== above.
	- In these systems (SBERT, ORQA, DPR, ANCE), every document is fed into a BERT encoder that produces a dense vector meant to capture the semantics of the document.
	- At search time, the query is encoded, separately, through another BERT encoder, and the top-k related documents are found using a dot product between the query and document vectors.
	- By removing the expensive interactions between the query and document, these models scale far more efficiently than the approach in 2a.
		- ((I think it's more like in the former, you have to encode each query/document pair, whereas in this one, you encode all the documents ahead of time, and then at query time you just encode the query and do some sort of fast approximate-nearest-neighbor search.))
	- Nonetheless, ==representation-similarity models suffer from an architectural bottleneck==:
		- They encode the query and document into *coarse-grained representations*  (one vector for a whole document chunk!) and model relevance as a *single* dot product!
		- This greatly diminishes quality compared with expensive re-rankers that model ***token-level* interactions** between the contents of queries and documents.
			- But is there a way to efficiency scale *fine-grained*, contextual interactions to a massive corpus, without compromising speed or quality? The answer is yes, using a paradigm called [[Late Interaction]], first devised in the ColBERT model, which appeared at SIGIR 2020!

![[Pasted image 20240425230234.png|250]]
- As depicted in figure `2c`, ColBERT independently encodes queries and documents into fine-grained ==multi-vector representations==; it then attempts to softly and contextually *locate each query token inside the document!*
	- For each query embedding, we find the most similar embeddings in the document with a "==MaxSim==" operator, and then sum up all the MaxSims to score the document.
	- This MaxSim is a careful choice that allows us to index the document embeddings for [[Approximate Nearest Neighbor Search|Approximate Nearest Neighber]] (ANN) search, enabling us to scale this rich interaction to millions of passages with latency on the order of tens of milliseconds.
	- ==For instance, ColBERT can search over all passages in English Wikipedia in approximately 70ms per query!== On MS MARCO passage ranking, ColBERT preserved the MRR@10 quality of BERT re-rankers while boosting recall@1k to nearly 97% against the official BM25 ranking's recall@1k of just 81%.


# ColBERT and Baleen: Specializing neural retrieval to complex tasks, with tracked provenance

- While scaling expressive search mechanisms are critical, NLP models need more than just finding the right documents
	- We want NLP models to use retrieval to answer questions, fact-check claims,  respond informatively in a conversation, and identify the sentiment of a piece of text.
	- Many of these ==knowledge-intensive tasks== above are collected in the KILT benchmark. The most popular task is open-domain question-answering, where systems are given a question from any domain and must produce an answer, often by reference to the passage in a large corpus.

Two popular models in this open-domain QA space are [[REALM]] and [[Retrieval-Augmented Generation|RAG]]. these rely on the ORQA and DPR retrievers mentioned earlier. REALM and RAG ***jointly*** tune a ==retriever== as well as a ==reader==, a modeling component that consumed the retrieved documents and produces answers/responses.

Take [[Retrieval-Augmented Generation|RAG]] as an example: its reader is a generative [[BART]] model, which attends to the passages while generating the target outputs.

These two models constitute important steps toward retrieval-based LMs, but ==they suffer from two major limitations==
1. They use the restrictive paradigm of ==Figure 2b== above for retrieval ((I think we'd call this a [[Bi-Encoder]]? Anyways, this uses document-level vectors, instead of token-level vectors, so surely some nuance is lost?)), sacrificing recall: they're ==often unable to find relevant passages== for conducting their tasks.
2. When training the retriever, REALM and RAG collect documents by searching for them inside the training loop, and, to make this practical, ==they freeze the document encoder while fine-tuning, restricting the model's adaptation to the task== ((Why does not freezing the document encoder make it impractical? Because when you change your document encoder weights, you should really be re-encoding all of your documents, which is incredibly expensive. Some strategies batch updates to their encoder to minimize this cost.))

[[ColBERT]]-QA is an open-QA system (TACL'21) that we build on top of ColBERT to tackle both problems
- By adapting ColBERT's expressive search to the task, ColBERT-QA finds useful passages for a larger fraction of the question, thus enabling the reader component to answer more questions correctly and with ==provenance==.
- In addition, ColBERT-QA introduces ==relevance-guided supervision (RGS)==, a training strategy whose goal is to adapt a retriever like ColBERT to the specifics of an NLP task like Open-QA. 
	- RGS proceeds in discrete rounds, using the retriever trained in the previous round to collect "positive" passages that are likely useful for the retriever -- specifically, passages ranked highly by the latest version of the retriever and that also overlap with the gold answer of the question -- as well as challenging "negative" passages. By converging to a high coverage of positive passages and by efficiently sampling hard negatives, ColBERT-QA sets impressive performance benchmarks.

---

A more sophisticated version of the Open-QA task is ==multi-hop reasoning==, where systems must answer questions or verify claims by gathering information from *multiple* sources!
- Systems in this space like GoldEn, MDR, and IRRR find relevant documents and "hop" between them, often by running additional searches, to find all pertinent sources.

((==This sounds awesome!==))

While these models have demonstrated strong performance for two-hop tasks, scaling robustly to *more hops* is challenging, since the search space grows exponentially.

To tackle this, our ==Baleen== system introduces a richer pipeline for multi-hop retrieval: After each retrieval "hop," Baleen summarizes the pertinent information from the passages into a short context that's used to inform future hops.
- ((This just sounds like something like a RNN hidden state, which feels like a (bad) information bottleneck.))
- In doing so, Baleen obviates the need to explore each potential passage at every hop. ((They then claim that this doesn't sacrifice recall, but I don't see how that's impossible, because summarization is compression, and is not lossless.))

Baleen also extends ColBERT's late interaction; it allows the representations of different documents to "focus" on distinct parts of the same query, as each of those documents in the corpus might satisfy a distinct aspect of the same complex query.

As a result of its more deliberate architecture, Baleen saturates retrieval on the popular two-hop [[HotpotQA]] benchmark and dramatically improves performance on the harder four-hop claim verification benchmark HoVer, finding all required passages in 92% of the examples -- up from just 45% for the official baseline, and 75% for a many-hop flavor of ColBERT.

----

# Generalizing models to new domains with robust neural retrieval
- In addition to helping with efficiency and transparency, retrieval approaches promise to make domain generalization and knowledge updates much easier in NLP (vs fine-tuning complete models).
	- Exhibiting up-to-date questions over recent publications on (eg) COVID-19, or developing a chatpot that guides customers to suitable products among those available in a fast-evolving inventory are applications that require NLP models to leverage documents!

Though LLMs are trained with plenty of data from the web, but that snapshot is:
1. ==Static==: The web evolves as the world does; an LM trained from 2020 data doesn't know about 2021 events.
2. ==Incomplete==: Many topics are under-represented in web crawls like C4 and the Pile; there's no guarantee that The Pile contains all papers from the ACL Anthology, and there's no way to plug that in ad-hoc. Even if they did, it's difficult to constrain the model to rely on information in those documents.
3. ==Public-only==: There are many applications that hinge on private text, like internal company policies; GPT-3 will never see such data during training.

With retrieval-based NLP, models learn effective ways to encode and extract information, allowing them to generalize to updated text, specialized domains, or private data without updating the parameters of the model.
- A vision where developers "plug in" their text corpus, like in-house software documentation, which is indexed by a powerful retrieval-based NLP model that can then answer questions/solve classification tasks/generate summaries using the knowledge from the corpus.

An exciting benchmark in this space is [[BEIR]], which evaluates retrievers on their capacity to search "out-of-the-box" on unseen IR tasks, like *Argument Retrieval*, and in new domains, like *Covid-19 research literature*. 
- Not ever IR model generalizes equally; the BEIR evaluations highlight the impact of modeling and supervision choices on generalization.
- For instance, due to its late interaction modeling, a vanilla off-the-shelf ColBERT retriever achieved the strongest recall of all competing IR models in the initial BEIR evaluations, outperforming the other off-the-shelf dense retrievers.

# Summary: Is Retrieval "all you need?"
- The black-box natures of LMs like T5 and GPT-3 makes them 
1. Inefficient to train and deploy
2. Opaque in their knowledge representations, not backing claims with provenance
3. Static in facing a constantly evolving world.

This post as a result explored retrieval-based NLP, where models retrieve information pertinent to solving their tasks from a plugged-in text corpus.
This paradigm allows NLP models to leverage the representational strenghts of language models, while needing much ==smaller architectures==, offering ==transparent provenance== for claims, enabling ==efficient updates and adaptation==.