#article 
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
- In 2019, search was revolutionized with [[BERT]] for ranking.
![[Pasted image 20240425221743.png]]
- The standard approach is above in ==2a==; each document is concatenated with the query, and both are fed jointly into a BERT model, fine-tuned to estimate relevance.
	- BERT doubled the MRR@10 quality metric over BM25 on the popular [[MS MARCO]] passage ranking leaderboard, but also posed a ==fundamental limitation==: *scoring* each query-document pair required *billions* of operations!
	- As a result, ==BERT can only be used to *re-rank the top-k (eg 1,000)* documents that were already extracted (eg by simpler methods like BM-25)==.
	- The key limitation of this approach is that it encodes queries and documents *jointly*.

- Many representation-similarity systems have been proposed to tackle this, some of which repurpose BERT in the manner of ==2b== above.
	- In these systems (SBERT, ORQA, DPR, ANCE), every document is fed into a BERT encoder that produces a dense vector meant to capture the semantics of the document.
	- At search time, the query is encoded, separately, through another BERT encoder, and the top-k related documents are found using a dot product between the query and document vectors.
	- By removing the expensive interactions between the query and document, these models scale far more efficiently than the approach in 2a.
		- ((I think it's more like in the former, you have to encode each query/document pair, whereas in this one, you encode all the documents ahead of time, and then at query time you just encode the query and do some sort of fast approximate-nearest-neighbor search.))
	- Nonetheless, ==representation-similarity models suffer from an architectural bottleneck==:
		- They encode the query and document into *coarse-grained representations*  (one vector for a whole document chunk!) and model relevance as a *single* dot product!
		- This greatly diminishes quality compared with expensive re-rankers that model *to*






