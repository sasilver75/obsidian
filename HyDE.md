---
aliases:
  - Hypothetical Document Embeddings
---
December 20, 2022
LTI @ Carnegie Mellon, UWaterloo
Link: [Precise Zero-Shot Dense Retrieval Without Relevance Labels](https://arxiv.org/abs/2212.10496)
#zotero 
Takeaway: Given a query, ask an LLM to hallucinate a hypothetical document that would be relevant to answering the question. Embed the resulting hypothetical document, and retrieve the *actual* documents that near it in document-embedding space! The hope is that the document encoder's dense bottleneck filters out incorrect details in the hypothetical document. This is all in the context of zero-shot information retrieval, where we don't have relevance labels between queries and documents. If we *do*, then perhaps HyDE isn't an appropriate technique -- but if we're in a situation where we don't have relevance judgements for query-passage pairs, but need to bootstrap a retrieval system, HyDE might be an interesting place to start.


----

Notes
- The model hopes to decompose dense retrieval into two tasks:
	1. Generative component, where ==we use an instruction-following language model to generate hypothetical documents relevant to our query==.
		- This document can contain factual errors, but is hopefully otherwise like a relevant document.
	2. Document-Document similarity component, where ==we use a contrastive encoder to retrieve REAL documents nearby our hypothetical documents==.
		- We expect the encoder's dense bottleneck to serve as a lossy compressor, where the extra hallucinated details are filtered out from the embedding. ((I don't see why they would? This is the same Document embedder trained to embed _real_ documents, right? Hmmm))
		- The retrieval process leverages the document-document similarity encoded via inner-product during contrastive training, but the query-document similarity score in HyDE isn't actually explicitly modeled or computed. ==Instead, the retrieval task is cast into two NLI and NLG tasks!==
			- "HyDE appears unsupervised; No model is trained in HyDE ((It's just a technique)): Both the generative model and contrastive encoder remain intact. Supervision signals were only involved in instruction learning of our backbone LLM."
- The "backbone models" used for HyDE include [[InstructGPT]] for the generative component and Contriever (Izucard et al, 2021) as the contrastive retriever.
- Authors reference a concurrent paper, {Asai et al., 2022}, where they studied "Task-aware Retrieval with Instructions", in which they fine-tuned dense encoders that can also encode task-specific instructions prepended to a query.
	- In contrast, we use an unsupervised encoder and handle different tasks and their instruction with an instruction-following generative LLM.
- *Zero-shot* dense retrieval requires the learning of two embedding functions into the *same* embedding space, where inner product captures *relevance.* ***But without relevance judgements/scores to fit, learning becomes intractable***!
	- ==HyDE *circumvents* this problem by performing search in *document-only* embedding space that captures document-document similarity!==
		- ==This can be easily learned using unsupervised contrastive learning== (Izacard et al, 2021; Gao et al, 2021; Gao and Callan, 2022).
		- We take our query, use our LM with a prompt like `write a paragraph that answers the question`to generate a hypothetical document. We offload relevance modeling from the representation learning model to an NLG model that generalizes significantly more easily, naturally, and effectively. 
		- We can now encode the generated document using the document encoder. The inner product is computed between that vector and the set of all document vectors.
		- The encoder function also serves as a lossy compressor that outputs dense vectors, where the extra details are filtered out and left from the vector. ((How, when the encoder is trained in an unsupervised fashion?))
- "==We argue HyDE is of practical use, though not necessarily over the entire lifespan of a search system==. At the very beginning of the life of the search system, serving queries using HyDE offers performance compatible to a fine-tuned model, which no other relevance-score-free (supervision free, with respect to document/query relevances) model can offer. As the search log grows, a supervised dense retriever can be gradually rolled out. As the dense retriever becomes stronger, more queries will be routed to it, with only less common ones going to the HyDE backend."


Abstract
> While dense retrieval has been shown effective and efficient across tasks and languages, ==it remains difficult to create effective fully zero-shot dense retrieval systems when no relevance label is available==. In this paper, we recognize the difficulty of zero-shot learning and encoding relevance. Instead, we propose to pivot through Hypothetical Document Embeddings~(==HyDE==). ==Given a query==, HyDE ==first zero-shot instructs an instruction-following language model (e.g. InstructGPT) to generate a hypothetical document==. The document captures relevance patterns but is unreal and may contain false details. ==Then, an unsupervised contrastively learned encoder==~(e.g. Contriever) ==encodes the document into an embedding vector==. This vector ==identifies a neighborhood in the corpus embedding space, where similar real documents are retrieved based on vector similarity==. This second step ground the generated document to the actual corpus, with the encoder's dense bottleneck filtering out the incorrect details. Our experiments show that ==HyDE significantly outperforms the state-of-the-art unsupervised dense retriever Contriever== and shows strong performance comparable to fine-tuned retrievers, across various tasks (e.g. web search, QA, fact verification) and languages~(e.g. sw, ko, ja).

# Paper Figures
![[Pasted image 20240502162616.png]]
Above: Interesting that the prompt instructions change depending on the query. See that we generate a hypothetical document based on this instruction+query, then use our "Contriever" to retrieve *actual* documents near our hypothetical one in embedding space.