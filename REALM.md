---
aliases:
  - Retrieval-Augmented Language Model Pre-Training
---
February 10, 2020 (3 months before Meta's [[Retrieval-Augmented Generation|RAG]], 4 months before GPT-3)
[[DeepMind]] (Same org who would later go on to do [[RETRO]] in December 2021)
Paper: [REALM: Retrieval-Augmented Language Model Pretraining](https://arxiv.org/abs/2002.08909)
#zotero 
Significance: The Bi-Encoder [[Bidirectional Encoder Representations from Transformers|BERT]]-*inspired* retriever learns to find documents that help a knowledge-augmented encoder predict these masked tokens in an [[Masked Language Model|MLM]] fashion ("The knowledge retriever and knowledge-augmented encoder are jointly-pretrained). Realm is an [[Encoder-Only Architecture]] -- it doesn't have a decoder component.

---

Notes:
- This paper talks about jointly pretraining both a retrieval component as well as a knowledge-augmented encoder (given an input x and retrieved document z, the knowledge-augmented encoder defines p(y|z,x)).
- Because we're ==jointly pretraining== our retriever with our downstream knowledge-augmented encoder, what happens with all of the indexed documents when we update our retriever's parameters?
	- They note on page 5 some tips about the cold-start problem of retrieving useless documents, which are then learned to be ignored by the model. They warm-start their embedding model using a simple training objective known as the ==Inverse Close Task==.
		- This is the same warmup that's used in the earlier [[ORQA]] paper by the same lead author.
- One of the problems of finding our top k documents using Maximum Inner Product Search (MIPS) algorithms is that as we update the parameters of our document-embedding model, these ==embeddings become stale== -- and re-embedding all of our documents after every gradient update of our document-embedding model is too expensive! Our ==solution== is to asynchronously re-embed and re-index all documents every *several hundred training steps*, allowing the index to become slightly stale between refreshes. Empirically, this seems to be fine.


Abstract
> Language model pre-training has been shown to capture a surprising amount of world knowledge, crucial for NLP tasks such as question answering. However, this knowledge is stored implicitly in the parameters of a neural network, requiring ever-larger networks to cover more facts.  
> To capture knowledge in a more modular and interpretable way, ==we augment language model pre-training with a latent knowledge retriever==, which allows the model to retrieve and attend over documents from a large corpus such as Wikipedia, ==used during pre-training, fine-tuning and inference==. For the first time, we show how to pre-train such a knowledge retriever in an unsupervised manner, using masked language modeling as the learning signal and backpropagating through a retrieval step that considers millions of documents.  
> We demonstrate the effectiveness of Retrieval-Augmented Language Model pre-training (REALM) by fine-tuning on the challenging task of Open-domain Question Answering (Open-QA). We compare against state-of-the-art models for both explicit and implicit knowledge storage on three popular Open-QA benchmarks, and find that we outperform all previous methods by a significant margin (4-16% absolute accuracy), while also providing qualitative benefits such as interpretability and modularity.


# Paper Figures

![[Pasted image 20240501180803.png|250]]
Above: REALM augments LM *pre-training*, and signal from the LM objective backpropagates all the way through the retriever.

![[Pasted image 20240501182136.png|450]]
Above: The "language modeling task" on the left side is actually a *masked* language modeling task, a la BERT.

![[Pasted image 20240501185130.png]]
Explanation of how they train the embedder/retriever together, given that as they update the embedder/retriever, the document embeddings become stale/must be updated.



