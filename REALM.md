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
- Because we're jointly pretraining out retriever with our downstream knowledge-augmented encoder, what happens with all of the indexed documents when we update our retriever's parameters?


Abstract
> Language model pre-training has been shown to capture a surprising amount of world knowledge, crucial for NLP tasks such as question answering. However, this knowledge is stored implicitly in the parameters of a neural network, requiring ever-larger networks to cover more facts.  
> To capture knowledge in a more modular and interpretable way, ==we augment language model pre-training with a latent knowledge retriever==, which allows the model to retrieve and attend over documents from a large corpus such as Wikipedia, ==used during pre-training, fine-tuning and inference==. For the first time, we show how to pre-train such a knowledge retriever in an unsupervised manner, using masked language modeling as the learning signal and backpropagating through a retrieval step that considers millions of documents.  
> We demonstrate the effectiveness of Retrieval-Augmented Language Model pre-training (REALM) by fine-tuning on the challenging task of Open-domain Question Answering (Open-QA). We compare against state-of-the-art models for both explicit and implicit knowledge storage on three popular Open-QA benchmarks, and find that we outperform all previous methods by a significant margin (4-16% absolute accuracy), while also providing qualitative benefits such as interpretability and modularity.


# Paper Figures

![[Pasted image 20240501180803.png|250]]
Above: REALM augments LM *pre-training*, and signal from the LM objective backpropagates all the way through the retriever.

![[Pasted image 20240501182136.png|450]]
Above: The "language modeling task" on the left side is actually a *masked* language modeling task, a la BERT.



