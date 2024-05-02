---
aliases:
  - Open-Retrieval Question Answering
---

June 1, 2019 (Before all of [[REALM]], [[Retrieval-Augmented Generation|RAG]], [[RETRO]], etc.)
[[DeepMind]] (Same lead author Kenton Lee as [[REALM]])
Paper: [Latent Retrieval for Weakly Supervised Open-Domain Question Answering](https://arxiv.org/abs/1906.00300)
#zotero 
Takeaway: An early example of question-answering using a [[Bi-Encoder]], retriever, as far as I can tell? After documents are retrieved, they're fed another encoder that encodes both the question and the text of each evidence block. This encoder is used to predict the start and end portions of the answer within the text, so there isn't a "real" generative component in this question-answering as far as I can tell. Provides a lot of information on the Inverse Cloze Task pretraining for the Bi-Encoder, so that we can give it a "Warm start" before jointly training the rest.

----
ORQA is a jointly-trained retriever and reader model that learns to retrieve evidence from an open corpus, and is supervised only by question-answer string pairs. 
- The retriever component is "warmed up" using the *Inverse Cloze Task (ICT)*.
	- The goal of our pre-training procedure is for the retriever to solve an *unsupervised* task that closely resembles evidence retrieval for QA.
	- The unsupervised analog of a question-evidence pair is a *sentence-context pair* -- the context of a sentence is semantically relevant and can be used to infer information missing from the sequence. Given a sentence, the ICT task, asks us to predict its context.
	- Requires learning more than word matching features, since the pseudo-question isn't present in the evidence.
		- Eg the pseudo question: "They are generally slower than horses, but their great stamina helps them outrun predators" doesn't mention Zebras, but the retrieved context might be: "Zebras have four gaits: walk, trot, canter, and gallop."
		- Being able to infer these semantics from under-specified language is what sets QA apart from traditional IR!
	- ICT accomplishes two main goals:
		- Despite the mismatch between sentences during pre-training and questions during fine-tuning, we expect zero-shot evidence retrieval performance to be sufficient for bootstrapping the latent-variable learning.
		- There's no such mismatch between pretrained evidence blocks and downstream evidence blocks. We can expect the block encoder BERT_B to work well without further training. Only the question encoder needs to be finetuned on downstream data.
- The scoring components are derived from [[Bidirectional Encoder Representations from Transformers|BERT]]s, and we just use a dot product similarity metric.
- We fine-tune all parameters in the system *except those in the evidence block encoder*, so that we don't have to re-encode all of our documents as we fine-tune the model.

Notes: 
- "QA is fundamentally different from IR. Whereas IR is concerned with lexical and semantic matching, questions are *by definition* under-specified, and require more language understanding ((and possibly world understanding)), since users are explicitly looking for unknown information."


Abstract
> Recent work on open domain question answering (QA) assumes strong supervision of the supporting evidence and/or assumes a blackbox information retrieval (IR) system to retrieve evidence candidates. We argue that both are suboptimal, since ==gold evidence is not always available==, and QA is fundamentally different from IR. ==We show for the first time== ==that it is possible to jointly learn the retriever and reader from question-answer string pairs and without any IR system==. In this setting, evidence retrieval from all of Wikipedia is treated as a latent variable. Since this is impractical to learn from scratch, we pre-train the retriever with an ***Inverse Cloze Task***. We evaluate on open versions of five QA datasets. On datasets where the questioner already knows the answer, a traditional IR system such as BM25 is sufficient. On datasets where a user is genuinely seeking an answer, we show that learned retrieval is crucial, outperforming BM25 by up to 19 points in exact match.

# Paper Figures

![[Pasted image 20240501215708.png]]
![[Pasted image 20240501221305.png]]
![[Pasted image 20240501222533.png|250]]
Above: Explanation of the Inverse Cloze Task (ICT) used for pre-training/warm-starting of the retriever.

![[Pasted image 20240501224712.png]]


