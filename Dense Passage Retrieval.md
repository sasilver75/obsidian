---
aliases:
  - DPR
---
April 10, 2020
[[Meta AI Research]], UW, Princeton
Paper: [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
#zotero 
Takeaway: Introduces the use (?) of a BERT-based Bi-Encoder for retrieval in the context of open-domain question answering -- it seems to be better than BM25 and TF-IDF. Showed that using dense embeddings (instead of sparse vector spaces like TF-IDF) for documenet retrieval can outperform strong baselines like Lucene BM25.

----

Introduces DPR, which is a [[BERT|BERT]]-based [[Bi-Encoder]] that separately encodes (each with their own BERT base encoder, producing a d=768 representation at the CLS token) queries and passages, and finds relevant passages via the dot product between these two dense vector representations. For retrieval, they index document embeddings into FAISS offline, and at retrieval time they compute the embedding of a query and find the top k passages via approximate nearest neighbors, and provide it to the LM (BERT) that outputs the answer to the question. Beats the pants off of [[BM25]] and [[TF-IDF]] in terms of performance in open-domain question answering.

Notes
- Notes that reading comprehension models can be described with a two-stage framework, where a context *==retriever==* selects a small subset of passages, and then a machine *==reader==* examines the contents and identifies the correct answer.
- Uses the useful-to-me taxonomy of: 
	- ==Corpus== (a collection of documents)
	- ==Documents== (member of a corpus)
	- ==Passages== (chunk of a document)
	- ==Span== (relevant part of a passage)
- Notes that while there are more expressive model forms for measuring the similarity between questions and passages (referring to a [[Cross-Encoder]]), it notes that those models don't allow for efficient pre-computation of passage representations like Bi-Encoders do.
	- EG we don't want to have to do 1,000,000 forward passes of BERT at query time in a Cross-Encoder, since we have to jointly encode both the query and (each) passage in that situation.
- For a similarity function, they choose inner product (dot product). They do an ablation study and find that other similarity functions (like cosine similarity) perform about the same, so they just stick with simple dot products.
- When they precompute the passage encodings, they index them using [[FAISS]], offline. They say it's an extremely efficient, open-source library for similarity search and clustering of dense vectors. Given a query, they embed it and retrieve the top k passages from FAISS with embeddings closest to our query embedding.
- For the training (fine-tuning) data, ==each instance contains one question, one relevant (Positive) passage, and $n$ different, irrelevant (Negative) passages==.
	- They note that the manner in which you select negative examples is often overlooked, but could be decisive for learning a high-quality encoder. They consider a few options:
		- ==Random==: Random passages from the corpus that aren't our positives
		- ==BM25==: Top passages returned by BM25 which *don't* contain the answer, but match most question tokens. ((This seems like a "Hard" negative, perhaps?))
		- ==Gold==: Positive passages paired with other questions that appear in the training set.
	- Their best model includes gold passages from the same minim-batch, and one BM25 negative passage.
	- For the selection of *positive* passages for datasets that only provide the question and *answer*, they use the highest-ranked passage from BM25 that contains the answer as a positive passage.
- The loss function is the negative log likelihood of the positive passage.
	- ![[Pasted image 20240501155253.png|200]]
- The question-answering datasets that they used for experiments include [[Natural Questions]], TriviaQA, WebQuestions, CuratedTREC, and [[SQuAD]].






Abstract
> Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as [[TF-IDF]] or [[BM25]], are the de facto method. ==In this work, we show that retrieval ((for open-domain question answering)) can be practically implemented using dense representations alone==, where ==embeddings are learned== from a small number of questions and passages ==by a simple dual-encoder framework==. When evaluated on a wide range of open-domain QA datasets, our dense retriever ==outperforms a strong Lucene-BM25== system largely by 9%-19% absolute in terms of top-20 passage retrieval accuracy, and helps our end-to-end QA system ==establish new state-of-the-art on multiple open-domain QA benchmarks==.

Above:
- **[[Open-Domain]] [[Question Answering]]** is the task of providing answers over a wide, unrestricted range of topics (in contrast with closed-domain QA, where the system is limited to asking questions about (eg) law). Often requires/incorporates retrieval across documents to answer these questions.
	- *Who first voiced Meg on Family Guy?*
	- *Where was the 8th Dalai Lama born?*


# Paper Figures
![[Pasted image 20240501160110.png|300]]
Showing that the performance of DPR seems to beat DPR even at 1k training examples.



