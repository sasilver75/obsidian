---
aliases:
  - FiD
---
July 2, 2020 (2 months after the [[Retrieval-Augmented Generation (Model)|RAG]] paper, also from Meta)
[[Meta AI Research]]
Paper: [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282)
#zotero 
Takeaway: ...

See also: [[KG-FiD]]

----

Notes: 
- Related Work
	- Open-Domain Question Answering
		- Task of answering general domain questions, in which the evidence isn't given as input to the system (and retrieval must be performed). Usually, strong supervision is available to the learning systems in the form of spans corresponding to answers.
		- Various methods proposed to tackle settings where no gold spans are given to the system (and only the correct answer), or when only noisy supervision is available.
	- Passage Retrieval
		- Initially, sparse representations based on [[TF-IDF]] were used to retrieve supporting documents. Later, recent works show that retrieval based entirely on dense representations and approximate nearest neighbors were competitive with traditional approaches.
	- Generative Question Answering
		- Generate models are competitive for reading comprehension tasks like [[SQuAD]] where answers are spans. Since 2020, we've used large pretrained generative models combined with retrieval-augmented generative models.
- FiD Method
	- Retrieval
		- Authors consider two methods: [[BM25]] and [[Dense Passage Retrieval|DPR]].
			- BM25: Passages represented as bags of words, and ranking function is based on term and inverse document frequencies.
			- DPR: Passages and questions are each represented as dense vectors, computed using two BERT networks. The ranking function is the inner product between query and passage representations; retrieval is performed using [[FAISS]].
	- Reading
		- The generative model used is inspired by [[T5]] and [[BART]]. ((? Unclear))
		- The model takes as input the question as well as the support passages and generates the answer.
			- Specifically, for each retrieved passage, the title and passage are concatenated with the question (we also add special `question:`, `title:`, and `context:` tokens before corresponding sections). *These pairs are processed independently in the encoder*. The decoder attends over the *concatenation* of these retrieved passages.
			- Processing passages independently in the encoder allows us to scale to a large number of contexts, since it only performs self-attention over one context at a time, meaning the computation time of a model grows linearly with the number of passages, instead of quadratically.
				- (The decoder then processes passages jointly in the decoder, allowing it to better aggregate evidence from multiple passages.)





Abstract
> Generative models for open domain question answering have proven to be competitive, without resorting to external knowledge. While promising, this approach requires to use models with billions of parameters, which are expensive to train and query. In this paper, we investigate how much these models can benefit from retrieving text passages, potentially containing evidence. We obtain state-of-the-art results on the Natural Questions and TriviaQA open benchmarks. Interestingly, we observe that the performance of this method significantly improves when increasing the number of retrieved passages. This is evidence that generative models are good at aggregating and combining evidence from multiple passages.

# Paper Figures
![[Pasted image 20240603132259.png|200]]
Above: Just a simple figure showing RAG

![[Pasted image 20240603140135.png]]
Above: Fusion in Decoder

# Non-Paper Figures