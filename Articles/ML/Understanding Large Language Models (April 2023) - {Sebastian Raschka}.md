---
tags:
  - article
---
Link: https://magazine.sebastianraschka.com/p/understanding-large-language-models

This article is a summary of landmark papers in transformers

---------
### (1/19) *Neural Machine Translation by Jointly Learning to Align and Translate (2014)*
Bahdanau, Cho, Bengio
[https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
- ==Introduced an attention mechanism for recurrent neural networks== (RNN) to improve long-range sequence-modeling capabilities.
- Allows RNNs to translate longer sentences more accurately.
- Motivation for developing the original [[Transformer]] architecture later.

### (2/19) *Attention is All You Need (2017)*
Vaswani, Shazeer, Parmar, Gomez, ...
[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Introduces the ==original [[Transformer]] architecture== with an [[Encoder-Decoder Architecture]]. Introduces scaled dot product attention, [[Multi-Headed Attention]] blocks, and positional input encoding.

### (3/19) *On Layer Normalization in the Transformer Architecture (2020)*
Xiong, Yang, Zheng, ..
[https://arxiv.org/abs/2002.04745](https://arxiv.org/abs/2002.04745)
- The location of [[Layer Normalization|LayerNorm]] in the Transformer architecture diagram remains a hotly debated subject -- should it be between the residual blocks, or elsewhere? There are "Post-LN Transformers" and an updated implementation defaulting to a "Pre-LN Transformer" variant.
- It's suggested by some papers that the "Pre-LN" works better. There's still ongoing discussions.


### (4/19) *Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Neural Networks (1991)*
Schmidhuber
[Paper](https://www.semanticscholar.org/paper/Learning-to-Control-Fast-Weight-Memories%3A-An-to-Schmidhuber/bc22e87a26d020215afe91c751e5bdaddd8e4922)
- An interesting paper for those interested in historical tidbits and earlier approaches  fundamentally similar to modern transformers.
- This is a proposed alternative to RNNs called Fast Weight Programmers (FWP); involves a feedforward neural network that slowly leans by gradient descent to program the changes of the fast weights of another neural network.


### (5/19) *Universal Language Model Fine-Tuning for Text Classification (2018)*
Jeremy Howard and Sebastian Ruber
[https://arxiv.org/abs/1801.06146](https://arxiv.org/abs/1801.06146)
- Written one year after the original *Attention is all you need* paper; It didn't involve transformers, but instead focuses on recurrent neural networks.
- It ==proposed pretraining language models and transfer-learning them for downstream tasks==.
- While transfer learning was already established in CV, it wasn't yet prevalent in NLP. The [[ULMFiT]] paper was among the first to demonstrate that pre-training a language model and finetuning it on a specific task could yield state-of-the-art results in many NLP tasks.
- Process:
	- Train language model on large corpus of text
	- Finetune this LM on task-specific data
	- Finetune a classifier on the task-specific data with *gradual unfreezing of layers to avoid catastrophic forgetting*.
		- This is typically not done in practice when working with *Transformer* architectures, where all layers are typically finetuned at once.

### (6/19) *[[Bidirectional Encoder Representations from Transformers|BERT]]: Pre-training of Bidirectional Transformers for Language Understanding (2018)*
Devlin, Change, Lee, Toutanova
[https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
- After the original Transformer paper, LLM research bifurcated in two directions:
	- [[Encoder-Only Architecture]] for predictive modeling tasks like text classification
		- BERT is an ==encoder-only architecture==
	- [[Decoder-Only Architecture]] for generative modeling tasks like translation, summarization, and other forms of text creation.
- The BERT paper ==introduces the original concept of masked-language modeling, and next-sentence prediction==.
- Highly recommended:
	- Follow up with [[RoBERTa]], which simplified the pre-training objectives by *removing* the next-sentence prediction tasks.

### (7/19) *Improving Language Understanding by Generative Pre-Training (2018)*
Radford and Narashiman
[Paper](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)
- The ==original GPT paper== ==introduced the popular decoder-only-style architecture and pretraining by next-token prediction== 
- While BERT was a bidirectional transformer due to its masked language model pretraining objective, GPT in contrast was a ==unidirectional, autoregressive model==.


### (08/19) *Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (2019)*
Ghazvininejad, Mohamed, Levy, Stoyanov, Zettlemoyer
[https://arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461)
- We mentioned earlier that BERT-type encoder-only LLMs are usually preferred for predictive modeling tasks, whereas GPT-type decoder-only LLMs are better at generating texts.
- To get the best of both worlds, the [[BART]] paper above combines both the encoder and decoder parts (not unlike the original transformer)

### (09/19) *Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond (2023)*
Yang, Jin ,Tang, Han ,Feng, Jiang, Yin, Hu
https://arxiv.org/abs/2304.13712
- This isn't a research paper, but is ==probably the best general architecture survey to-date, illustrating how different architectures evolved!==
![[Pasted image 20240124184220.png]]


### Scaling Laws and Improving Effficiency
- If you want to learn more about the various techniques to improve the *efficiency* of transformers, check out:
	- *2020 Efficient Transformers: A Survey* followed by *2023 A Survey on Efficient Training of Transformers*


### (10/19) *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)*
Dao, Fu, Ermon, Rudra, Ré
[https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)

While most transformer papers don't bother about 