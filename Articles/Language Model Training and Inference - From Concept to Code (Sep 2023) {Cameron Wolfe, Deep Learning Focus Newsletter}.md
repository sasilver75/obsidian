#article 
Link: https://cameronrwolfe.substack.com/p/language-model-training-and-inference

-------
![[Pasted image 20240212191235.png]]

Despite all that's been accomplished with LLMs, the underlying concept that powers these models is quite simple -- just accurately predict the next token.
- Some many (reasonably) argue that recent research on LLMs goes beyond this basic idea, but the core concept still underlies the pretraining, finetuning, and inference processes of all causal language model.

Let's dive more into how this objective works, at conceptual and concrete levels.
# Relevant Background concepts
- Transformer architecture
	- We've covered the [[Transformer]] architecture in other blog posts ([here](https://cameronrwolfe.substack.com/i/136366740/the-transformer-from-top-to-bottom) and [here](https://cameronrwolfe.substack.com/i/85568430/decoder-only-transformers)).
	- It helps to understand the idea of self-attention and the role that it plays in the transformer architecture. The large models that we will study in this overview use a particular variant called [[Multi-Head Attention]] (specifically the causal, self-attention version of it).
- Training neural nets with PyTorch
	- We're going to write some PyTorch that relies on distributed training techniques like ==Distributed Data Parallel (DDP) training==.  (Shares some links on more on these)
	- Beyond basic NNs in Pytorch, we'll also see ==automatic mixed precision (AMP)== training used, which selectively adjusts the precision between *full precision* (32 bits, float32) and *half precision* (16 bits, float16).
- Deep learning basics
	- Requires a baseline understanding of NNs; how they're trained and used.

# Understanding Next Token Prediction
- Let's now learn about next-token prediction (known as the standard language modeling objective); the workhorse behind all causal language models.

### Tokens and Vocabularies
- The first question we might ask is: What is a [[Token]]? 
	- It's a word or sub-world within a sequence of text.
	- Given a sequence of raw text as an input, one of first steps we take in using a language model is to tokenize this raw text, breaking it into a sequence of discrete tokens.

![[Pasted image 20240212191902.png]]
Above: See that tokens don't always correspond to words, or even "obvious" subwords. There are a variety of tokenizers available, with different trade-offs. Many are biased to the English language.

- To perform this tokenization, we rely on a [[Tokenizer]]! 
	- The tokenizer is trained over al unlabeled textual corpus to learn a fixed-size, unique set of tokens that exist. 
	- This fixed-size set of tokens is referred to as our ==Vocabulary==, and the vocabulary contains all tokens that are known by the language model.
	- Usually, we should try to make sure that the data used to train the tokenizer accurately reflects the kind of data that our model will see during training and inference.
- Tokenization Techniques





































