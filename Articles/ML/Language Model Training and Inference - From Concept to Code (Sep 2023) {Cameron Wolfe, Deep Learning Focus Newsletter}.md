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
	- Numerous different tokenization techniques exist -- [here](https://huggingface.co/docs/transformers/tokenizer_summary) are some examples.
	- The [[Byte-Pair Encoding]] (BPE) tokenizer is ==the most commonly used tokenizer for LLMs==. 
		- Another tokenization technique that's become recently popular is ==Byte-Level BPE (BBPE)==, which relies on Bytes (instead of textual characters) as the basic unit of tokenization.
- Token embeddings
	- Once we've tokenized our text, we look up the embedding for each token in an embedding layer that's stored as part of the language model's parameters... We use out learned embedding model to embed our tokens, turning our sequence of textual tokens into a sequence of ==token embedding vectors==. 
![[Pasted image 20240213131319.png]]

There's a final step required to construct the input that is actually passed to our decoder-only transformer architecture! We need to add [[Positional Embeddings]].
- Positional embeddings are the same size as our token embeddings and are treated similarly (i.e. they're stored as part of the language model and trained along with other model parameters).
- Instead of associating an embedding with each unique token, we associate an embedding with each unique position that can exist within a tokenized input.

![[Pasted image 20240213131443.png]]
Above: positional embeddings within a language model. See that we start with a raw input text sequence, turn it into a sequence of tokens using our tokenizer model, turn it into a sequence of token embeddings with our embedding model, and add positional encodings to each token embedding depending on their position.

We add these embeddings at the corresponding position; ==additive positional embeddings are necessary because the self-attention operation doesn't have a way of representing the position of each token.==
- Adding these positional embeddings allows the self-attention layers within the transformer to use the position of each token as a relevant feature during the learning process.
- Recent techniques have explored novel techniques for alternatively ==injecting positional information into self-attention itself, resulting in techniques like [[Rotary Positional Embedding|RoPE]]==.

![[Pasted image 20240213131727.png]]

### Context Windows
- Language models are pretrained with token sequences of a particular size, which is referred to as the size of the context window, or the context length. The model context length is (usually) selected based on hardware and memory constraints. It implies the limit of amount of input data that an LLM can process.
- Techniques like [[Attention with Linear Biases]] (ALiBi) have been developed to enable extrapolation to inputs longer than those seen during training.
	- ((Notice that you can't effectively just have a super long context for your model, but then train it with small-context examples. It needs to learn how to attend over that full context (as well as to smaller contexts)))

### Language Model Pretraining
![[Pasted image 20240213132521.png]]
- Language models are trained in several steps, as shown above.
- During [[Pre-training]], the first part, we get a large corpus of unlabeled text data and train the model by:
	1. sampling some text from the dataset
	2. masking some text
	3. training the model to predict the next word
- This is a [[Self-Supervised Learning]] objective, because no labels are required; the ground truth next token is already present within the corpus itself; the source of supervision is *implicit.* This specific type of SSL is referred to as *next token prediction*, and is the foundation of language modeling objectives.

### Predicting the next token
- After we've got our token embeddings with positional embeddings, we pass these vectors into a [[Decoder-Only Architecture]]'d Transformer, which produces a corresponding vector for each token embedding; see below.
![[Pasted image 20240213132935.png]]
- Given an output vector for each token, we can perform next-token-prediction, by
	1. Taking the output vector for a token
	2. Using this to predict the token that comes *next* in the sequence!
See:
![[Pasted image 20240213133024.png]]
- Above: The next token (corresponding to whatever the red input one should be) is predicted by passing the input token vector's corresponding output vector as input to a linear layer, which outputs a vector ==with the same size as our vocabulary==. This *expansion* is important.
	- Now that we have a vector with the same length as our vocabulary, a softmax is applied,  to get a probability distribution over our vocabulary as to what we believe the next token to be.
	- We can either:
		1. Sample the next token from this distribution during inference
		2. Train the model to maximize the probability probability of the correct next token during pretraining.

### Predicting tokens across a sequence
- During pretraining, we don't predict *only* a single next token; rather, we perform next-token prediction for EVERY token in a sequence, and aggregate loss over all of them.
	- ((Example)): ==So if we have a sentence of "I love to eat hotdogs" from our training data, we turn this into N prediction tasks==:
		- _
		- I _
		- I love _
		- I love to _
		- I love to eat _
	- This lets us maximally make use of the data that we have!
	- Due to this use of [[Masked Attention|Causal Attention]] (self variant), each output token vector only considers the current token and those that came before it in the sequence; so next token prediction can be performed across an entire sequence using a single forward pas of the decoder-only transformer, as each token has no knowledge of the tokens that came after it.
![[Pasted image 20240213135859.png]]
Above: Next-token prediction is used in all aspects of both training and inference for LLMs.

### Choosing the next token
- We've seen how a probability distribution over tokens is created.... but how do we actually CHOOSE the next token from this probability distribution?
	- Numerous sampling strategies exist that add slight variations by modifying the probability distribution over tokens. The exact ==decoding== approach varies depending on the application, but the main concepts to be aware of are:
		1. [[Temperature]] [link](https://twitter.com/cwolferesearch/status/1671628210180698112?s=20)
			- We typically have to set this when interacting with LLM APIs -- but what does it mean? Recall that language models don't output an actual token - they output a probability distribution over tokens in the vocabulary, typically around 100k-1M in size.
			- Higher values of temperature make output more random, while lower values of temperature make the output more deterministic. A value of 0, e.g., will output (almost) the same sequence every time. A higher temperature will increase the chance of the model generating something random, irrelevant, and (possibly) interesting.
			- Temperature is a hyperparameter in the softmax transformation, which transforms logits produced by the language model into a valid probability distribution. Higher values of temperature result in a more uniform distribution over tokens, while lower values of temperature create a distribution that assigns high probability to a single token.
		2. [[Greedy Decoding]]
			- Selects the token with the highest probability in the output distribution.
			- ![[Pasted image 20240213165251.png]]
		3. [[Top-P Sampling|Nucleus Sampling]] (Or Top-P Sampling) - [link](https://twitter.com/cwolferesearch/status/1692617211205022064?s=20)
			- Has a hyperparameter topP, which chooses from the smallest possible set of tokens whose summed probability exceeds topP during decoding. Given this set of tokens, we re-normalize the probability distribution based on each token's respective probability, and then sample!
			- Usefulness: Consider a case where a single token has a very high probability (higher than topP); In this case, nucleus sampling will always sample this token. Alternatively, assigning more uniform probability across tokens may cause a larger number of tokens to be considered during decoding. Put simply, nucleus sampling can dynamically adjust the number of tokens that are considered during decoding based on their probabilities.
			- Note: We should only use Nucleus Sampling ==OR== temperature; OpenAPI say that these parameters cannot be used in tandem -- they're different and disjoint methods of controlling the randomness of a language model's output.
			- ![[Pasted image 20240213165328.png]]
		1. [[Top-K Sampling]] - [link](https://docs.cohere.com/docs/controlling-generation-with-top-k-top-p)
			- Samples from the top-K tokens in the probability distribution that have the highest probability. This approach allows the other high-scoring tokens a chance of being picked. 
			- ![[Pasted image 20240213165241.png]]




# Creating a minimal implementation

... Skipping this section on implementing it in PyTorch ...









































