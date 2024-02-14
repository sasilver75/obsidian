Link: https://cameronrwolfe.substack.com/p/the-history-of-open-source-llms-early

-----
![[Pasted image 20240213165542.png]]

Research on language modeling has a long history that dates back to GPT-1 or RNN-based techniques like ULMFit.

Despite this longish history, language models have only become popular relatively recently; the first rise in popularity came with the proposal of ==GPT-3, which showed that impressive few-shot learning performance could be achieved across many tasks==, with a combination of self-supervised pre-training and in-context learning.

After this, GPT-3 inspired the proposal of a swath of large language models (MT-NLG, Chinchilla, Gopher, more). After, ==research on language model alignment led to the creation of even more impressive models like InstructGPT and its sister ChatGPT==.

Despite being incredibly powerful, these models were all *closed source!*

> The restricted access has limited researchers' ability to understand how any why these LLMs work, hindering progress on efforts to improve their robustness, and to mitigate known issues like bias and toxicity.

The community slowly began to create open-source variants of popular language models like GPT-3. At first those models lagged behind the best proprietary models, but they still laid the foundation for improved transparency within LLM research, and catalyzed the development of subsequent models like [[LLaMA 2]] and Falcon.

This overview is part of a three-part series where we'll learn about the history of open LLMs.


# The mechanics of a language model
- Open-source LLMs research catalyzed transparency and idea sharing, creating an environment in which researchers could collaborate and innovate more quickly.

#### The Language Modeling Objective
![[Pasted image 20240213171748.png]]
- At the core of language modeling is next-token-prediction (also called the standard language modeling objective), which is used to train nearly all language models.
- To train a language model using next-token-prediction, we need a large corpus of raw text -- using this corpus, we train the model via [[Self-Supervised Learning]] (SSL) by:
	1. Sampling some text from the dataset
	2. Training the model to predict the next work (from left to right).

What is a token?
- One can roughly consider the next token prediction task to be about predicting the next word in a sequence, given a few preceding words as context... but this analogy isn't always perfect.
- When a language model receives text as input, the raw text is first tokenized (converted into a sequence of discrete words or sub-words).
- The ==tokenizer== associated with a language model typically has a fixed-size vocabulary, or a set of viable tokens that can be created from a textual sequence.
- Predicting next tokens:
	- Once a sequence of tokens can be created, the language model has an *==embedding layer==* that stores a unique and learnable vector embedding for every token within the tokenizer's vocabulary. We convert each token within the input sequence into a corresponding vector embedding.
![[Pasted image 20240213172137.png]]
- After adding positional embedding to each token, we can pass this sequence of token vectors into a decoder-only transformer, which *transforms* each of these input token vectors, and produces a corresponding *output vector* for each token.
	- Notably, the number of output vectors is the *same* as the number of input vectors! 
		- There are special "blank" characters that can be used in the output (or input) if the full context window is not used.
- Once we have a sequence of *output tokens* for each of out *input tokens*, we take the output token vectors and use them to predict the token that comes next in the sequence!
	- ((So it's important to realize that the translated word doesn't just plop out the other side of the transformer. The output vector needs to go through some linear layers and a softmax))

![[Pasted image 20240213173121.png]]
Above: In practice, the next token prediction objective is simultaneously computed over all tokens in the sequence (and over all sequences in a mini-batch) to maximize efficiency.

Due to the use of causal (or masked) attention, each output token vector only considers the current token and those that come before it, in the sequence when computing its representation.

If we were to use bidirectional self-attention, each output token vector would be computed by looking at the entire sequence of vectors, which would allow the model to cheat and solve each token prediction by just copying the token that comes next in the sequence.

As such, ==*masked self-attention is NEEDED for next-token prediction.*==

----
Aside:
- The phrase "language model" may sometimes be used to refer to models beyond those that specialize in performing next token prediction.
	- [[Bidirectional Encoder Representations from Transformers|BERT]] is considered by some to be a "language model," but it's trained using a Cloze-style objective, and is not a generative model.
		- As such, language models that specialize in next-token-prediction are oftentimes distinguished as =="causal" langauge models==.
-----

# The Transformer Architecture and its Variants![[Pasted image 20240213173526.png]]
- ~All language models use some variant of the [[Transformer]] architecture. It was initially proposed in 2017 to solve sequence-to-sequence machine translation tasks, but has since been extended to solve a variety of different problems.
- In its original form, the Transformer architecture has two components:
	- ==Encoder==
		- Each block performs bidirectional self-attention and a pointwise feed-forward transformation, which are separated with a [[Residual Connection]] and [[Layer Normalization|LayerNorm]].
	- ==Decoder==
		- Each block performs causal [[Self Attention]], [[Cross Attention]] (i.e. self-attention across encoder and decoder tokens), and a pointwise feed-forward transformation, each separated by a residual connection and LayerNorm.
			- You can think of this cross-attention as both paying attention to the previously-generated German words, as well as the original English sentence.))













