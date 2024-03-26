#article 
Link: https://cameronrwolfe.substack.com/p/the-history-of-open-source-llms-early
Part Two: [[The History of Open-Source LLMs - Better Base Models (Part Two) (Jul 2023) {Cameron Wolfe, Deep Learning Focus Newsletter}]]

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
		- Each block performs causal [[Self-Attention]], [[Cross-Attention]] (i.e. self-attention across encoder and decoder tokens), and a pointwise feed-forward transformation, each separated by a residual connection and LayerNorm.
			- ((You can think of this cross-attention as both paying attention to the previously-generated German words, as well as the original English sentence.))
- When both components of the architecture are present:
	1. The encoder processes the input sequence and produces an output sequence.
	2. The decoder generates its own output sequence, given the encoder's output sequence as input.
- In other words, the encoder processes the *entire input sequence* to form a *representation* that the decoder uses as a context when generating output.


#### [[Decoder-Only Architecture]] and [[Encoder-Only Architecture]] Transformers
- Nearly all causal language models use a decoder-only Transformer as their underlying architecture, which is just a normal Transformer with the encoder portion of the architecture removed.
	- Additionally, the cross attention portion of each decoder block is removed due to the lack of an encoder (we can't attend to an encoder that doesn't exist).
- Alternatively, one could form an "encoder-only" architecture by just using the encoder portion of the architecture.
	- Encoder-only architectures like [[Bidirectional Encoder Representations from Transformers|BERT]] excel at solving a variety of discriminative natural language tasks, but aren't used for generating text. See more [here](https://cameronrwolfe.substack.com/p/language-understanding-with-bert). 

Why the decoder?
- The choice of using the decoder-only architecture (as opposed to encoder-only or the full encoder-decoder transformer) for LLMs is not arbitrary -- it's driven by the use of next-token prediction for training language models.
- ==The use of masked self-attention within the decoder ensures that the model cannot look forward in the sequence when predicting the next token -- otherwise, the next-token prediction would be trivial, as the model could simply copy the next token.==
![[Pasted image 20240213174642.png]]
- To perform next token prediction *without cheating*, both encoder-only and encoder-decoder transformers would have to avoid including any ground truth next token in their input sequence.
	- To do this, we could:
		1. Ingest a prefix
		2. Predict the token that follows the prefix
	- But this is a bit inefficient because we can only predict a single next token at a time.
	- ==In contrast==, decoder-only models, due to their use of masked self-attention, can ingest an entire sequence of tokens and apply a language modeling objective to every token within the sequence.

### How do we generate text?
- Given the decoder-only architecture outlined above, generating text follows a simple autoregressive process. We just continually predict the next token, add this token to our input, and repeat; see below.
![[Pasted image 20240213180821.png]]

### Training and Using Language Models
- To complete our understanding of language models, we need to quickly explore how these models are typically trained and used in practice.

Most are trained using a few standard techniques, illustrated below:
![[Pasted image 20240213181531.png]]
- We'll focus on [[Pre-training]], [[Alignment]], and [[In-Context Learning]].

Pre-Training
- The initial and most computationally expensive step.
- Begin with randomly initialized LLM, and train the model using a language modeling objective over a massive corpus of raw text. Scale in terms of both data and model size.

What else do we need?
- There's a reason that LLMs didn't explode is popularity until the proposal of models like ChatGPT -- ==just being able to predict the next token isn't very interesting==! It doesn't intrinsically make for a useful assistant -- instead, it often produces output that is repetitive, simple, and generally not helpful.
- [[Alignment]] refers to the process of fine-tuning an LLM to better align with the desires of human users. This is primarily done via two techniques, at the time of this article:
	- [[Supervised Fine-Tuning]] (SFT) 
	- [[RLHF]]

Using LLMs in practice
- After we've pre-trained and fine-tuned (or aligned) our language model, teh final step is to specialize the model to our desired application! This process may require extra fine-tuning over domain-specific data. More training is not always necessary, as we can accomplish a lot by using just in-context learning.

![[Pasted image 20240213182646.png]]
Above: During unsupervised pre-training, the model develops broad skills and pattern-recognition abilities. It can use those during inference time to adapt to/recognize the task as stated in the prompt. We call this in-context learning, which occurs *within the forward pass* on a sequence!

In-context learning refers to the idea of a single model being able to solve a variety of different problems by cleverly varying the textual prompt.
![[Pasted image 20240214145930.png]]

# Initial attempts at open-source LLMs
- Given how expensive LLM pretraining is, it took some time for the research community to pursue the creation of an open-source LLM on the scale of GPT-3.

#### GPT-NeoX-20B
- ==One of the first open-source LLM created by [[Eleuther]].==
- Created after the initial GPT-Neo model (2.7B) was trained over [[The Pile]].
- Achieves impressive few-shot learning performance on a variety of natural language benchmarks.
- Although this model was somewhat small compared to GPT-3 (20B vs 175B), it was the largest model we had at the time, and all of its training and evaluation code was released alongside its weights under ==Apache 2.0 license==, permitting commercial use!
- The model:
	- Uses the standard decoder-only transformer architecture, except:
	- [[Rotary Positional Embedding|RoPE]] embeddings
		- Improving on the standard position embeddings; provides a new methodology from *injecting positional information of embedding vectors into the self-attention operation itself!* 
		- Achieves a balance between absolute and relative position information, and used in a variety of other models ([[PaLM]], [[Falcon]])
	- Parallel Attention and Feed-Forward layers
		- 15% improvement in training throughput
![[Pasted image 20240214150409.png]]
==Above==: Example of performing attention and feed-forward layers in parallel
- Interestingly, they used a custom tokenizer trained from scratch on The Pile (a large corpus of diverse text, including code). As a result, the resulting tokenizer is effective at also tokenizing code! As a result, several open-source models like [[MPT]] adopted this tokenizer, even today.
- Performance:
	- Compared to both GPT-3 and other open-source models like GPT-J; in the evaluations, we see that GPT-NeoX-20B performs quite well.

#### Open Pre-Trained Transformers (OPT) Langauge Models
- OPT was released by [[Meta AI Research]] and was created as an initiative to democratize access of powerful LLMs to the public, and is comprised of several LLMs ranging in size from 12155M to 175B parameters.
- Trained over a curated dataset compiled from sources like Reddit, [[The Pile]], [[BooksCorpus]].
- The largest model in the suite, OPT-175B, was one of the first truly *large* language models to be open sourced.
- The models were accompanied by a code repo and even a logbook that details the pre-training process of all models.
- Although OPT models are not commercially-usable by license, they're an incredible resource that influence the open availability of LLMs for research.
- Impact:
	- ==First large-scale effort to make massive LMs available to the research community.==
	- OPT's open-source training code makes a highly-efficient training framework, using common techniques like FSDP and tensor parallelism. This achieves resource utilization that's 17% better than research published directly by NVIDIA, making a great resource for training LLMs.

- The logbook helps us understand the full cost of training an LLM and the many struggles that might occur in the process (loss spikes, hardware failures, other "midflight" training adjustments). This became a topic of conversations.
- Performance
	- OPT-175B Found to achieve comparable performance to GPT-3 in zero and few-shot learning settings. Overall, ==not notable performance== -- considered to lag behind proprietary models in terms of quality.  Still a big step== forward for boosting interest== in Open source LLMs.

#### [[BLOOM]], an Open, Multilingual Language Model
- BLOOM is a 176B LLM trained as part of a massive open collaboration of AI researchers called the Big Science Research Workshop. 
- Running over the timespan of one year, (May 2021-May 2022), the goal was to:
	- Create a massive multilingual text dataset
	- Create are large multilingual language model that's trained on this dataset
- ==The resulting model is slightly larger than GPT-3's 175B, and can generate text in 46 different languages and 13 programming languages.==
- The dataset developed for training BLOOM (the ROOTS corpus) is comprised of 498 HuggingFace datasets and contains over 11.6B of text spanning 46 languages and 13 programming languages.
![[Pasted image 20240214152136.png]]

After obtaining the raw data, the authors appleid a pipeline of different quality filters to remove text that isn't natural language.
- The exact filtering components that are used change depending on the source of the data. The pipeline has the common goal of filtering out as much "low-quality text" as possible.

Architecture
- A standard decoder-only transformer, with some modifications:
- [[ALiBi]], attention with linear biases, which aids the model in generalizing to longer context lengths than those seen during training.
- Embedding Layer Norm: An extra [[Layer Normalization|LayerNorm]] is placed after the model's embedding layer, which is empirically found to improve training 
- Overall, not much different than most LLMs! The authors actually perform an extensive analysis between different types of transformer architectures (eg [[Encoder-Only Architecture]], [[Encoder-Decoder Architecture]], [[Decoder-Only Architecture]]), finding that ==the decoder-only model (used by nearly all causal language models) achieves the best performance after pre-training.==
- Compared to other models, BLOOM performs relatively well. It achieves comparable or improved results relative ot OPT in natural language benchmarks, and excels in marchine translation tasks, given its multilingual training corpus. Still falls below that of proprietary models ([[Codex]], [[Chinchilla]], [[PaLM]]). ==Research in open-source LLMs was still lagging, at the time that BLOOM was proposed.==


### Other notable models
- [[GPT-J]]
	- A 6B parameter, english-only causal language model that was proposed prior to GPT-NeoX-20B; This was similarly pretrained on [[The Pile]]. At the time of its release, it was the largest publicly-available [[GPT-3]] style language model.
- GLM
	- More of a pre-training objective than a language model.
	- Explores the idea of unifying different pre-training techniques (eg from [[Bidirectional Encoder Representations from Transformers|BERT]], [[T5]], [[GPT]]) by proposing an autoregressive blank-infilling objective.
		- In other words, we predict masked words in a sentence in an autoregressive manner, similar to a language model. The resulting model is quite small (<1B params), but is found to outperform BERT, T5, GPT on several popular NLP benchmarks.

![[Pasted image 20240214153453.png]]
# What's next?
- Given the initial attempts at open-source LLMs yield models that did not perform nearly as well as proprietary counterparts, we might wonder: *What should we do to make these models better?*
- There are two areas of exploration (at least):
	1. Creating better base LLMs
	2. Fine-tuning open-source LLMs (i.e. alignment and imitation)

Research here has progressed at a shocking pace: We went from OPT to near-SoTA models (eg [[LLaMA 2]], [[Falcon]]) in less than a year!

In the next two parts of the survey, we'll look into each of these avenues of research.

















