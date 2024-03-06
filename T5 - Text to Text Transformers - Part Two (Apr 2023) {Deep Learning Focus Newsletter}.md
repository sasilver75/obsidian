#article 
Link: https://cameronrwolfe.substack.com/p/t5-text-to-text-transformers-part-354

-------

![[Pasted image 20240305001135.png]]

The proposal of [[Bidirectional Encoder Representations from Transformers|BERT]] led to the popularization of transfer learning approaches for natural language processing (NLP).

Due to the widespread availability of unlabeled text on the internet, we could easily:
1. Pre-train large transformer models over large amounts of raw text
2. Fine-tune these models to accurately solve downstream tasks

This approach was incredibly effective, but its newfound popularity led many alternative methods and modifications to be proposed.

But which of these techniques actually represented *best practices* for transfer learning in nLP?

The question was answered by analysis performed with the unified [[T5|Text-to-Text Transfer Transformer]] (T5) paper and model. 
- T5 reformulates all tasks with a text-to-text format, meaning that the model receives textual input and produces textual output.
- With this unified format, T5 can analyze various different transfer learning settings, allowing many approaches to be compared.

# Preliminaries
- The proposal of BERT popularized the transfer learning paradigm for NLP. Its effectiveness led many researchers to focus on this topic and propose various modifications and improvements. 
- T5 aimed to:
	1. Convert all language tasks into a unified, text-to-text format using ==task-specific prefixes== 
	2. Study a bunch of different settings for transfer learning in NLP to deduce the techniques that work best.

## Language Modeling vs Denoising
- Initial transfer learning approaches in NLP leveraged a causal language modeling objective for pre-training.
- However denoising (also called [[Masked Language Model]]ing) objectives were subsequently shown to perform better.
	- Given a set of textual tokens to be passed as input to some model, MLM operates by:
		1. Randomly (uniformly) selecting 15% of the tokens
		2. Replacing 90% of selected tokens with a MASK token
		3. Replacing 10% of selected tokens with a random token
		4. Training the model to predict/classify each MASK token
- This percentage of tokens that are uniformly selected is called the =="corruption rate"==
- Within T5, we will see a few different variants of this denoising objective, but the basic idea remains the same.


## Benchmarks and Evaluation
- T5 attempts to derive a set of best practices for transfer learning in NLP. 
- To determine which techniques work best, T5 is evaluated on a variety of different tasks and natural language benchmarks, all solved using T5's ==text-to-text format==.
- A brief summary
	- [[GLUE]] and [[SuperGLUE]]
		- Benchmarks include many tasks, like sentence acceptability judgement, sentiment analysis, paraphrasing, sentence similarity, natural language inference (NLI), coreference resolution, sentence completion, word sense disambiguation, and question answering.
	- CNN + Daily Mail Abstractive Summarization
		- Pairs news articles with a short, summarized sequence of text that captures the main highlights of the article.
	- [[SQuAD]]: A question-answering dataset on Wikipedia articles, where the answer to each question is a segment of text from the related article.
	- Several translation datasets


# What do we learn from T5?
To do the experiments mentioned above, T5:
- Sets a baseline approach
- Varies several aspects of this baseline
	- Model architecture/size
	- Dataset
	- Pre-training objective
	- ....
- ... And changes them one-by-one to see what works best -- this mimics a "coordinate descent" strategy.

### T5 Baseline Model
- The T5 baseline architecture uses a standard, encoder-decoder transformer architecture. Both the encoder and decoder are structured similarly to BERTBase. 
- The authors found that the encoder-decoder architecture (rather than either of the "single-stack" architecutres) achieved impressive results on *both* generation and classification tasks!
	- Encoder-only models are not considered due to the fact that they're specialized for token/span prediction and don't solve generative tasks well.
	- ((I know that this wasn't trained (?) with a NTP pretraining objective... so is there still anything "cheating" about generating text using an encoder-decoder? How do you learn to generate text from a MLM/NSP pretraining objective? hmmm))
- Training Objective
	- The T5 model is pre-trained on a total of 34B tokens from the C4 corpus.
		- For comparison, BERT is trained over 137B tokens, while RoBERTa is trained over 2.2T tokens.
		- ((So it's a smaller model than BERT. This makes sense, if they're going to be training many of such models to compare))
	- Inspired by the MLM objective from BERT, T5 is pre-trained using a slightly-modified denoising objective  that:
		1. Randomly selects 15% tokens in the input sequence
		2. Replaces all consecutive spans of selected tokens with a single "sentinel" token
		3. Gives each sentinel token an ID that is unique to the current input sequence
		4. Constructs a target using all selected tokens, separated by the sentinel tokens
	- This seems complex, but it's not so bad:![[Pasted image 20240305131752.png]]
By replacing *entire spans* (eg "for inviting" above), we reduce the computational cost of pre-training, as we tend to operate over shorter input and target sequences.

Fine Tuning
- After pre-training has been performed, T5 is separately fine-tuned on each downstream task prior to being evaluated.
- Due to the Text-to-text format used by T5, both pre-training and fine-tuning use the same s
























