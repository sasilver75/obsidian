#article 
Link: https://cameronrwolfe.substack.com/p/t5-text-to-text-transformers-part-354

-------

![[Pasted image 20240305001135.png]]

The proposal of [[BERT|BERT]] led to the popularization of transfer learning approaches for natural language processing (NLP).

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
- Due to the Text-to-text format used by T5, both pre-training and fine-tuning use the same same maximum likelihood objective! (We formulate the correct answer as a textual sequence (during both pre-training and fine-tuning) and train the model to output the correct textual sequence.

How does the baseline perform?
- The baseline T5 model performs similarly to prior models like BERT... but these models aren't directly comparable.


# Searching for a better approach
- After testing the baseline architecture and training approach, authors in modify one aspect of this approach at the time.

#### The Architecture
- To study the impacts of architecture choice on transfer learning results, we can test different variants of the transformer architecture.
- Tested architectures include:
	- The normal [[Encoder-Decoder Architecture]]
	- The [[Decoder-Only Architecture]]
	- A prefix language model, which performs fully-visible attention over a fixed prefix within a sequence, then generates output using causal self-attention.
- The main difference between these architectures is the *type of masking used within their self-attention mechanism!*

When several architectures are tested, we see that ==the encoder-decoder transformer architecture (with a denoising objective) performs the best==, leading this architecture to be used in the remainder of experiments!
- This encoder-decoder variant has twice the parameters of the decoder-only model with P parameters.
	- ==To reduce the total number of parameters to P, we can share parameters between the encoder and decoder, which is found to perform quite well.==

#### The Pretraining Objective
- T5 is trained using three different types of pretraining objectives:
	- A BERT-style MLM objective
	- A deshuffling strategy
	- A prefix-based language modeling objective, where the text is separated into two spans, where the first span is passed as input to the encoder, and the second span is predicted by the decoder.
- ==Each of these pre-training variants tend to perform similarly!== 
	- But by selecting pre-training objectives that ==replace entire spans of corrupted tokens with single sentinel tokens== and only attempting to predict corrupt tokens within the target, we can minimize the computational cost of pre-training.
	- ==As such, the baseline strategy of masking entire spans of consecutive tokens is efficient because it produces shorter target sequences.==

Authors test different corruption rates, finding that the corruption rate doesn't significantly impact results and that a setting of 15% works just fine.

#### Data and Model Size
- The impact of scale on T5 quality is studied!
- T5 is trained with several different datasets, including one that is not filtered, a news-specific dataset, a dataset that mimics GPT-2's webtext corpus, and a few variants of the Wikipedia corpus.
- We see that:
	1. ==Not filtering the pre-training corpus is incredibly detrimental==
	2. ==Pretraining on domain-specific corpora can be helpful== in some cases. ((Duh))

> The main lesson behind these findings is that pre-training on in-domain unlabeled dataset can improve performance on downstream tasks. This is unsurprising, but also unsatisfying if our goal is to pre-train a model that can rapidly adapt to language tasks from arbitrary domains.

Going further, T5 is pre-trained using truncated versions of the C4 corpus with varying sizes.

==From these experiments, we learn that more data is (unsurprisingly) better, and that looping through smaller versions of a dataset multiple times during pre-training (unsurprisingly) causes overfitting and damage downstream performance.==

To scale up the T5 model, authors test the following modifications
- 4x more training iterations (or 4x larger batch sizes)
- 2x more training iterations and 2x larger model
- 4x larger model
- Train an ensemble of 4 encoder-decoder transformers

These results roughly correspond with what we would expect
- Increasing training time improves performance
- Combining this with a larger model yields a further benefit compared to increasing training iterations or batch size .
	- In other words, increasing the amount of pre-training data and the model size are both complementary in terms of improving model performance.

The bitter lesson of machine learning research argues that general methods that can leverage additional computation.

#### Other stuff
- T5 is also fine-tuned using different multi-task training strategies
	- These models are found to perform slightly worse than those models that are fine-tuned separately for each task.
	- However, strategies do exist to minimize the performance gap between task-specific fine-tuning and multi-task learning.
# T5: Putting it all together
- Main takeaways
	- Baseline settings
		- An encoder-decoder architecture trained using the unified text-to-text format.
		- After pre-training with a denoising objective, the model is separately fine-tuned on each downstream task before evaluation.
			- The model is fine-tuned separately for each task, as training over all tasks together yields a slightly lower performance ((!))
	- Pretraining
		- Instead of uniformly selecting tokens, the final T5 methodology performs *span corruption* with an average length of three.
		- Still, 15% of tokens are selected for corruption.
		- Performs slightly better than the baseline and yields shorter target sequence length.
	- Amount of training
		- Additional pretraining is helpful to the performance of T5.
		- Both increasing the batch size and the number of training iterations benefits T5's performance.
		- Final T5 model is trained over >1T tokens in total, which is much larger than the baseline's 34B tokens during pretraining, but still far short of [[RoBERTa]] @ 2.2T token pretraining.
		- Pretraining is performed over the generic, filtered C4 dataset, as task-specific pre-training didn't yield a consistent benefit across different tasks.
	- Model scale
		- Using larger models is helpful, but sometimes a smaller model might make sense for your situation.
		- For this reason, five different sizes of T5 models are released with anywhere from 220M to 11B parameters -- thus, T5 is actually a suite of models!
- 




























