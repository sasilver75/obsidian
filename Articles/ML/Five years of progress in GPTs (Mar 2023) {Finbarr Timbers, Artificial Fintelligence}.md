---
tags:
  - article
---
Link: https://www.artfintel.com/p/five-years-of-progress-in-gpts

-------



There's a ton of prior work before large GPTs (n-gram model, [[Bidirectional Encoder Representations from Transformers]]) and after ([[RWKV]]), but we're going to constrain this to discussions about GPTs. We also aren't going to go into [[Reinforcement Learning from Human Feedback|RLHF]] or other fineetuning methods.


### [[GPT-1]] (June 2018)
- The first GPT paper is interesting to read with hindsight; it doesn't appear as if it's anything special, and doesn't follow any of the conventions that have been developed in the years since.
	- The dataset is described in GB
	- The number of parameters in the model isn't explicitly stated
	- It was basically a side-project at [[OpenAI]] and wasn't viewed as particularly important.
- Architecture (remarkably unchanged compared to GPT-3)
	- ==Decoder-only [[Transformer]]== (like the rest)
		- 12 layers, 768 embedding dimension, 12 attention heads, 3072 (4x) __
		- [[Adam]] with a warmup, and anneal to 0 using a cosine schedule.
		- Initialized weights to N(0, 0.2) using [[Byte-Pair Encoding|BPE]] with a vocab of 40k merges.
		- Activations are [[GeLU]]
		- ==Context length of 512==
		- ==~ 117M parameter model==
	- Used the [[BooksCorpus]] dataset (~1B tokens, 4GB of data) training for *100 epochs* with a batch size of 64.
		- A small dataset by modern standards.
		- ==Surprising that they trained for 100 epochs==; modern GPTs rarely *ever* see repeated data, and if they do, it's only certain datapoints a small number of times (2-4x); the entire dataset would *never* be seen 100x!

### [[GPT-2]] (February 2019)
- This is where language models started to get big; the first time that OpenAI trained a model with >1B parameters.
- Authors trained ==a *range/family* of models== here, rather than a single model
- Architecture
	- They used [[Layer Normalization]] on the inputs and add an additional [[Layer Normalization|LayerNorm]] to the output of the final self-attention block.
	- Weights are scaled by layer, by 1/sqrt(n)
	- Vocabulary of ~50k (up from ~40k)
	- Context of 1024 (up from 512)
	- Batches of 512 (up from 64)
	- Largest model is ==1.5B parameters==, about ==10x the size of GPT-1==
	- Unclear whether they used 100 epochs again; they say they followed the "same training procedure."
- The ==dataset is 10x bigger== -- going from 4GB of data consisting of publicly available books, to 40GB of text of text scraped from the internet ([[WebText]]).
- ==Nothing particularly different from GPT-1; mostly just making the model bigger.== the only other changes are the layernorm changes and the weight scaling, which don't seem to make a big difference.


### [[Scaling Laws]] paper (Kaplan et. al) (January 2020)
- This was the first scaling laws paper for LLMs
- The authors train a large number of GPT-style models to make empirical ==predictions for how model characteristics vary with scale.==
- This ==formed the basis for GPT-3, justifying scaling to 175B parameters (from 1.5B)==

![[Pasted image 20240123120552.png]]
Above: As you increase compute, data size, and the number of parameters, loss just... goes down (note Y isn't linear)

- This paper did *real science*, running a number of experiments and making predictions as to how models should scale.
- Results
	- Model performance (test loss) relies heavily on the number of parameters and the number of tokens trained on, with the ==model architecture having very little impact.==
	- If any of N (# params), D (# tokens trained on) or C (compute used for training) are held fixed, then performance rapidly hits diminishing returns.
	- ==Large models are more sample-efficient than smaller models.== (this foreshadows [[Chinchilla]], which improved on this work.)
	- It's possible to determine the optimal batch size for a model by measuring the gradient noise scale. This is just interesting because many practitioners determine batch size empirically, when it's possible to calculate this directly.


### [[GPT-3]] (June 2020)
- ==This is where the era of large language models began==, and the current AI excitement took off.
- In it, the authors train a family of 10 models, ranging from 125M to ==175B parameters (up from 1.5B, a ~115x increase)==
- For each model, the architectures are identical to GPT-2, with the exception of their use of "alternating dense and locally banded sparse attention patterns in the layers of the transformer." ([[Sparse Attention]])
	- Not clear *why* they used sparse attention; reproductions and later papers used dense attention.
	- Maybe because this paper came before [[FlashAttention]] and other faster variants of dense attention, they thought dense attention was a computational bottleneck?
- Architecture
	- ==They don't provide any details about the computational architecture; how they distributed the model==. The authors claim that it doesn't matter, but this article's author thinks it was for competitive reasons, to make the paper more difficult to reproduce.
		- In contrast, [[Nvidia]]'s [[Megatron]] was highly influential *because* it went into detail about how it made model parallelism work for their model.
- Used a dataset size of ==45TB== of text.
- Takeaways
	- ==It was an incredible advance in capability without a lot of novelty==; they just took their existing methods and scaled it up.
		- Outside of labs like OpenAI, because of the need for novelty, there are many research projects that don't get pursued because they're "only" engineering projects, or they "only" do hyperparameter tuning. You could frame this as a weakness of the more academic labs that have review policies driven by publications.

### [[Jurassic-1]] (August 2021)
- This model was developed by Israeli company [[AI21]], who specialize in NLP.
- They trained a ==178B parameter model that outperformed GPT-3 in a few categories,== ==despite only having raised $<10M== at the time.
- Paper is remarkably sparse on details, which was done for competitive reasons.
	- [[Meta AI Research]] is the only company to go into details about their experiences training a 175B parameter model.
- It uses a different architecture from GPT-3, but doesn't go into much detail. The changes aren't important; ==what we see is that there's a relatively large degree of freedom in model architectures which produce similar results.== This is corroborated by the Scaling Laws paper we discussed above.
- A consistent pattern begins to emerge:
	- Papers introduce a bunch of changes, their own dataset, and a new SOTA score, but they don't do a proper ablation, so it's tough to understand what was important and what *drove* the improvements.


### [[Megatron]]-Turing NLG (September 2019)
- A highly influential paper from [[Nvidia]] that introduced efficient ==model-parallel architectures==.
- ==If you're interviewing for an LLM job today, you're going to be expected to be familiar with it==.
- Megatron ==introduced tensor parallelism==, a variant of ==model parallelism== that splits the models to allow for intra-layer model parallelism, achieving 76% as efficient as a single GPU baseline.
- Prior to Megatron, the published SOTA for model parallelism was to use ==model pipelining== (eg GPipe), but this was difficult to do and not well supported by code.


## [[Gopher]] (December 2021)
- Gopher was an LLM trained by [[DeepMind]]. Interestingly, the lead author joined OpenAI shortly after it was published, along with a few of the coauthors.
- The architecture was the same as GPT-3, except:
	- Used [[RMSNorm]] (instead of [[Layer Normalization|LayerNorm]])
	- Used relative positional encoding scheme from [[Transformer-XL]] (from Google)
	- Use [[SentencePiece]] instead of [[BPE]]; using this seems to be an Alphabet-specific thing.
- From a computational perspective (how they trained it):
	- Used optimizer state partitioning (ZeRO)
	- Megatron-style model parallelism
	- Rematerialization/[[Gradient Checkpointing]] to save memory.
- It's interesting that while big labs don't often include details for competitive reasons, here, ==because DeepMind was (arguably) behind, they went into extensive detail==! (So it seems like this paper is useful for similar reasons to the Megatron paper)
- For this paper, DeepMind built a dataset called [[MassiveText]] (10.5TB) that is much smaller than the dataset OpenAI used for GPT-3 (45TB)
	- Deepmind only ended up training on a 300B token subset of the dataset's 2.3T tokens (13%). Compare that while the earlier GPTs used 100 epochs, Gopher saw only 10% of their tokens *once*!
- This paper *actually did ablations*
	- Adafactor vs Adam; found Adafactor was much less stable.
	- Played with lower-precision training
	- Scaled context length; showed performance increases as the context length increases (roughly sqrt(n)).


### [[Chinchilla]] (March 2022)
- An incredibly influential paper that established scaling laws. It's one of the author's favorite papers from the last few years, because it actually *does science*.
- Chinchilla trained over 400 GPT-style transformers, and fit an equation to them that was:
![[Pasted image 20240123133249.png]]
- L = loss, N = number of parameters in the LM, D = number of tokens in the dataset
- ==The implication here is that the model size and data size matter roughly equally, which is interesting, given how much attention goes into scaling up models, and how little attention is given to datasets.==
- The authors then ==use this equation to determine the optimal model size for the Gopher compute budget==; They ended up training a 4x smaller model on more tokens that had the same performance.
- This paper has been highly influential; almost all teams talk about training ==*Chinchilla optimal models*==.
- The ==standard practice before Chinchilla was to train on 300B tokens arbitrarily== (since this is what GPT-3, Gopher, and Jurrassic-1 all did)
	- Chinchilla reveals how wasteful that was; all of those papers made themselves more expensive to infer by training mmodels that were too large!

- Changes from Chinchilla:
	- [[AdamW]] instead of [[Adam]]
		- "A model trained with AdamW only passes the training performance of a model trained with Adam around 80% of the way through the cosine cycle, though the ending performance is notably better"
	- Used a modified  [[SentencePiece]] tokenizer
	- Computed the forward + backward passes in bfloat16, but store a float32 copy of the weights in the optimizer state.
		- They find that this is basically identically efficient to using float32 everywhere.


# [[PaLM]] (Google, April 2022)
- Speaking of training models that were too large, we have PaLM! PaLM was *really really big*
- It's the ==largest dense language model trained to date, at 540B parameters==
	- Used 6144 TPUs; 3 entire TPU pods, each of 2048 TPUs.
- This was ==incredibly expensive==; only Google would have the resources and infrastructure to do this.
- Unfortunately, they were training PaLM at the same time that Chinchilla was being written; ==very suboptimal==!
- Changes from GPT-3:
	- Used [[Multi-Query Attention]], which is different from [[Multi-Headed Attention]]; the key/value projections are shared for each head. This takes the same training time, but faster autoregressive decoding in inference.
	- Parallelized transformer blocks, which improved training time by 15%. 
	- [[SwiGLU]] activation functions, rather than the [[GeLU]] activations used by GPT-3s
	- Used [[Rotary Positional Embedding|RoPE]] rather than the learned embeddings of GPT-3
		- The learned embeddings that GPT-3 had are very passé, and almost no one does it now.
	- Shares the input-output embeddings
	- No bias vectors
	- [[SentencePiece]] with 256k tokens


### [[LLaMA]] (February 2023)
- ==Combined a bunch of the best features from [[PaLM]] and [[Chinchilla]]==:
	- Pre-normalize the input of each transformer sub-layer
	- Use [[RMSNorm]] instead of [[Layer Normalization|LayerNorm]]
	- [[SwiGLU]] activation function from [[PaLM]]
	- Uses [[Rotary Positional Embedding|RoPE]], as [[PaLM]] did.
	- Uses [[AdamW]], and [[Chinchilla]] did.
- Computational changes:
	- Uses efficient attention ([[FlashAttention]])
	- Uses [[Gradient Checkpointing]]
	- Interestingly, appears to use float32s everywhere (or at least, don't say otherwise). Curious why they didn't use lower precision like Chinchilla did.
- This paper was a ==shining example of how well smaller models can do when trained well!==
	- Chinchilla assesses optimality in a very narrow sense: "With a given compute budget, and ignoring inference costs, how do we choose between the number of parameters of our model and the number of tokens we train on?"
	- It can make sense to train a model that's *smaller* than Chinchilla optimal, and train it for *longer* than Chinchilla would tell us, because if we're going to deploy the model at mass scale, we care *much more* about inference cost than we do training cost!


### [[GPT-4]] (March 2023)
- This is where there would be interesting information about GPT-4, if there was any. The GPT-4 technical report contains almost no information.
	- "GPT-4 is a Transformer-style model pre-trained to predict the next token in a document, using publicly available data and data licensed from third-party providers. The model was then fine-tuned using [[Reinforcement Learning from Human Feedback|RLHF]]."
	- Given both the competitive landscape and the safety implications of large-scale models like GPT-4, the ==report contains no further information about the architecture, hardware, training compute, dataset construction, training method, or similar.==


# Conclusion
- This is March 2023; something will surely come along and invalidate all of this
	- ((As of Jan 2023, not much has! LLama-2, Phi-2, Falcon, etc have come out, but haven't unseated ))





















