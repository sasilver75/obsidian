---
tags:
  - article
---
Link: https://kipp.ly/transformer-taxonomy/

----------

This document is a running literature review for people trying to catch up on AI; it covers:
- 22 models
- 11 architectural changes
- 7 post-pre-training techniques
- 3 training techniques
- 5 grab-bag items

Everything's loosely in the order of importance and somewhat uniqueness.
Systems/performance and alignment are excluded, to be saved for another article.


# (1/5) Models

### [[GPT-3]] (May 2020, [[OpenAI]])
175B params, 96 layers, 12288 embedding dimension, 96 heads
- ==A seminal paper for LLMs==, following both the [[GPT-2]] paper and the [[Scaling Laws]] paper. Trained on a 300B token dataset of mostly Common Crawl, along with some books, webtext, and wikipedia.
- BPE tokenizer
- Alternates dense and sparse attention layers
- Warms up to .6x10^-4 learning rate in first 375M tokens, cosine-decayed to 10% after 260B tokens.
- Batch size ramps from 32k to 3.2M tokens over the first 12B tokens.
- 4x MLP projection ratio as done in the 2017 transformer paper.
- 50k vocabulary size.
- ==Many of these characteristics form a standard recipe that has been reused by later models.==
### [[GPT-4]] (March 2023, [[OpenAI]])
Architecture Unknown - "Transformer-like"
- The technical report contains mostly evals, as well as results of their continued scaling, which are accurately extrapolated from smaller models as per Scaling Laws.
- The report also documents safety mitigation, and has a demo of their multi-model capabilities, which seem trained á la [[Flamingo]]. 
- It also has ==the best Acknowledgement section of all time==, which reads like a movie credits.
	- ((This is true, it's fucking sick))
- Interestingly, while it was released March 2023, it finished pre-training in August 2022. That's a lot of time spent on RLHF and getting everything in order.
### [[Gopher]] (December 2021, [[DeepMind]])
280B params, 260B non-embedding params, 80 layers, 16384 embedding dimension, 128 heads
- ==[[DeepMind]]'s first LLM release==.
- Uses an [[RMSNorm]] instead of a [[Layer Normalization|LayerNorm]], and uses a relative positional encoding scheme from [[Transformer-XL]] instead of an absolute positional encoding, which is why there are so many embedding parameters.
- Tokenizes with [[SentencePiece]]
- Vocabulary size 32k
- Trained on 300B tokens, with half being from MassiveText (collected for Gopher), along with books, [[CommonCrawl]], Wikipedia, news, and Github.
- Note that ==Gopher was actually trained end of 2020== and released a year later.
### [[AlphaCode]] (February 2022, [[DeepMind]])
41B params, 8 encoder layers, 56 decoder layers, 6144 embedding dimension
- A [[DeepMind]] model trained on 715GB (967B tokens) of code to do ==competitive programming==.
- The only model in this list with an [[Encoder-Decoder Architecture]]; it ==treated contest programming as a translation== task (problem statement -> solution) to gain bidirectionality.
- Uses [[Multi-Query Attention]] and generates thousands of samples at inference time, and then selects a subset of the solutions to submit.
### [[RETRO]] (February 2022, [[DeepMind]])
7B params
- *Retrieval* is a general technique, if you given a model a database to look up while doing inference.
- This was ==the inaugural retrieval paper for transformers==, using a 2T token database.
- It embeds the token-database in *chunks* using a pre-trained BERT-style model, and then performs chunked [[Cross Attention]] to nearest neighbors in the database during training and inference.
### [[GPT-3.5]] (March 2022, [[OpenAI]])
Architecture Unknown
- Delineates three models as GPT-3.5; specifically anything in the `davinci-002` or `davinci-003` family.
	- [[code-davinci-002]] is the base model, [[text-davinci-002]] is a version with FeedME non-RL [[Instruction Tuning]].
- This was turned into the ==original "ChatGPT" product==.
### [[Chinchilla]] (March 2022, [[DeepMind]])
- ==New and improved [[Scaling Laws]]==
- Trained with 1.5T tokens (similar dataset as Gopher) and same amount of compute as Gopher, yet outperforms it!
- Results in scaling laws that have parameters and ==tokens linearly increase at a 20:1 token to parameter ratio==.
### [[Flamingo]] (April 2022, [[DeepMind]])
80B params
- A [[Multimodal]] model (text and image). It only generates text, and image inputs are run through a vision encoder, and cross-attention is used to attend to those outputs.
- Uses a resampler after the vision encoder to produce a fixed (small) number of visual tokens, no matter the number of input features.
- They build on frozen Chinchilla models; the 80B params come from the cross-attention layers added to the 70B Chinchilla model.
- Google's PaLI paper followed up on image/language multimodality.
### [[Gato]] (May 2022, [[DeepMind]])
1.18B Params (That's a small model!)
- A generalist agent; sort of a follow-up to [[Flamingo]] with more modalities.
- Uses images and text, as well as button-press data formatted into tokens, as well as encodings of continuous data from robotics proprioception.
- It tries to use as little data as possible for additional tasks, which ==varied from robotics stacking tests, image captioning, and Atari==.
### Anthropic LM (Dec 2021, [[Anthropic]])
52B params, 64 layers, 892 embedding dimensions
Trained on 400B tokens, though, in a later, post-Chinchilla paper, Anthropic used a model with the same architecture trained for 850B tokens.
### [[PaLM]] (April 2022, Google)
540B parameters
- Current (as of Jan 2023) ==largest publicly-known dense language model==, but was unfortunately pre-Chinchilla (and thus overparametrized as hell, as it was trained on 780B tokens).
- PaLM activates with [[SwiGLU]], uses parallel attention, [[Multi-Query Attention]], [[Rotary Positional Embedding]], and uses the same matrices for input and output embeddings.
- No biases were used and a SentencePiece tokenizer with 256k tokens was used.
### GPT-NeoX (Feb 2022, [[Eleuther]])
20B parameters
- An Eleuther open-sourced model, trained on GPUs with DeepSpeed and Nvidia Megatron. 
- Uses the same architectural modifications as the earlier [[GPT-J]] had, and is trained on the entirety of [[The Pile]], 400B tokens.
### [[GPT-J]] (July 2021, [[Eleuther]])
6.7B parameters
- ==Notable for being a fully open-sourced model, while matching the 6.7B performance from the GPT-3 paper==.
- Trained on TPUs, and done with [[Rotary Positional Embedding]]. 
- Trained on [[The Pile]]
- Only dense attention layers are used to reduce complexity.
### [[GLaM]] (Dec 2021, Google)
1.2T parameters 
- Named "Generalist Language Model"
- A [[Mixture of Experts]] ==(MoE) model==, where parameters are sparsely activated.
- Has 64 experts per layer, with each token activating 96.6B parameters.
- Each layer has a gating unit which selects two (?) of the 64 MLPs per each token.
### LaMDA (Jan 2022, Google)
137B params, 64 layers, 892 embedding dimension, 128 heads, 2.81T dataset.
- Dialog model; dataset with a lot of dialog/forums.
- Based model is often called "LaMBDA GLM" or "GLM-137B."
	- LaMBDA itself adds a lot of dialog finetuning on top.
### [[Switch]] (June 2022, Google)
- An improvement on [[GLaM]], SwitchTransformer is a== [[Mixture of Experts]] that only routes to *one* expert==, reducing the amount of compute on inference. 
- It uses a different routing mechanism, with the main update being that routing to a single expert *works*.
### [[BLOOM]]  (July 2022, [[HuggingFace]])
176B params, 70 layers, 14336 embedding dimension, 112 heads
- The ==current largest open-source model== (as of this article, Jan 2023).
- Trained on a Hugging Face corpus called [[ROOTS]], which is 498 Hugging Face datasets
- Trained for 366B tokens, positional encodings done with [[ALiBi]]. 
- 250k vocab size BPE tokenizer, to help accommodate for multilingual data.
### Galactica (Nov 2022, [[Meta AI Research]])
120B parameters
- Galactica is a ==science model== pretrained mostly on papers, along with small amounts of code, other knowledge-based data, and a bit of [[CommonCrawl]].
- Uses a `<work>` token to encode working memory, as well as special tokens for citations.
### [[LLaMa]] (Feb 2023, [[Meta AI Research]])
65B parameters
- [[Chinchilla]] replication; fairly standard training mix of mostly CommonCrawl.
- Followed by [[LLaMA 2]], which had a permissive use license.
### Jurassic J1-Grande v2 (Dec 2022, [[AI21]])
17B parameters
- An Israeli research lab; results look good for the size
### [[OPT]] (May 2022, [[Meta AI Research]])
175B params; same architecture as GPT-3.
- Meta replication of GPT-3; Trains on [[The Pile]] and PushShift reddit, for only 180B tokens.
- Note: The Meta papers at around this time aren't connected projects; [[LLaMA]], [[OPT]], and [[Galactica]] only share *one author* out of 41.
### GLM-130B (Oct 2022, Tsinghua University)
130B params
- An open-source bilingual (Chinese and English) model. 
- Uses rotary embeddings, DeepNorm, and activates the MLP with [[GeGLU]].
- Includes prompts in pretraining; instead of the standard GPT architecture, it uses GLM for bidirectional attention.

# (2/5) Architectural Changes

### [[Multi-Query Attention]]
-  A [[Noam Shazeer]] solo paper, where the *Keys* and *Values* are shared across heads, greatly reducing the amount of memory required at inference time, improving latency and throughput.
- It's a perfectly concise 9-page paper complete with Code -- check it out!
- [[AlphaCode]] and [[PaLM]] both use this.
### [[Sparse Attention]]
- A mechanism where attention is *not* applied to all previous tokens (as it is in causal/masked attention). 
- Two styles are described:
	- 

### [[Mixture of Experts]]

### [[Flash Attention]]

### [[Parallel Attention]]

### Activation Alternatives: [[GeGLU]], [[SwiGLU]], [[SoLU]]

### [[LayerNorm]] Alternatives: [[DeepNorm]], [[RMSNorm]]

### [[RoPE]]

### [[BPE]] vs [[SentencePiece]] tokenizers

### [[ALiBi]]

# (3/5) Post-Pre-Training Techniques
- 


# (4/5) Training Techniques
- 


# (5/5) Grab Bag
- 




















