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

-------
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
- It embeds the token-database in *chunks* using a pre-trained BERT-style model, and then performs chunked [[Cross-Attention]] to nearest neighbors in the database during training and inference.
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
### [[GPT-NeoX]] (Feb 2022, [[Eleuther]])
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
- Trained for 366B tokens, positional encodings done with [[Attention with Linear Biases]]. 
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

--------
# (2/5) Architectural Changes

### [[Multi-Query Attention]]
-  A [[Noam Shazeer]] solo paper, where the *==Keys* and *Values* are shared across heads==, greatly reducing the amount of memory required at inference time, ==improving latency and throughput==.
- It's a perfectly concise 9-page paper complete with Code -- check it out!
- [[AlphaCode]] and [[PaLM]] both use this.
### [[Sparse Attention]]
- A mechanism where ==attention is *not* applied to all previous tokens== (as it is in causal/masked attention). 
- Two styles are described:
	- ==Strided==, where it looks at the last N tokens
	- ==Fixed==, where *sections* of tokens in the sequence are attended to.
- Relevance: In the GPT-3 paper, the model is described to have alternative dense and "locally banded" sparse layers.
### [[Mixture of Experts]]
- Technique where ==multiple expert networks are used to divide a problem space into subspaces==, with each expert network specializing in a specific subspace, and ==each input being routed to appropriate expert(s).== Lets models be pretrained with less compute, ==enabling the scaling up of the model or dataset size==.
- Differs from ensemble techniques, in which all models are run on every input; Only 1-to-a-few models are run for each input in MoE.
### [[FlashAttention]]
- An architectural change to== do attention with less memory access== (which is most of the cost). 
- It tiles and incrementally performs the softmax reduction and avoids storing the whole intermediate attention matrix for the backwards pass.
- ==Cites a 1.7x training speedup compared to megatron, and up to over 4x on inference.==
### Encoder+Decoder
- A la original transformer paper, the encoder-decoder architecture was originally made for translation tasks.
- Where the classic GPT architecture uses alternating attention and MLP blocks, the original transformer had an encoder block which was attention -> MLP, and a decoder block which was masked attention -> encoder-decoder attention -> MLP.
### [[Parallel Attention]]
- PaLM uses parallel attention (poorly named), where ==the model is trained with the attention and MLP layers running in parallel, taking the same vectors==.
- This makes it so that you can do your attention and feed-forward matmuls together to increase arithmetic intensity for better performance. GPT-J also uses it.
### Activation Alternatives: GeLU, GeGLU, SwiGLU, SoLU
- The original paper uses [[Rectified Linear Unit|ReLU]] to activate the MLP block, which is a simple `x if x > 0 else 0` in between two linear transformations (matmuls).
- [[GeLU]] is ==similar to ReLU but smooths it out a bit==.
- [[SoLU]] (softmax) introduced in an anthropic paper and is simply `x*softmax(x)`, and ==is used to improve the interpretability of models==.
- [[SwiGLU]] is the most sophisticated of these, and is a [[Noam Shazeer]] solo paper. Builds upon [[Gated Linear Unit]] (GLU), which was meant to be ==more stable than ReLU==, and does the "swish" operation before the GLU. It ==softens out the ReLU and allows some values to be under zero==.
	- Is quoted in the paper as coming through "==divine benevolence=="
### [[Layer Normalization|LayerNorm]] Alternatives: [[DeepNorm]], [[RMSNorm]]
- LLMs norm twice per block (once for attention, and once to feed-forward), which does some normalization functions to imrpve training.
- DeepNorm and RMSNorm are alternatives.
- RMSNorm is simply the square root of the mean of teh values
### [[Rotary Positional Embedding|RoPE]]: Rotary Position Embedding
- A way of finding a ==positional encoding function== for transformer architectures.
- It ==improves performance in NLP tasks by more effectively leveraging positional information in sequences==. Combines the strengths of both absolute and relative positional embeddings.
	- Good at handling sequences of different lengths
	- Decays inter-token dependency with increasing relative distances
- Better than sinusoidal positional embedding techniques
### [[BPE]] vs [[SentencePiece]] tokenizers
- [[Byte-Pair Encoding]]s are the default for most language models, and were used by the original GPT paper, GPT-3, and presumably GPT-3.5.
- Note: An obvious reason to *not* use plain BPE (and instead use SentencePiece_ is if your distribution *doesn't contain* space-separated words (eg AlphaCode, GLM (chinese), PaLM (multilingual))
	- In other words, ==if your distribution of text isn't always cleanly space-separated (code, non-english characters), use SentencePiece instead of BPE.==
### [[Attention with Linear Biases]]
- [[Attention with Linear Biases|Attention with Linear Biases]] is a ==long-context positional embedding scheme to support extrapolation to longer lengths==, by biasing (linearly) the Query and Key scores according to their distance.

-------
# (3/5) Post-Pre-Training Techniques

### [[Reinforcement Learning from Human Feedback|RLHF]] with [[Proximal Policy Optimization|PPO]]
- In RLHF, a reward model is trained, where the labeler evaluates an array of model generations. Then PPO is used for the RL, where the policy generates an output evaluated by the reward model to then improve on the policy.
### [[Constitutional AI]]
- ==Basically a form of [[Reinforcement Learning from Human Feedback with AI Feedback|RLAIF]]==, though actually called [[Constitutional AI|CAI]]
- It has a supervised learning phase where a helpful-only AI is used to generate adversarial prompts.
- The assistant then iterates on its own response based on the provided constitutions (a set of short values for the model to follow)
	- This is like RLHF with PPO, except substituting AI feedback
- Then finetuning is done on the responses.
### Minerva
- From the Google Blueshift team, [[Minerva]] is a ==finetuned model on math and science data==.
- It's a finetuned model from [[PaLM]], with datasets from ArXiV and some websites that were carefully preprocessed to preserve mathematical formatting.
### Codex
- Launched in July 2021, OpenAI's [[Codex]] is a ==GPT-finetune on 100B tokens of code== (in this case, publicly available Github code).
- The paper also debuted [[HumanEval]], a human written code evaluation benchmark.
- ==The paper most notably demonstrates that code data is really important for code performance==, as GPT-J was outperforming 3 at code. 

### Just Finetune on CoTed Outputs
- Writer forgot which paper did this, but they ==finetuned their model on [[Chain of Thought]] outputs from the same model, and it did better==. This is an expected, but notable result.

### FeedME (SFT)
- Described in the [[InstructGPT]] paper (though perhaps not the origin), [[Supervised Fine-Tuning]] uses human-generated content which is then used to fine-tune the pre-trained model.
- This paper finds that SFT performs better than the base pre-trained models, but [[Reinforcement Learning from Human Feedback|RLHF]] performs better than SFT.
### [[FLAN]]
- An instruction-tuned model (finetuned on instruction-formatted NLP tasks) that results in improved zero-shot performance

------
# (4/5) Training Techniques

### Being good at setting hyperparameters
- There's no one paper for this, but it's obvious that getting the hyperparameters right is pretty important.
- Some baseline of this is available by reading papers:
	- [[Chinchilla]] paper
	- [[Scaling Laws]] paper

### Pre-training with Human Feedback
- Pre-training tends to have a very unsupervised format, though [[Pretraining with Human Feedback]] (PHF, Feb 2023) applies ==a simple technique to label data at pretraining-time==, to align with human preferences.
- It uses two tokens (a good and bad one) prepended to samples at training, and then samples with them at inference.
	- They tried other objectives (eg filtering out bad data) that all performed worse.

### [[MUP]]
- A method of parametrization that ==makes hyperparameters (ones related to learning rates and optimizers) predictable== (consistent) across model sizes.
- This not only saves the parameter sweep compute, but should be closer to optimal.


---------
# (5/5) Grab Bag

### [[Chain of Thought]] (CoT)
- ==Technique that makes the model "think step by step" and yield better results.==
- The phrase is now used to describe techniques that aren't just prompting.

### Tool Use
- A good canonical tool use paper is probably the Dec 2021 [[WebGPT]] paper, in which capabilities are greatly enhanced by giving GPT-3 access to the web.
- Deepmind has trained some RL tool use agents, and Meta has [[Toolformer]].
- ((A recent paper on this is the [[Gorilla]] paper))

### Fill in the Middle
- This July 2022 paper describes a simple data transformation that ==moves a substring from the middle of a text to the end, and asks the model to fill in the middle.==
- Allows the model to gain a capability that's really useful for tasks like code completion without damage to performance on strictly left-to-right tasks

### Sampling Techniques: Top-K, Top-P, Beam Search
- The output of language models is fundamentally logits for every possible token, which are then softmaxed into becoming probabilities.
	- The most naive way of turning your logits into tokens is to take the most likely token.
- When there are *temperature* controls with langauge models, it's dividing the logits by the temperature, which makes the model more/less confident in its top choice.
- [[Top-K Sampling]] takes the top K tokens and samples from that distsribution
- [[Top-P Sampling]] (Nucleus Sampling), uses the top P percentage (think CDFs) of tokens and samples from there.

### [[Tail-Free Sampling]]
- Tail Free Sampling takes the derivative of Top-P sampling, and is named as such to find the "tail", as Top-P sampling could fail in cases of being cut off at a point where many tokens have similar probabilities.



















