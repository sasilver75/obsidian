#article 
Link: https://cameronrwolfe.substack.com/p/the-history-of-open-source-llms-better
Part One: [[The History of Open-Source LLMs - Early Days (Part One) (Jul 2023) {Cameron Wolfe, PHD}]]

-----
![[Pasted image 20240214153829.png]]

Open-source language models are incredibly valuable, as they aim to democratize powerful and influential pieces of technology. Open-source LLMs are now commonly used and widely studied, but this area of research saw some initial struggles.
- Open-source LLMs initially performed comparatively poorly and were heavily criticized.

Given that pre-training a language model are so expensive, the models we study here are highly impactful; after they were released, people could conduct research on these pre-trained models at marginal added cost, and do cool things like pre-training.

In this section of our three-part series, we'll study the most popular open-source *base models* that are currently available. Next time, we'll talk about *fine-tuning* of the models.

# Early Days of Open-Source LLMS
- In part one of this series, we saw that the early days of research of open-source LLMs resulted in the proposal of important base models like OPT and [[BLOOM]].
- However, these models were widely considered to perform quite poorly compared to closed-source pre-trained models ([[GPT-3]]). 
	- To solve this, let's look closer at the LLM training process

LLM Training Pipeline
1. First, we pre-train the model over a lot of raw text
2. Then, we perform *alignment* with techniques like [[Supervised Fine-Tuning]] and [[Reinforcement Learning from Human Feedback]].
3. Finally, we can perform further fine-tuning or in-context learning to specialize the LLM to a particular task.

![[Pasted image 20240214161459.png]]
- *Generally*, a model's general knowledge and capabilities are learnt almost entirely during pre-training, whereas alignment teaches it which subdistribution of formats should be used when interacting with users.

So what's the solution?
- Given the poor performance of initial open-source LLMs, it became clear that we need to create higher-quality *base models* from scratch, if any forward progress was to be made. These need to be pre-trained over *much more data*, so that their performance could be improved.
	- Given that pretraining is incredibly expensive, such an effort is not trivial!
	- The creation of better open-source models had to be an undertaking of organizations with sufficient funding (eg [[Meta AI Research]] or [[MosaicML]]), who then chose to make these available to the community.

# Towards better Base Models
- Let's review several models that changed the narrative of opens-source models being too poor in performance to warrant significant usage and exploration:

### (1/4) [[LLaMA]]: A leap in open-source quality
- One of the first pre-trained LLMs to be released that was both high-performing and opens-ource.
- Was a suite of different LLMs ranging from 7B to 65B parameters.
	- These models each achieve a different tradeoff between performance and inference efficiency. 
- ==LLaMA cannot be used commercially==, only for research.
	- An impactful proposal that served to catalyze several directions of open-source research with LLMs.
- The data
	- Inspired by [[Chinchilla]]'s notes on scaling laws, LLaMA models are pretrained over a 1.4T token text corpus, significantly *larger* than any prior open-source LLM (which were often over-parametrized, and trained on 300B tokens for no particular reason, mimicking GPT-3).
	- LLaMA is pre-trained solely on publicly-available data sources, meaning it could be replicated by anyone with sufficient compute.
- Improved performance
	- A huge leap forward in performance compared to predecessors, but still lagged behind that of top proprietary LLMs (ChatGPT, GPT-4) -- but not that the ==LLaMA models still hadn't undergone alignment==!
	- Notably, LLaMA-13B performs comparablhy to GPT-3, while LLaMA-65B outputerforms PaLM in several cases, indicating that the LLaMA suite performs comparably to other widely-used base models
		- ((Is this "punching above its weight" just a consequence of it abiding by scaling laws as laid out by Chinchilla?))
- The open source explosion
	- The wake of open-source research that immediately followed LLaMA was inspiring, to include [[Alpaca]], [[Vicuna]], Koala, GPT4ALL, and more.
	- These developments included anything from fine-tuned versions of LLaMA to a c++ library for efficiently running inference with any of the LLaMA models from a laptop.

### (2/4) [[MPT]]: LLMs that are high-quality, commercial, and open-source
- Although LLaMA was impressive, none of the models within this suite could be used in commercial applications -- they were valuable solely from a research perspective.
- So [[MosaicML]] trained and released the MPT suite of models under a ==commercially-usable== Apache 2.0 license. The MPT-7B model was downloaded over 3M times before the larger MPT-30B model was made available.
![[Pasted image 20240214162827.png]]
Above: You can see that MPT-7B compared favorably against other options, including LLaMA.
- MPT-7B matches the performance of LLaMA-7B across a variety of standard benchmarks; MPT-30B tends to match the performance of GPT-3. The 30B model was slightly worse than other open-source models like LLaMA-30B and Falcon-40B.
- Fun fact:
	- A few variants of MPT were trained, including a "StoryWriter" version of MPT-7B that was created by fine-tuning on data with a 64K token context length.
	- The models were also accompanied by an entire suite of "LLM Foundry" osftware that can be used to pre-train and fine-tune MPT models.

### (3/4) [[Falcon]]: Reaching new heights in open-source performance
- Although many advances had been made in the space of open-source LLMs, available models still lagged behind proprietary LLMs in terms of performance for quite some time. The proposal of the Falcon suite was the first time that proprietary LLMs were truly rivaled by open-source alternatives.
- Two variants of Falcon (7B, 40B) are available, and they're ==commercially-licensed==.
- These Falcon models perform incredibly well due to being pre-trained on a massive, custom-curated corpus. ==Notably, the instruct variant of Falcon-40B was the top-performing model on the OpenLLM leaderboard (by a large margin) for several months.==
- Curating data from the web
	- Trained on a massive textual corpus called RefinedWeb that contains over 5T tokens of text. Only 1.5T tokens and 1T tokens of RefinedWeb are actually used for pre-training the &B and 40B models, respectively.
	- Majority of LLMs are pre-trained over public sources of curated data, but Falcon instead chose to construct their own pre-training dataset exclusively using data from the web (eg CommonCrawl). They implemented a novel pipeline that emphasizes simple, but effective components:
		- ![[Pasted image 20240214163413.png]]
	- The RefinedWeb corpus shows that a massive amount of high-quality text data (beyond the scale of datasets explored previously) can be efficiently curated from the web.
- A new SOTA
	- The only formal evaluation of the model at the time of writing is via the OpenLLM leaderboard, where it fared quite well.
	- Some practitioners have claimed that Falcon-40B seems to underperform LLaMA-based models. Useful, but anecdotal knowledge.
### (4/4) [[LLaMA 2]]: Current State-of-the-Art (as of Jul 2023)
- ![[Pasted image 20240214163554.png]]
- Although Falcon-40B was the SoTA open source LLM fora while, the recent release of LLaMA-2 dethroned it.
- LLaMA-2 is a suite of models ranging from 7B to 70B parameters, and uses only publicly available data for pre-training. 
- The model still falls short of matching the quality of proprietary models, but they come much closer than any open-source model before them.
- How is it different?
	- Adopts an approach similar to its predecessor, but with a few differences:
		- Pretrained over 40% more data (2T tokens in total, vs. 1.4T tokens)
			- ==Authors note that they up-sampled sources of data in the training corpus that were known to be more knowledgable, in an attempt to emphasize factual sources.==
		- Trained with a slightly longer context length, and the larger models use [[Grouped Query Attention]] (GQA) within their underlying architecture.

![[Pasted image 20240214165238.png]]
What is [[Grouped Query Attention|GQA]]?
- A modification to multi-headed self-attention that can improve inference efficiency.
- A typical multi-headed self-attention mechanism has N total query, key, and value heads. Creating N self-attention heads in total.
	- In GQA, we divide these N total heads into *groups*, where key and value heads are shared within each group; see above.
	- ==This is an interpolation between vanilla multi-headed self-attention and [[Multi-Query Attention]].==
	- It's found to improve inference speed comparably to multi-query attention, while maintaining the performance of vanilla multi-headed attention.

- LLaMA2 is really good
	- Compared to popular open-source models (MPT, Falcon, LLaMA), LLaMA2 performs quite well -- ==it set a new SoTA among open-source LLMs on all tasks considered.==
		- It was vaguely criticized for not being good at coding
- While LLaMa 1 was not commercially licensed, ==LLaMA 2 has a commercially-permissible license!== (But it's NOT an Apache 2.0 license and has a few caveats)



# Trends in Open Source LLMS
What led the current generation of open-source LLMs to perform so well?

#### Better Data = Better Performance
- ==The key difference between recent LLMs and those that came before is the dataset used for pre-training (Size and Quality/Composition)==
	- Models like OPT and BLOOM used 180B and 341B token models (respectively), compared with:
		- LLaMA: 1.4T Tokes
		- MPT: 1T Tokens
		- Falcon: 1-1.5T Tokens
		- LLaMA-2: 2T Tokens
	- This seems to just be scaling laws in action, and another example of the "Bitter lesson" ;) 
- Current open source LLMs increase the amount of data used for pre-training by an order of magnitude!
- Size isn't everything, though:
	- In addition to increasing the amount of data, recent models also pay close attention to thecomposition and quality of data.
		- For example, the proportion of code is increased within the datasets used for training MPT, allowing the resulting models to perform much better on coding-based tasks.
		- Additionally, Falcon-40B proposes an entirely-new pipeline for constructing high-quality corpora of text from the web, and LLaMA-2 claims to use an updated data pipeline and mix for pre-training. 

#### Optimizing for Faster Inference
- Making the decision between using an open or closed-source LLM,, practitioners consider more than just performance!
- On the other hand, a major consideration when building applications with open source LLMs is the cost of deploying the model and running inference.
	- ==Models like MPT-30B are specifically sized so that it can be hosted on a single GPU!==
![[Pasted image 20240214175519.png]]
Modified Architectures
- current open source LLMs adopt a variety of architectural tricks (shown above) to ==speed up the inference process==:
	1. Low Precision [[Layer Normalization|LayerNorm]]
	2. [[FlashAttention]]
	3. [[Multi-Query Attention]]
	4. Parallel Transformer
	5. [[Grouped Query Attention]]
- Additionally, several other architecture modifications (eg) [[Rotary Positional Embedding|RoPE]], [[ALiBi]], [[SwiGLU]] activations, and more, are adopted to improve performance.
- Current open-source LLMs apply simple modifications to the [[Decoder-Only Architecture]] to improve performance and inference speed..

# Final Thoughts
- Within this overview, we have studied the evolution of open-source LLMs from initial, lower-quality models (BLOOM, OPT) to the more recent, powerful base models (LLaMA and MPT).
	- New models primarily focused on:
		- Curating larger, higher-quality datasets for pretraining.
		- Small architectural innovations to improve efficiency in training and inference.





