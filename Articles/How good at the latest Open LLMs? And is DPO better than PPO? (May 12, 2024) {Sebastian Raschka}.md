#article 
Link: https://magazine.sebastianraschka.com/p/how-good-are-the-latest-open-llms

This is a rundown of recent LLMs by Seb

------

Spring is finally here, and four major open LLM releases:
1. Mistral's [[Mixtral]]
2. Meta's [[LLaMA 3]]
3. Microsoft's [[Phi-3]]
4. Apple's [[OpenELM]]

In addition to those models, we've got some new research on alignment-related methods, from PPO to DPO:

----

# Mixtral, LLaMA 3, and Phi-3: What's new?

### Mistral 8x22B: Larger models are better!
- The [[Mixtral]] 8x22B [[Mixture of Experts]](MoE) model by Mistral is similar to the 8x7B Mixtral model released in January 2024.
- In this model, we replace the feed-forward with 8 expert layers (and a router).
![[Pasted image 20240513005435.png|300]]
Above: A claim that shows that Mixtral performance maximizes MMLU performance while minimizing active parameter counts.

### LLaMA 3: Larger data is better!
- Meta's first LLaMA model releases in Feb 2023 were big breakthroughs for openly-available LLMs. LLaMa 2 followed in July 2023, and now we've got the third iteration.
- Meta's still training their large 400B variant of it, leaving us initially with an 8B and 70B pair of variants. Interestingly, all of these models are *dense* models.
- The main differences between LLaMA 2 and 3 is that the latter has an increased vocabulary size, context, and the fact that LLaMA 3 uses [[Grouped Query Attention]]. In addition LLaMA 3 is trained on 15T tokens, as opposed to "only" 2T tokens for LLaMA 2. This is well beyond Chinchilla optimality.
![[Pasted image 20240513005852.png]]
Above: See LLaMA added to the same chart, with both 8B and 70B variants of LLaMA 3.

- For IFT and alignment, it's interesting that the researchers seemed to use [[Proximal Policy Optimization|PPO]] AND [[Direct Preference Optimization|DPO]], as opposed to one or the other?

### PHi-3: Higher-Quality data is better!
- Just one week after LLaMA 2, Microsoft shared their new [[Phi-3]] LLM, with the Phi-3-mini 3.8B seeming to outperform the LLaMA 3 8B model (On MMLU) despite being less than half its size.
	- ((Lots of questions about the extent to which the Phi models train on contaminated data, or to how much benchmark hacking they're doing))
	- ==At the time of writing, people are still unsure whether Phi-3 is as good as promised -- many people note that Phi-3 seems much worse than LLAMA 3 for non-benchmark tasks.==
- ![[Pasted image 20240513143419.png]]
- Phi-3 is based on the LLaMa architecture, but has been trained on 5x fewer tokens than LLaMA 3 (3.3T instead of 5T). Phi-3 even uses the same tokenizer with a 32k vocabulary size as LLaMA 2, which is much smaller than the LLaMA 3 vocabulary size of 128k.

# OpenELM: An Efficient LM family with open-source training and inference framework
- [[OpenELM]] from Apple is the latest model suite aiming to provide small LLMs for deployment on mobile devices, and, like [[OLMo]], they share details discussing the architecture, training methods, and training data!
- ![[Pasted image 20240513144011.png]]
- Comes in four sizes: 270M, 450M, 1.1B, 3B, with each size also having an instruction-tuned version available with [[Rejection Sampling]] and [[Direct Preference Optimization]].
- The main architecture tweak is a *layer-wise scaling strategy*.
- ![[Pasted image 20240513144241.png]]
- ==Training Dataset==
	- They sampled a relatively small subset of 1.8T tokens from various public datasets ([[RefinedWeb]], [[RedPajama]], [[The Pile]], [[Dolma]]). This resulting subset was 2x smaller than Dolma, which was used to train OLMo.
	- Regarding the sampling rationale: "We wanted to use public datasets of about 2T tokens, following LLaMA 2"
	- ![[Pasted image 20240513144438.png]]
- ==Layer-wise Scaling==
	- The layer-wise scaling strategy, adopted from the *DeLight: Deep and Light-weight Transformer* paper, is interesting. 
		- ==Essentially, researchers gradually wide the layers from the early to the later transformer blocks.==
		- In particular, they keep the head size constant, and researchers increase the numbers of heads in the attention module.
		- They also scale the hidden dimension of the feed-forward modules, illustrated below:
		- ![[Pasted image 20240513144602.png]]
- [[Low-Rank Adaptation|LoRA]] vs [[Weight-Decomposed Low-Rank Adaptation|DoRA]]
	- Authors compared LoRA and DoRA for PEFT, and it turns out that there wasn't a noticeable difference between the two methods.
	- ![[Pasted image 20240513144835.png]]
- Conclusion
	- It's a great, transparent write-up of LLM implementation details.
	- The layer-wise scaling strategy might be something we could see more often in LLMs from now on.

# Is DPO superior to PPO for LLM alignment? A Comprehensive study
- Both [[Proximal Policy Optimization|PPO]] and [[Direct Preference Optimization|DPO]] are popular methods for aligning LLMs with human preferences, improving safety and helpfulness of LLM-generated responses.

### RLHF-PPO
- RLHF-PPO has been the backbone of OpenAI's [[InstructGPT]], and the LLMs deployed in ChatGPT -- but the landscape has shifted in recent months with the emergence of DPO-finetuned LLMS, which have made an impact on public leaderboards
	- This is largely due to DPO's reward-free alternative, which is notably easier to use -- it doesn't require training a separate reward model, and instead uses a classification-like objective to update the LLM directly.
- Today, most LLMs on top of public leaderboards have ben trained with DPO rather than PPO, but there haven't been direct head-to-head comparisons where the same model was trained with either PPO or DPO using the same dataset until the new paper came along.

### PPO is generally better than DPO
- In *Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study* (Wu et al, April 2024), they do many experiments -- the main takeaway is that ==PPO is generally better than DPO, and DPO suffers more heavily from out-of-distribution data.==
	- Here, out-of-distribution data means that the LLM has previously been trained on instruction data (using SFT) that is different from the preference data used in DPO.
	- So if an LLM has been trained on the general Alpaca dataset before then being DPO-finetuned on a *different dataset* with preference labels.
	- ==One way to improve DPO on out-of-distrubution data is to add supervised IFT round on the preference dataset before then following up with DPO finetuning... on the same dataset==
		- ((That's fucking weird))
- Best Practices
	- If you use ==[[Direct Preference Optimization|DPO]]==, make sure to perform supervised finetuning on the preference data first.
		- Iterative DPO, which involves labeling additional data with existing reward models, is better than DPO on the existing preference data.
	- If you use ==[[Proximal Policy Optimization|PPO]]==, the key success factors are large batch sizes, advantage normalization, and parameter updates via exponential moving averages.
- A good practical recommendation is that you can/should use PPO if you have ground truth reward labels (so you don't have to pretrain your own reward model), or if you can download an in-domain reward model -- otherwise, use DPO for simplicity.
	- The LLaMA 3 blog post shows that we don't have to decide whether to use PPO or DPO, but we can use both! For instance, the recipe behind LLaMA 3 has the following popline:
		- Pretraining -> SFT -> rejection Sampling -> PPO -> DPO




















