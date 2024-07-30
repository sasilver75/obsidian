---
aliases:
  - SALMON
---
October 9, 2023
MIT-IBM Watson AI Lab, CMU, Amherst (Sun et al.)
[SALMON: Self-Alignment with Instructable Reward Models](https://arxiv.org/pdf/2310.05910)
#zotero 
Takeaway: ... Instructable reward model... synthetic data


Zhiqing Sun (lead author, now at [[OpenAI]]) said that this paper was sort of similar to the [[Rule-Based Rewards]] paper, which came out months later. Sort of [[Constitutional AI|CAI]]-like too, it looks like.

Note that there were a lot of abstract figures here too, so make sure not to skip those.

---
## Introduction
- The prevailing alignment paradigm exemplified in models like ChatGPT and LLaMA2-Chat involves employing [[Supervised Fine-Tuning|SFT]] + [[Reinforcement Learning from Human Feedback|RLHF]], but this is limited in assuming that humans can always *demonstrate* or *evaluate* tasks undertaken by AI systems (which might be doable today, but in the future advanced models will challenge human evaluation).
- We aim to develop a methodology that facilitates [[Scalable Oversight]] -- our vision to define a few general principles, akin to Asimov's three laws in robotics.
- Our goal is in-line with recent research of ==self-alignment== (eg [[Constitutional AI|CAI]]) where the primary focus is to use AI models to improve themselves, e.g. with bootstrapping over the model-generated critiques (eg [[Self-Refine]]) or self-refined outputs ([[Self-Instruct]], [[Instruction Backtranslation]]). Methods like [[Constitutional AI|CAI]] leverage feedback from automated AI systems, reducing the reliance on exhaustive human-annotated preferences... but previous RLAIF work often involves enhancing the safety of models that have already undergone RLHF training.
	- Question: Can [[Reinforcement Learning from from AI Feedback|RLAIF]] fully replace RLHF to align language models from scratch in enhancing their general alignment and capabilities?

Introducing: [[Self-Alignment with Instructable Reward Models|SALMON]]
- At the heart of the approach lies the ==instructable reward model==, which interprets/adheres to arbitrary human-written preference guidelines, and subsequently generates the rewarding scores based on those principles.
	- In previous RLAIF methods (eg [[Constitutional AI|CAI]]), the "principles" are only used to produce synthetic preferences ((i.e. data)), and the *model-generated scores* are not conditioned on any principles explicitly.
- The instructable reward model can be trained with ==synthetic data== and applied to a range of language models *without collecting model-specific human preference data.*
- Authors train a self-aligned model called `Dromedary-2` by only manually crafting 6 exemplars for in-context learning and a ==combined total of 31 principles== (17 from [[SELF-ALIGN]] and 14 from SALMON itself).
	- Despite minimal human supervision design, our model outperformed the extensively RLHF-trained [[LLaMA 2]]-Chat model (trained on 20k+ human-curated response demonstrations for SFT and 1M+ human-annotated response preferences.)

## Related Work
AI Alignment from Scratch
- Our work focuses on the problem of aligning LLMs from scratch, without the need for to rely on existing well-aligned models like ChatGPT. 
	- This differs from works where the primary focus is distilling the capabilities of well-aligned 

Self-Alignment and [[Scalable Oversight]]
- Many AI alignment methods have a heavy dependency on human-annotated data. 
- To overcome this limitation for ==self-alignment==:
	- A few notable exception involve bootstrapping by finetuning on model-generated synthetic data, like in [[Self-Instruct]].
	- [[SELF-ALIGN]] involves using 16 principles and 5 ICL exemplars to guide the AI in generating appropriate examples for an SFT model trained on 3200 seed examples.
	- [[Instruction Backtranslation]] uses web documents to create new training examples for an SFT model trained on 3,200 examples.

## Our Methodology
- ==A significant challenge in the current RLHF paradigm is the need to iteratively gather "fresh" human preferences, aimed at countering reward hacking.==
	- There's a risk that the current RL-optimized model might exploit certain vulnerabilities in the fixed reward model.
	- In a 2022 Anthropic paper, they talk about both the reward model and RLHF policies requiring weekly updates. The LLaMA 2 paper similarly documented weekly collection of human preferences over 5 iterations.
- As a result, the ==RLHF paradigm becomes highly-reliant on human annotation==, undermining its scalability for LM alignment, and limiting the utilization of pre-existing open source preference data.

Collecting Principle-Driven Synthetic Preferences
- Following [[Constitutional AI|CAI]], ==we sample two responses from an initial policy model, and the *the policy model itself* to select the preferred response based on a certain human-written principle.==
- Our method *diverges* from prior RLAIF approaches that focus on AI safety when defining principles

## Experiments


## Conclusion and Limitations


Abstract
> Supervised Fine-Tuning (==SFT==) on response demonstrations combined with Reinforcement Learning from Human Feedback (==RLHF==) constitutes a powerful paradigm for aligning LLM-based AI agents. However, a ==significant limitation of such an approach is its dependency on high-quality human annotations==, making its application to intricate tasks challenging due to ==difficulties in obtaining consistent response demonstrations and in-distribution response preferences==. This paper presents a novel approach, namely ==[[Self-Alignment with Instructable Reward Models]]==, to ==align base language models with minimal human supervision, using only a small set of human-defined principles, yet achieving superior performance==. Central to our approach is an ==instructable reward model==. ==Trained on synthetic preference data==, this model ==can generate reward scores based on arbitrary human-defined principles==. By merely adjusting these principles during the RL training phase, we gain full control over the preferences with the instructable reward model, subsequently influencing the behavior of the RL-trained policy models, and reducing the reliance on the collection of online human preferences. Applying our method to the LLaMA-2-70b base language model, we developed an AI assistant named ==Dromedary-2==. With only 6 exemplars for in-context learning and 31 human-defined principles, Dromedary-2 significantly surpasses the performance of several state-of-the-art AI systems, including LLaMA-2-Chat-70b, on various benchmark datasets. We have open-sourced the code and model weights to encourage further research into aligning LLM-based AI agents with enhanced supervision efficiency, improved controllability, and scalable oversight.


# Paper Figures

![[Pasted image 20240729183830.png|500]]
Above: Shows the number of demonstration annotations (SFT) and preference annotations (eg for DPO) for various open-source and closed source models.

![[Pasted image 20240729190038.png|550]]
Above: 
- In RLHF, we use human-labeled preferences to train a reward model, then train our LM policy with (eg) PPO against the RM.
- In CAI/RLAIF, we generate AI-labeled preferences (using prompting/exemplars/principles) and use them to train a reward model, and similarly train our LM policy against the RM.
- In [[Self-Alignment with Instructable Reward Models|SALMON]], the process is different. 

![[Pasted image 20240729191843.png|600]]
Above: ...



