---
aliases:
  - SALMON
---
October 9, 2023
MIT-IBM Watson AI Lab, CMU, Amherst (Sun et al.)
[SALMON: Self-Alignment with Instructable Reward Models](https://arxiv.org/pdf/2310.05910)
#zotero 
Takeaway: An alignment technique building on a [[SELF-ALIGN]] SFT finetune of LLaMA 2 70B (artifact: Dromedary-2), using existing open preference datasets (prompt-only) and ==instructable reward models== (reward models that also condition on a subset of ~31 different principles (or their negatives), like "Honest and Accurate: The AI must furnish reliable and factual information, and candidly disclose its limitations and the extent of its knowledge") using [[Proximal Policy Optimization|PPO]] with the usual KL0penalty. Authors also define some additional permissions at RL-time based on observed reward hacking (as an alternative to the usually strategy where we have to intermittently collect additional online preference labels and retrain the RM during RL training).


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
- Our method *diverges* from prior RLAIF approaches that focus on AI safety when defining principles: ==In addition to harmlessness principles, we also set forth principles emphasizing honesty and helpfulness of the responses.== Because of this, ==we *do not need an RLHF-trained model as the initial policy model.*== (as is the case for other RLAIF approaches).

Training Instructable Reward Models
- We train an ==instruction-following reward model== which can comprehend and assign reward scores contingent upon arbitrary human-defined principles. 
- We construct a special synthetic preference dataset, where each preference is paired with a pre-defined ==principle==.
1. For every principle, we define corresponding *==negative principles==* to increase the diversity of those principles.
	- `Positive`: The response should effectively address the task or answer the question, conveying necessary information succinctly.
	- `Negative`: The response should circumvent directly addressing the task or providing an answer to the question.
2. For each user Prompt, a *subset of principles is randomly sampled* from the established principle list, with certain principles being randomly negated.
3. The {user prompts, the model response, and sub-sampled principles} are aggregated as a single training instance for the reward model; the preference label is then calibrated by the *principle exhibiting the most pronounced difference in preference scores.*
	- (This isn't super clear from this sentence, but look at one of the last pictures in the Paper Figures section)

RL with Instructable Reward Models
- In original RLHF or RLAIF, the reward model needs to judge the response *only based on the user prompt*:
```
User: [PROMPT]
Assistant: [RESPONSE]
Reward Model: [SCORE]
```
- In [[Self-Alignment with Instructable Reward Models|SALMON]], the instructable reward model is trained to generate reward scores following human-defined judging principles, including both ==pre-defined ones== and ==RL-time *preference intervention* ones:==
```
User: [PROMPT]
Assistant: [RESPONSE]
Judging Principles: [RL-TIME INTERVENTION + PREDEFINED]
Reward Model: [SCORE]
```
- ==RL with *Predefined Principles*==
	- Training on synthetic instructable preference data enables the reward model interpret arbitrary instructions accurately.
	- Notably, we use a set of principles different from the reward model training stage.
	- During RL training, we randomly sample k=3 principles for each user prompt.
		- With respect to the idea of *prompt-dependent principle selection,* we raise the ratio of sampling the Consistent Reasoning principle for reasoning prompts, and the Ethical principle for red-teaming prompts.
- ==RL-time *Preference Intervention*==
	- In preliminary experiments, we noticed three tendencies of the policy model ==reward hacking== the reward model:
		1. Assistant often provided high-level advice instead of concrete solutions.
		2. Assistant engaging is self-praise, disrupting reward model's evaluation capabilities.
		3. Assistant tends to over-educate, such as providing analogous examples *following the solutions of math problems.*
	- ==To mitigate these reward hacking tendencies, we manually compose an additional *RL-time intervention principle* for each pattern, respectively.==
		- Conventionally, to avoid reward hacking, it necessitates the collection of *online preference data*, training a new reward model (eg at weekly intervals). We show that we can re-use the same instructable reward model, but steer its preference by defining prohibition instructions (instead of training it).

## Experiments
- Starting from LLaMA-2-70B Base, ==Dromedary-2== is SFT'd with bootstrapping data generated by an improved version of [[SELF-ALIGN]] with 6 In-Context Learning exemplars. Following this, a RL fine-tuning stage is conducted using the [[Self-Alignment with Instructable Reward Models|SALMON]] method.
- Datasets (all are "prompt datasets" coming without corresponding response demonstrations)
	- ==Self-Align==: We use a combination of 90k [[ShareGPT]], 10k prompts from [[Dolly]], 10k prompts from [[oasst1]], and 40k prompts sub-sampled from [[OpenOrca Dataset]] (which is made of prompts from T0 and FLAN).
	- ==Preference Modeling==: The synthetic principle-driven reference modeling data is collected by generating responses to the first prompts in each conversation tree of [[oasst1]] (9.8k prompts). We use existing open-source preference datasets to enable better generalization for reward models and prevent reward hacking (160k [[Helpful and Harmless|HH]], 160k [[Stanford Human Preferences]]).
	- ==RL Training==: Uses the same collection of unlabeled prompts as the Self-Align stage, with 7.5k additional math problems from [[MATH]].
- Training Details
	- Architecture of reward model is the same as base LLaMA, but embedding output is linearly projected to a scalar value to indicate reward of response.
	- Value model initialized from the reward model.
	- Authors use [[Quantized Low-Rank Adaptation|QLoRA]] for all fine-tuning processes in Self-Align and SALMON.
	- Authors use [[Proximal Policy Optimization|PPO]] with a [[Kullback-Leibler Divergence|KL-Divergence]] penalty for RL training.
- We compare mostly with non-distilled models aligned from scratch (eg [[Orca]], [[WizardLM]], [[LLaMA 2]]-Chat). It's more important to consider the LLaMA 2 comparison, because that's the best open-source model aligned from scratch (whereas Orca/Wizard I believe used a stronger LM to generate the responses.)
- Authors evaluate using Vicuna-Bench and [[MT-Bench]] for chat capabilities and [[BIG-Bench Hard]], [[HumanEval]], TydiQA for capability evaluation, and [[TruthfulQA]] for truthfulness. Across all of these, Dromedary-2 achieves new SoTA scores among open, non-distilled models (ie beating LLaMA-2-Chat-70B handily).


## Conclusion and Limitations
- SALMON: The use of instructable reward models trained to effectively and flexibly align language models with human values and intentions.
	- By merely adjusting the principles that the reward model follows, we can gain control over the preferences of the reward model, which influence the behavior of the RL-trained policy model.
	- Combined with the [[SELF-ALIGN]] technique, we build a Dromedary-2 with only 6 exemplars for In-Context-Learning, and 31 human-defined principles. This surpasses performance of several SoTA RLHF-trained AI systems.
- Limitations
	- Reliability concerns: Still hallucinates unverified information and has reasoning errors.
	- ==Principle design challenges==: Crafting robust and encompassing principles for SALMON is intricate, due to the wide range of scenarios the model might encounter during RL. Our approach is presented as a starting platform to foster community discourse.
	- ==Context-Dependent principle selection==: We use randomly-sampled principles to instruct the reward model for general prompts... but the effectiveness of these principles can be problem-dependent. Some tasks might benefit from specialized principles tailored to address the specific challenges posed by those tasks. We can likely better do adaptive principle selection.
	- ==Intrinsic Knowledge Limitations==: Leverages intrinsic knowledge of an LLM, which binds us to the model's inherent limitations. Integrating techniques from [[Retrieval-Augmented Generation|RAG]] could potentially mitigate some of these knowledge limitations.

## Appendix
- 


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

![[Pasted image 20240730135244.png|600]]
Above: This just seem to be examples of annoying-to-humans behavior that result from usual RL training (eg too many step-by-step instructions, too many examples, too judgy)... in Salmon, we can notice this annoying characteristics of responses and turn them into principles.

![[Pasted image 20240730144144.png|500]]
See Dromedary beating LLaMa-2-Chat-70b (on [[MT-Bench]] and Vicuna Bench), which is the chat version of the base model that Dromedary was trained from. At the time, this is the SoTA for *non-distilled* open-source models.

![[Pasted image 20240730144338.png|500]]
Above: Capabilities evaluated on BBH, HumanEval, and TydiQA (multilingual). See that again it wipes the floor with LLaMA-2-Chat-70B.


![[Pasted image 20240730145655.png|500]]

![[Pasted image 20240730150416.png|500]]
Explanation for how the winner is chosen in a scenario where there are are multiple principles selected. Basically, in a Response A (2,3,6) and Response B (1,5,5) scenario, because Response B wins by the maximum difference of 2 for the second principle, Response B is selected as the winner (even though Response A is better in 2/3 of the principles).

![[Pasted image 20240730150909.png|500]]
Full list of principles used in synthetic preference modeling.
((Wait, above it said "31 human-defined principles"))

![[Pasted image 20240730151005.png|500]]

![[Pasted image 20240730151018.png|500]]

![[Pasted image 20240730151032.png|500]]

![[Pasted image 20240730151043.png|500]]

---

![[Pasted image 20240730151209.png|500]]
![[Pasted image 20240730151224.png|500]]
![[Pasted image 20240730151238.png|500]]
![[Pasted image 20240730151252.png|500]]
Above: Their improved prompt from [[SELF-ALIGN]].



