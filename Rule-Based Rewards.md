---
aliases:
  - RBR
---
July 24, 2024
[[OpenAI]] (*Mu et al.*)
Paper: [Rule-Based Rewards for Language Model Safety](https://cdn.openai.com/rule-based-rewards-for-language-model-safety.pdf)
Blog: [Improving Model Safety Behavior with Rule-Based Rewards](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/)
#zotero 
Takeaway: ...

---

## Introduction
- Most recent alignment work has focused on using human preference data to align models, such as the line of work in [[Reinforcement Learning from Human Feedback|RLHF]].
- Human preference feedback has some challenges!
	1. It's often costly and time-consuming to collect
	2. It can become outdated as safety guidelines evolve with model capability improvements, or with changes in user behavior.
	3. Even when requirements are stable, they can be hard to effectively convey to annotators.
		- Especially in the case of safety, where desired model responses are complex and require nuance.
	- Fixing issues related to the above often requires relabeling or collecting new data, both of which are expensive and time-consuming.
- In response, methods have been developed using AI feedback, most prominently [[Constitutional AI]] (CAI), which use AI feedback to synthetically generate training data to combine with the human data for the SFT and RM training steps.
	- The constitution involves general guidelines like "*choose the response that is less harmful*", ==leaving the AI model a lot of discretion to decide what is harmful==.
	- ==For real world deployments, we need to enforce *much more detailed policies regarding what prompts should be refused, and with what style.*==
- We introduce a novel AI feedback method that allows for *==detailed human specification of desired model responses in a given situation==*, similar to specific instructions one would give to a human annotator:
	- *refusals should contain a short apology*
	- *refusals should not be judgmental toward the user*
	- *responses to self-harm conversations should contain an empathetic apology that acknowledges the user's emotional state*
- This separation into rules is similar to the human feedback method proposed in [[Sparrow]], but we focus on utilizing *AI feedback* rather than *human feedback*. 
	- We combine LLM classifiers for individual behavio

## Related Works


## Setting and Terminology


## Rule-Based Rewards for Safety


## Experiments


## Results


## Discussion and Conclusion





Abstract
> Reinforcement learning based fine-tuning of large language models (LLMs) on human preferences has been shown to enhance both their capabilities and safety behavior. However, in cases related to safety, ==without precise instructions to human annotators, the data collected may cause the model to become overly cautious, or to respond in an undesirable style==, such as being judgmental. Additionally, as model capabilities and usage patterns evolve, there may be a costly need to add or relabel data to modify safety behavior. ==We propose a novel preference modeling approach that utilizes AI feedback and only requires a small amount of human data==. Our method, ==Rule Based Rewards (RBR),== ==uses a collection of rules for desired or undesired behaviors (e.g. refusals should not be judgmental) along with a LLM grader==. ==In contrast to prior methods using AI feedback, our method uses fine-grained, composable, LLM-graded few-shot prompts as reward directly in RL training==, resulting in greater control, accuracy and ease of updating. We show that RBRs are an effective training method, achieving an F1 score of 97.1, compared to a human-feedback baseline of 91.7, resulting in much higher safety-behavior accuracy through better balancing usefulness and safety.


# Paper Figures


# Non-Paper Figures