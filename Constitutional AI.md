---
aliases:
  - CAI
---
December 15, 2022
[[Anthropic]]
Paper: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
#zotero 
Takeaway: CAI has two phases, a supervised learning phase and a reinforcement learning phase. In the supervised learning phase where we get the model "on distribution" by asking it to generate responses to a prompt, critique the prompt according to one of 16 principles, revise its response, given the prompt, origin response, and critique. This creates a dataset that we can train on in a supervised manner. In the RL phase, we hope to ameliorate the problem of asking humans for preference data. For "helpful" data, we still have to get human preference information, but for "harmless," we skip the RM-creating portion of the pipeline completely, and just use a reward model with CoT + 1/16 principle as a judge of "human" (via AI) preferences.

References:
- Notes: [[Synthetic Data - Anthropic CAI, from fine-tuning to pretraining, OpenAI's superalignment, tips, types, and open examples (Nov 2023) {Nathan Lambert, Interconnects}]]
- Notes: [[Constitutional AI with Open LLMs (February 1, 2024) {Huggingface Blogpost, multiple authors}]]

Note: Anthropic's constitution (at time of paper) is available [here](https://raw.githubusercontent.com/anthropics/ConstitutionalHarmlessnessPaper/main/prompts/CritiqueRevisionInstructions.json)
Note: I don't believe that this paper does anything besides zero-shot asking the model to critiques its responses, according to a principle(s?). In reality, the HuggingFace reimplementation article says that a good system prompt, post-processing of responses, and using few-shot prompting is required, especially when you're using small models.

----

Notes:
- Motivations for CAI include:
	1. Study simple possibilities using AI systems to supervise other AI teams, and thus *==scale supervision==*.
	2. Improve on our prior work training a harmless AI assistant by ==eliminating evasive responses==, reducing tension between helpfulness and harmlessness, and encouraging AI to explain its objections to harmful requests.
	3. Make the principles governing AI behavior more transparent.
	4. Reduce iteration time by ==obviating the need to collect new human feedback labels when altering the objective.==
- The idea with CAI is that human supervision comes *entirely from a set of principles that should govern AI behavior (along with a small number of examples for few-shot prompting)*. It has **two phases**:
	1. Supervised phase, where the model gets "on distribution"
		- **Generation -> Critique -> Revision -> Supervised Learning**
		- We first generate responses to helpfulness prompts using a *helpful-only AI assistant*. These responses are initially likely quite harmful and toxic. We then ask the same model to *critique its response according to an **independently sampled principle in the constitution***, and then ask the model to *revise* the original response in light of this critique.
		- We revise responses repeatedly in a sequence, where we randomly draw principles from the constitution at each step.
		- Once this process is complete, we finetune a pretrained language model (a different model?) with supervised learning on the final revised responses.
		- **The goal of this phase is to easily and flexibly alter the distribution of the model's responses, to reduce the need for exploration and the total length of training during the following RL phase.**
		- Note: "*Are Critiques Necessary? Why can't we just skip it, instructing the model to generate a revision directly*?" The authors do it both ways, and find that critiqued revisions achieved better harmlessness scores for small models, but ***made no noticeable difference for large models***. For the paper, they choose to use critiqued revisions as they might provide more transparency into the model's reasoning process.
	2. RL phase, where we refine and significantly improve performance.
		- **AI Comparison Evaluations -> Preference Model -> Reinforcement Learning**
		- This stage mimics RLHF except we replace human preferences for harmlessness with "AI Feedback", where the AI evaluates responses according to a set of constitutional principles.
		- In this stage, we distill LM interpretations of a set of principles back into a hybrid human/AI preference model (since **we use human labels for helpfulness, but only AI labels for harmlessness**).
		- We begin by taking the AI assistant resulting from the previous supervised learning first stage, and use it to generate a *pair of responses* to each prompt in a dataset of harmful prompts. 
		- We then formulate each prompt and pair into a multiple-choice question, where we ask which response is best, according to a constitutional principle. This produces an AI generated preference dataset for *harmlessness*, which we mix with our *human feedback helpfulness dataset.*
		- We then train a preference model on this comparison data, resulting in a preference model that can assign a score to any given sample.
		- We then finetune the supervised learning model from the first stage via RL against this reward/preference model.
		- ![[Pasted image 20240505000619.png]]
		- The authors also experiment with using [[Chain of Thought]] (both zero-shot and few-shot) prompting:
		- ![[Pasted image 20240505000736.png]]
		- One issue that arrises is that CoT samples typically state very explicitly which multiple choice option is to be preferred, and so the probability targets are typically very confident (eg close to 0 or 1) and are not well-calibrated. We found that clamping the CoT probabilities to lie in the 40-60 range led to better and more robust behavior.
- 


A lot of ***confusion*** around CAI is around the SFT, rather than the RL one that they named RLAIF and promoted more heavily in the aper/media, but ==it requires *BOTH* the instruction-finetuning and human-preference training to be called CAI!==

==CAI = Principled Instruction Correction + Principle-following RLAIF (rather than just generic RLAIF)==

Jared Kaplan refers to this whole idea as "Scaled Supervision," meaning that as AI grows in capabilities, perhaps it will be able to provide alignment feedback/supervision/oversight on more powerful models.
Question@Jared: "Why use a scalar for the reward as opposed to anything else?" -> "Interesting research questions; could imagine a bunch of functions applied to the reward; imagine punishing bad behavior more extremely than good behavior, or changing the way that you sample. We've mostly done the simplest thing, but there's interesting research to be done on variations."

Abstract
> As AI systems become more capable, we would like to enlist their help to supervise other AIs. We experiment with methods for training a harmless AI assistant through self-improvement, without any human labels identifying harmful outputs. ==The only human oversight is provided through a list of rules or principles==, and so we refer to the method as 'Constitutional AI'. The process ==involves both== a ==supervised learning phase== and a ==reinforcement learning phase==. In the ==supervised phase we sample from an initial model, then generate self-critiques and revisions, and then finetune the original model on revised responses==. In the ==RL phase, we sample from the finetuned model, use a model to evaluate which of the two samples is better, and then train a preference model from this dataset of AI preferences==. ==We then train with RL using the preference model as the reward signal==, i.e. we use 'RL from AI Feedback' (RLAIF). As a result we are able to train a harmless but non-evasive AI assistant that engages with harmful queries by explaining its objections to them. Both the SL and RL methods can leverage chain-of-thought style reasoning to improve the human-judged performance and transparency of AI decision making. These methods make it possible to control AI behavior more precisely and with far fewer human labels.


# Paper Figures
![[Pasted image 20240504185553.png]]
- Above: CAI consists of BOTH a supervised learning stage as well as an RL stage! In both, the feedback is steered by a small set of principles drawn from a "constitution." The supervised stage improves the model and gets the model into a state where it's "ready" for RL (hoping that it doesn't get stuck in strange exploration).

![[Pasted image 20240504190945.png]]
Above: A comparison of the frontiers of Standard RLHF and Constitutional AI. It seems that the CAI models score much better on Harmlessness, however that's evaluated.

![[Pasted image 20240504191534.png]]
Above: It makes sense to me that Helpfulness (which is sort of the capability to *do some action*) scales better with model size, whereas Harmlessness (oftentimes the ability to *not do some action*) doesn't scale *quite* as well with size, though it still *does* scale with size.

![[Pasted image 20240505000957.png]]




# Non-Paper Figures

![[Pasted image 20240412190738.png]]

![[Pasted image 20240801192424.png]]
This is such a good picture. It's from [here](https://huggingface.co/blog/constitutional_ai)
To make the process more concrete, here's an example of a conversation that shows how the self-critique actually looks:
![[Pasted image 20240801194003.png]]
