---
aliases:
  - CAI
---
December 15, 2022 -- [[Anthropic]]
Paper: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

Good articles:
- [[Synthetic Data - Anthropic CAI, from fine-tuning to pretraining, OpenAI's superalignment, tips, types, and open examples (Nov 2023) {Nathan Lambert, Interconnects}]]

Paper Abstract:
> As AI systems become more capable, we would like to enlist their help to supervise other AIs. We experiment with methods for training a harmless AI assistant through self-improvement, without any human labels identifying harmful outputs. ==The only human oversight is provided through a list of rules or principles, and so we refer to the method as 'Constitutional AI'.== The process involves ==both== a ==supervised learning== and a ==reinforcement learning phase==. In the supervised phase we sample from an initial model, then ==generate self-critiques and revisions, and then finetune the original model on revised responses==. In the RL phase, we sample from the finetuned model, ==use a model to evaluate which of the two samples is better, and then train a preference model from this dataset of AI preferences==. ==We then train with RL using the preference model as the reward signal==, i.e. we use 'RL from AI Feedback' (RLAIF). As a result we are able to train a harmless but non-evasive AI assistant that engages with harmful queries by explaining its objections to them. Both the SL and RL methods can leverage chain-of-thought style reasoning to improve the human-judged performance and transparency of AI decision making. These methods make it possible to control AI behavior more precisely and with far fewer human labels.


![[Pasted image 20240412190738.png]]

 1. Generate responses to harmful prompts
 2. LLMs critique and revise responses, creating an instruction-following dataset that's concordant with the constitution
 3. Instruction-fine-tune your model on this dataset
 4. We use an LLM as a Judge as our preference model, and create a preference dataset (?)
 5. We finetune the original model with RL using the new preference model (an LLM)


A lot of ***confusion*** around CAI is the second step above is the one that they named RLAIF and promoted more heavily in the aper/media, but ==it requires *BOTH* the instruction-finetuning and human-preference training to be called CAI!==

==CAI = Principled Instruction Correction + Principle-following RLAIF (rather than just generic RLAIF)==


