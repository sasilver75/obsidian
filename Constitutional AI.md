---
aliases:
  - CAI
---

![[Pasted image 20240412190738.png]]

 1. Generate responses to harmful prompts
 2. LLMs critique and revise responses, creating an instruction-following dataset that's concordant with the constitution
 3. Instruction-fine-tune your model on this dataset
 4. We use an LLM as a Judge as our preference model, and create a preference dataset (?)
 5. We finetune the original model with RL using the new preference model (an LLM)


A lot of ***confusion*** around CAI is the second step above is the one that they named RLAIF and promoted more heavily in the aper/media, but ==it requires *BOTH* the instruction-finetuning and human-preference training to be called CAI!==

==CAI = Principled Instruction Correction + Principle-following RLAIF (rather than just generic RLAIF)==


