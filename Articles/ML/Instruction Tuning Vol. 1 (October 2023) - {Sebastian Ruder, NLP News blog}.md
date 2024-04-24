#article 
Link: https://newsletter.ruder.io/p/instruction-tuning-vol-1?utm_source=profile&utm_medium=reader2

Followed by: [[Instruction Tuning Vol. 2 (Nov 2023) {Sebastian Ruder, NLP News}]]
- Instead of the most important IFT datasets, this covers the *latest*, and some new models!
---------

Let's cover some of the most important instruction-datasets for [[Instruction-Tuning]]!

# Instruction Tuning
- The main difference between Instruction Tuning and Supervised Fine-Tuning generally lies in the *data* that the model is trained on!
	- Supervised fine-tuning trains models on ***input examples and their corresponding outputs.***
	- Instruction-tuning *augments* input-output examples with *instructions*

![[Pasted image 20240423171449.png|450]]
Above: Supervised-finetuning versus instruction-tuning


Methods differ for how the instruction-tuning data should be constructed.

Existing instruction-tuning datasets fall into one of two categories:
1. Instructions are added to existing NLP tasks
2. Data from (a) is used to *condition a model* to *generate* new instruction input-output tuples

Let's look at some of the most popular instruction datasets

## [[Natural Instructions]] (Mishra et al, 2022)
- 193k instruction-output examples sourced from 61 existing NLP tasks.
- Crowd-sourcing instructions from each dataset are aligned to a *common schema*, so the instructions are more structured compared to other datasets.
- Outputs are relatively short, however, which makes the data less useful for generating long-form content.




























