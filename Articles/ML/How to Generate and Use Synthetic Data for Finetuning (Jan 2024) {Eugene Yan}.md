#article 
Link: https://eugeneyan.com/writing/synthetic/

This was an article that Eugene said that he wrote in order to better understand the landscape of synthetic data, in late Jan 2024 (pre-Claude3, pre-Sora, pre-Gemeni 1.5)

---------

==Synthetic data== refers to data generated via a model or simulated environment, instead of naturally occurring on the internet or annotated by humans.
- Useful for pre-training, instruction-tuning, and preference-tuning.

----
Aside: [[Instruction-Tuning]] vs [[Preference-Tuning]]
- Preference Tuning
	- A method where the model is fine-tuned to align its outputs with human preferences.
	- This can be based on factors like coherence, relevance, politeness, etc.
- Instruction Tuning
	- Involves fine-tuning the model on a dataset of prompt that are accompanied by the desired outputs.
	- Aims to improve the model's ability to follow specific instructions or understand tasks as described in natural language.
	- The training data for instruction-tuning typically consists of a wide variety of tasks, along with examples of how those tasks should be completed.

These are both subsets of [[Supervised Fine-Tuning]]

-----

Relative to human annotation, ==synthetic data is *faster* and *cheaper* to generate task-specific synthetic data.==

Furthermore, ==synthetic data's *quality* and *diversity* of synthetic data often exceeds that of human annotators, leading to improved performance and generalization when models are finetuned on synthetic data!==
- ((Note that this cuts both ways -- Unless you specifically engineer it, I don't imagine that GPT-4 is going to produce synthetic data that has mis-spellings... which *might* be important to you for robustness?))

Finally, ==synthetic data sidesteps privacy and copyright concerns==.

There are two main approaches to generate synthetic data:
1. [[Distillation]] from a stronger model
2. Self-Improvement on the model's own output

Notable Models in table:

|                  | Pretraining                     | Instruction-tuning                                                                                                         | Preference-tuning           |
| ---------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| Distillation     | TinyStories, [[Phi-1]], Phi-1.5 | Unnatural Instructions, [[Alpaca]], [[Vicuna]], [[Orca]], [[Orca 2]], WizardLM, WizardCoder, MagicCoder, Phi-1 (exercises) | Starling-7B                 |
| Self-Improvement | AlphaGeometry, [[WRAP]]         | Self-Instruct, [[SPIN]], Instruction Backtranslation, ReST, [[Constitutional AI]]                                          | SteerLM, Self-Rewarding CAI |




































