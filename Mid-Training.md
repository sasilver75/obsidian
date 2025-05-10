References:
- [Article: Vintage Data: What's the deal with pre-training?](https://vintagedata.org/blog/posts/what-is-mid-training)

[[OpenAI]] has had a mid-training division since July 2024, and [[xAI]] is setting one up.

-----

==Mid-training== is vaguely in-between [[Pretraining]] and [[Post-Training]].
****
In July 2024, OpenAI **quietly**** announced that it had a "mid-training" division in two research job ads. Ever since, it's been rumored what the mid-training team's purpose is.

It's possible that the OpenAI mid-training team stems from the internalized continuous pretraining program.

The concept was formally introduced in academic literature by [[Microsoft Research]]'s [[Phi-3.5]]: "phi-3.5-mini and phi-3.5-MoE, which incorporate more multilingual and long-text data during mid-training."

The [[Yi]] report includes the first standard definition to date, mixing things that we'll delineate later: "In the mid-training stage, we focus on ==enhancing model capabilities and extending context length through gradual data distribution shifts==."

Since 2025, [[Allen Institute]] set a standard but opinionated definition in their [[OLMo 2]] report: "In previous OLMo iterations, we also found that both ==learning rate scheduling== and ==data mixture== play an important role. We refer to interventions at this stage of model development as mid-training."

It seems that the rise of mid-training conveys two parallel vibe shifts:
- Base and instruct training are blurred
	- Scheduling and annealing are now part of the standard approach of training. Introducing instruct-like data and/or filtered data close to to expected use has repeatedly yielded performance gains. 
	- Post-training has scaled has scaled. Compute plans, datasets, and organizational structure has been rebalanced in many organizations, and some new reasoning ==models like O3 may have been exclusively the result of iterating on post-training. In short, post-training is the new pre-training==.

There's at least one constant attribute of mid-training: It's done on a mid-range of dataset sizes, with the datasets used in the OLMo, Phi, and Yi reports ranging between 10-300 billion tokens.

Phi 3.5 underwent a mid-training stage for better language support, among other things.
Yi's report also singles out enhancing "multilingual capabilities for low-resource language," though more as a part of general data filtering/upsampling strategy.

These practices are commonly-associated with [[Continued Pretraining|Continuous Pretraining]]. 
The shift we see here is mostly organizational.

Models are routinely released with different context lengths than the one used for training.
Until OLMo 2, all early mentions of mid-training centered on long-context, with some variation.
- Phi 3.5 report is relatively evasive  but seems to mention an "initial, long-context mid-training, followed by mixed long-short post-training with SFT and DPO."

Context length extension seems to fit well within the most common definition of mid-training, but blurs any distinction between pre/mid/post training... because fundamental design decisions eg (Rope Theta values) are now done in anticipation of context length extension, in order to optimize positional embeddings.

We end up with weird schedules where long-context mid-training happen after post-training:
- Stages 1 through 3, the process starts with alignment, pre-training, and supervised fine-tuning. In stage 4, the model undergoes mid-training context extension."

The Yi report may be the first academic publication to associate mid-training to [[Curriculum Learning]], as they "implement an incremental upsampling strategy for high-quality data, emphasizing complex reasoning and multilingual capabilities for low-resource language."


Mid training quality datasets include:
- Filtered selection from pre-training, using a quality classifier (as in [[FineWeb-Edu]])
	- Classifiers are usually fairly small to be runnable at scale -- either a BERT or FastText-based model.
	- An emerging practice is to use larger decoder language model to further filter specific subsets requiring some more reasoning-based judgement.
- Curated datasets which come from some large-scale collection (eg Math, Scientific publications). Curation improves to improve specialized capacities and knowledge....
- Instruction datasets -- This is the part that "blurs" the most distinction between pre- and post-training as these collection largely anticipate finetuning.

Scaling Synthetic Data
- Large-scale synthetic data is an integral part of mid-training. Recipes are focused on math, but could be transferable elsewhere...
	- Straight generation from a more powerful model, acting here as a "teacher." This was probably the most widespread initial approach of synthetic data as introduced by Phi.
	- Question generation from answers, also frequently termed "[[Back-Translation]]."
		- Leverage the fact that a quality dataset already contains answers that could be "instructionized."
		- ==Generating questions isn't a hard task -- generating sufficiently-diverse and realistic questions in multiple languages is the actual challenge!==
	- Answer generation from formal verification -- for math and code this is relatively straight forward, but even soft-sciences can be occasionally formalized.
	- Answer generation from a LLM verification using an [[LLM-as-a-Judge]] model.


We've yet to see any research paper include RL in mid-traininhg.

RL is not necessarily just about assessing accurate answers, but accurate solutions. Reasoning traces are currently relatively brittle.
- [[DeepSeek V3]] report: "White the R1-generated data demonstrates strong accuracy, it suffers from issues such as overthinking, poor formatting, and excessive length."
- Similarly, [[QwQ]] from Qwen does weird rambling and expression switching, and has only been released as a preview for now.

[[Process Reward Model]]s aim to assess chain-of-thoughts internally... More broadly, RL rather than inference seems to have become the space where search methods are actually usable: Generations with fixed temperature and hyperparameters can be replaced with wider explorations of possible token trees.

Let's recap what we have:
- Mid-training datasets of superior quality
- Models are more and more specialized -- O1 has set new SOTA for math but at the expense of many things (translation, text-writing, etc.). 
- Ripples in training space-time -- All pretraining data is bound to by synthetically processed, and RL could be reinjected further down the line. 












