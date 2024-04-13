#article 
LInk: https://www.interconnects.ai/p/llm-synthetic-data

----

> Synthetic data is the accelerator of the next phase of AI; what it is, and what it means.

The notion of synthetic data (data created by a human rather than a machine) has a long history in NLP and ML broadly -- it's usually closely tied to the notion of Data Augmentation, where a piece of data can be modified slightly to add diversity to the dataset.

One of the older links in NLP is [[Back-Translation]], where synthetic data is a *new translation task* from the mode's outputs to the origianl text.

---
Aside: [[Back-Translation]]
- Given an input text in some *source language* (eg English)
- Translate this text to some temporary *destination* language (eg French)
- Translate *back* the previously translated text into the source language (eg English)
---

Today, synthetic data has taken on a much grander task -- removing humans from the loop of making AI both aligned and enjoyable to use -- a task spearheaded by Anthropic's training methods and OpenAI's mysterious new Superalignment team, tasked with using AI feedback to solve alignment (because humans won't be powerful enough).

In the meantime, synthetic data has become a go-to resource for many of the most popular boutique, open model providers fine-tuning Meta/Mistral's models.
- There are even rumors that Mistral is just Llama 2 pretraining continued on GPT 4 tokens!

# Can Synthetic Data provide the next breakthrough?
- The current/next generation of models will have likely trained on all the high-quality data on the internet, with the latest sources coming from things like YouTube and podcasts. 
- Model providers will be looking for new directions to get the last few orders of magnitude of data needed for scaling laws to hold. A core assumption behind proponents of synthetic data at scale is that simply adding more data will make the model better at solving long-tail tasks/evaluations.

Nato thinks that we have ~2 more generations of models for scaling to be worth it, as computational costs skyrocket.

==The argument against synthetic data follows that all the data that we're generating is from the same distribution as the current best models, so some do not expect the SOTA to be advanced by it.==

Regardless -- in the open, we're well behind GPT-Turbo, so we have a ton of runway to go by copying this data at many stages of development.
- ==Nate thinks that GPT4 tokens are good enough that training on them is likely to help GPT5, because most generated sequences are still so unique in the possible space of all tokens that it provides more useful diversity than bias.==

Many of the trending models on [[HuggingFace]] Hub use synthetic data as a way to move fast and afford to try techniques behind the SOT language models in industry.
- At the same time, the [[Anthropic]] and [[OpenAI]]s of the world use it because it's the only way to move forward at scale and capabilities in their ballpark. ==Frontier-model builders== are creating ==nearly-pretraining-sized synthetic datasets==, spending $10M+.
- Meanwhile, ==smaller models== use it because human data at the same scale is thousands of times more expensive. They're ==creating instruction datasets, spending (eg) $10.==


The feature that Nato expects models to cultivate using synthetic data is ==ROBUSTNESS==
- A wider variety of context within a given task will improve models across the board; Synthetic data lets models see an obscure data point *a few times* in training, rather than just once.


# Anthropic's [[Constitutional AI]] and Synthetic-Data Sophistication

There's a general rumor that Anthropic uses a ton of synthetic data, which helps them get a lot of robustness, such as their new results for Claude 2.1, where the model answers incorrectly by a factor of 2x and more precisely refuses questions that it doesn't know.

[[Constitutional AI]] (used extensively in Claude) is the largest confirmed use of synthetic data so far (Nov 2023).

Constitutional AI has two uses of synthetic data:
1. Critiques of *==instruction-tune data==* to follow a set of principles, asking questions like "Is the answer encouraging violence?" or "Is the answer truthful?"
	- ==When the model generates answers to questions, LLM critics first check the answer against the list of principles on the constitution, refining the answer over time.==
	- Then, we fine-tune the model on the *resulting* dataset ((in a supervised manner?))

2. Generates ==pairwise preference data== by ==using a ***language model*** to answer which completion was better (LLM as a judge), given the context of a random principle from the constitution.== Then we use this AI generated data to finetune the model -- [[Reinforcement Learning from from AI Feedback|RLAIF]], rather than [[Reinforcement Learning from Human Feedback|RLHF]]. ((Though I can't imagine that they didn't use human data too? I wonder what language model they used as a judge, and whether *that* was finetuned on human preferences.))

![[Pasted image 20240412190728.png]]
Above: [[Constitutional AI]] process
1. Generate responses to harmful prompts
2. LLMs critique and revise responses, creating an instruction-following dataset that's concordant with the constitution
3. Instruction-fine-tune your model on this dataset
4. We use an LLM as a Judge as our preference model, and create a preference dataset (?)
5. We finetune the original model with RL using the new preference model (an LLM)


A lot of ***confusion*** around CAI is the second step above is the one that they named RLAIF and promoted more heavily in the aper/media, but ==it requires *BOTH* the instruction-finetuning and human-preference training to be called CAI!==

==CAI = Principled Instruction Correction + Principle-following RLAIF (rather than just generic RLAIF)==

The details and complexities of CAI aren't fully appreciated because of ths ubtle differences between:
1. Generating critiques/scoring multiple answers
2. Finding an empirically stable set of principles to use in the synthetic data generation phase (RLAIF part)

When doing CAI, Anthropic needs to:
1. Keep the principles to similar lengths
2. Avoid overlapping concepts
3. Cannot use any size of constitution

Additionally, It's simply the case that some principles generate data that's numerically unstable!


# Synthetic Instructions vs Preferences vs Critiques
- Among open models, it's easy to see a progression of how people are using synthetic data.
	- Models like [[Alpaca]] and [[Vicuna]] used synthetic instruction data for the [[Supervised Fine-Tuning|SFT]] of [[LLaMA]] 1 models to reach impressive performance in the 7-13B range.
		- A large debate permeated this space for months: "Can we actually (legally) fine-tune models on the outputs of OpenAI's models?" These days we *all* train on synthetic outputs.
	- Many open instruction datasets are ==advances in [[Self-Instruct]]-style methods, where you create a "seed" of instructions and use an LLM to generate instructions similar to them==.
		- There are now many ways of doing this, but they're all in the early days of trying to figure out the right ways to add diversity to the dataset.
	- Today, ==synthetic preference datasets== are starting to emerge. These are mostly scores/asking which is better, analogous to [[MT Bench]] and [[AlpacaEval]] scoring method, but keeping the score or win/loss as a datapoint for training.
		- An example is [[UltraFeedback]], which was used for some of the first open models, highlighting the potential of RLHF methods for chat models, like [[Zephyr]] and Tulu -- this is not normal practice, weeks later.
			- UltraFeedback collects prompts from user sources like ShareGPT and existing instruction datasets like FLAN, and includes (?) model-generated critiques and completions.
	- The final frontier is preference or instruction data generated through ==AI critiques==. Critiques are a process of repeatedly using an LLM to fine-tune data with respect to a specific principle or question.
		- The addition of more context into the process makes the prerequisite capabilities of the model much higher, when it comes to critiquing synthetic data.
		- See: Critic models like Meta's [[Shepherd]] (Aug, 2023) and KAIST's [[Prometheus]] (Oct, 2023)

Note: Nato says "Be warned: Prompt datasets like ShareGPT have low average quality and narrow distributions!"


Nato suspects it will take some substantial effort to reproduce Anthropic's CAI results without having the right models to start with; it's inferred by Anthropic that many of these methods don't really work without their reward model/RLHF capabilities at 50B parameters and up!

![[Pasted image 20240413135830.png]]
We've shared a rough graphic used to highlight the fact that synthetic instruction generation is upstream of preferences, which are both upstream of critiques in infra and training difficulty.

# Two Synthetic Data Tricks
1. ==Always use the best model for generation==
	- Your models are only good as your data -- many don't want to pay OpenAI for training data, but the cost-benefit so far is on the side of using the best model -- do so!
2. ==Expect variation in APIs, so lock the API version when you can==
	- API model endpoint changes can result in large deltas in your results.