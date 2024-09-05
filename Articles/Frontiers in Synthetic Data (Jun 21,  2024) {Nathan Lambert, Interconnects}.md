
https://www.interconnects.ai/p/frontiers-in-synthetic-data

"Trends in synthetic data that I'm watching closely in the leading open and closed models."

---


Synthetic data is being used to:
- Expand vanilla pretraining data (eg [[Web Rephrase Augmented Pre-training|Rephrasing the Web]]/WRAP)
- Creating large amounts of fine-tuning data (eg [[Nemotron-4]], where >98% of alignment data was synthetically generated)
- Alignment a la [[Constitutional AI]]
- Mistral AI's first models being pretrained on OpenAI outputs (data distillation)

Nathan is working on a project to take a strong open-weights model and use synthetic data to create as strong of a fine-tune as possible. In doing this, he's been thinking a lot about synthetic data, and has some commentary on what the academic literature is likely missing.

## Direct Distillation is still king
- The distillation phase is a window of time we have only been in for a short period; the [[Zephyr]] models really brought this into gear, with the title "DPO Distillation Process from AI Feedback."
	- But many popular datasets use completions from many models in the learning process
		- [[Alpaca]] and [[Koala]] are examples of these, and they weren't very good at even simple tasks.

- It's intuitive that there are some tasks that will be easier to "solve" with synthetic data.
	- On most open-ended generation queries like "write me an email," or "summarize this text," models like GPT-4 do remarkably well! It's easier to improve these tasks with synthetic data.
	- On tasks with complicated math or code, the best models available still fail wildly. Creating synthetic data in this vein will take much more effort.

Many datasets out there are "Created with GPT-4," but then used a form/iteration of GPT-4 that's now *100+ points behind* on LMSYS's ChatBotArena leaderboard!
- This is the delta from the current GPT-4o to the first GPT-4 version!
- 100 points below the original GPT-4 is LLaMA-2-chat-70B, which was known to be a very disliked model.
- The point here is that ==regenerating data with an updated version of frontier models can have a huge effect!==

It's largely unknown which models are best for which tasks, but surely the answer is not to always use the same GPT-4 model variant with the same sampling parameters (temperature, top-k, system prompt, etc.)


## Are Gemeni Flash and Claude Haiku Distilled?
- [[Gemeni Flash]] and Claude Haiku are becoming the most coveted model endpoints for many developers building applications with language models, due to their speed and low price.
	- Signs point to this being based on some form of [[Distillation]], rather than these models being trained from scratch.
		- EDIT: Gemeni Flash is confirmed as distilled; the updated report says that Flash is a *dense model* that's been *distilled* from Pro, which is an [[Mixture of Experts|MoE]].
	- In contrast, models of bigger size, like Claude 3.5 Sonnet or Gemeni Pro/Ultra seem to be models trained from scratch.

Roughly, the bigger model is being trained, and the smaller, distilled model is updated to reflect this with *simultaneous training!* They call this ==Online Distillation==.

## Filtering prevents Collapse
- Right now, the most popular research direction for alignment techniques is some form of iterative [[Direct Preference Optimization|DPO]].
- These papers, the first of which was the [[Self-Reward]]ing language models paper, often tout some slightly absurd vibes-based evaluation scores (eg [[AlpacaEval]])
- Directionally, these papers are all relabeling or regenerating the same dataset to then apply the same alignment method to the model.
	- This creation of iterations is inspired by the "online" nature of the other side of algorithms based on Proximal Policy Optimization ([[Proximal Policy Optimization|PPO]]). A lot of discussion points to the "online" nature of PPO being key for its performance.

These iterative (maybe synthetic?) methods look very similar to other insights that we've gotten from popular models, such as LLaMA 2 or Nemotron 340B.
- These models use *iterative rounds of alignmnet training* to get to a final checkpoint!
- After each iteration, they get *more data* to then train further on!
- There might be a large blind spot the academic-sided projects where not enough data filtering is done! ==There's plenty of research that says that self-consuming models go mad== ([link](https://arxiv.org/abs/2307.01850)), which isn't too surprising.

The core difference is to use iterative methods to *expand and accumulate data*, rather than just running it all as one closed feedback loop (like true on-policy RL).
- Recent work shows that ==accumulating data==, rather than doing true on-policy iteration, helps models avoid mode collapse ([link](https://arxiv.org/abs/2404.01413))

==The best industry papers rely heavily on filtering of synthetic data, which is likely combined with some form of accumulation== (eg keeping the absolutely best human data for every iteration).
- It seems that most of [[LLaMA 3]]'s synthetic data is in the pretraining phase (unconfirmed), where they state that they use LLaMA 2 as a filterer:
> We found that previous generations of LLaMA are good at identifying high-quality data, so we used LLaMA 2 to help build the text-quality classifiers that are powering LLaMA 3. We also leveraged synthetic data to train in areas such as reasoning, coding, and long context. For example, we used synthetic data to create longer documents to train on.

Similarly, [[Nemotron-4]]'s entire paper is focused on the training of a very good reward model, to then use as their data filterer.
> Throughout the entire alignment procedure, we conduct multiple rounds of data generation and refinement, continually improving the quality of our models.
> We utilize Nemotron4-340B-Reward to assess the quality of dialogues, assigning a score to each sample and filtering out those that fall below a predetermined threshold.

## Synthetic data strategy taxes
- Many companies will not use the same synthetic data that academics and open-source community members do, due to terms of service and license restrictions.
	- Popular synthetic data models from the community, like the recent LLaMA 3 Hermes fine-tune from Nous, rely heavily on closed models like GPT-4.
	- Big organizations like Nvidia are restricted to using permissive models (eg Mistral) or their own model outputs for training.

The open-source community benefits from the ability to use GPT-4, Gemeni Pro, and Claude 3 as teachers to finetune LLaMA 3 in hopes of building LLaMa 3 Instruct.
- It seems the case that everyone has accepted that small players will train on the outputs of models, and no one will care.
- ==Interestingly, many of the "you can't train on outputs from our model" clauses in licenses actually come from the data providers used in training the model rather than the model owners themselves!== These clauses are to protect their business, due to the strength of synthetic data.


## Pros and Cons of training on multi-output-source synthetic datasets
- [[UltraFeedback]] and [[Nectar]] are perhaps the most popular preference-tuning datasets out there. 
	- Since their release, a ton of models trained with [[Direct Preference Optimization|DPO]] (or many of the similar new methods) have used them -- they're the standard baseline in current (2024 June) open alignment research.
	- A core aspect of these datasets is that they use generations frmo *many models* in both the chosen and rejected columns.
	- There are pros and cons to training on such a diversity of generations!
- Pros
	- The diversity of model outputs means that the learning signal is a bit softer, and can be applied to more bases (??)
- Cons
	- The ceiling of these combined datasets is likely lower than full on-policy synthetic generation plus filtering, because we're forcing models to try and learn sequences that are low probability -- those weird phrases that GPT-4 repeats a lot might be very low in your LLaMA 3 base model that you're trying to train, so we don't know what applying that loss would do to the model.

==These general datasets will always have a space in the ecosystems for getting started, but we need better data selection mechanisms for taking the samples out of them that best help our *specific* training run!==


## Structured Synthetic Data
- Synthetic data is being used to generate specific instructions and responses corresponding to *verifiable behavior*.
- In [[Nemotron-4]], they mention creating instruction "prompts which explicitly define the format of the anticipated response, e.g. "the output has to be in JSON format"".
	- The goal of this is to improve on [[IFEval]].
	- Cases like this, where the model is really good at doing *exactly what you ask for*, is a big part of what we consider in vibes-based evaluations.

The 10-times harder version of this is what we expect top labs to be doing -- using ==verifiable synthetic data== for things like code and math.
Imagine how much better a model will be if the only code data you pass in is code that's been verified to not error!


## Weak-to-strong generalization is maybe real
- It's time to no longer be a skeptic of the idea of "[[Weak-to-Strong Generalization]]" -- the idea that we can use a weak model to bootstrap training data that can be used to generate a net stronger model. This sort of feedback isn't easy to be convinced by, but more evidence supporting it is building. 

> "Interestingly, we find that the teacher model does not impose a ceiling on the student model. Specifically, as the base model and alignment data are refined, the newly-aligned model is able to surpass the initial aligned model by a significant margin."
> - Nemotron

## Creating synthetic prompts is overlooked again
- The entire synthetic data movement started with [[Self-Instruct]], a paper designed to create more prompts to improve instruction following performance.
	- Today, most of the discussion is about *aggregating prompts* and creating *high-quality responses!* [[UltraFeedback]] and [[Nectar]], for example, are conglomerates of other pre-existing prompt sets.
- ==A big opportunity in synthetic data is around creating a better distribution of prompts *where you need them!*==
	- Nemotron lists the 23 prompts they used to generate synthetic prompts(rather than re-using prompts in the training set). Their reasoning:

> "Despite the availability of existing prompts, such as the LMSYS-Chat-1M prompts (Zhang et al 2023), generating synthetic prompts is an important first step in SDG (Nvidia's alignment method)"
> - Nemotron

There's been interesting research in this direction, like:
- Copying prompts from models by prompting the undocumented model with an empty prompt -- it then prints prompts from its training distribution.
- Expanding on techniques like self-instruct ([[GenQA]])








