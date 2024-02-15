Link: https://www.interconnects.ai/p/the-dpo-debate

-----

Two weeks ago, we released a blog post saying that we'd scaled DPO to a 70B parameter model. Since then, the debate on the FINAL and CORRECT method for integrating human preferences into large language models (LLMs) has only burned brighter.

The argument boils to the question:
- ==Do we need the inner workings of reinforcement learning, with value functions, policy gradients, and all, to align language models with RLHF?==

The problem we need to address before we can answer this question is that we first need to build a lot of high-quality datasets and tools to answer it definitively. At least we can lay out the groundwork for what we need to test to get those answers.

DPO can clearly create a good model; results show that the 70B Tulu DPO model we trained is on average about on the level of the original CHatGPT (175B). This is a marked achievement for open-source and academic progress.

One year out from ChatGPT, open models have just about caught up, and DPO is part of it. PPO has solid models, and its true limits aren't known due to how tricky it is to work with!

Tulu and [[Zephyr]] were breakthroughs, and not due to tweaking the optimizer -- instead, due to data and exploration of hyperparameters. DOP has a straightforward implementation, which is why people love it and it can be implemented in tons of existing LLM toolkits. This goes to show that the loss function is not universal magic.

==The long story short for open RLHF efforts is that we have more limitations with data and tooling and evaluation than we do for optimizer choice.==

In the Starling blog post (a 7B RLHF'd model), they used Advantage-Induced Policy Alignment (APA), and said:
> DPO is simpler in implementation, which directly updates the language model on the pre-collected offline preference dataset.
> In contrast, RL methods like PPO sample new responses using the current language model, score the new responses with the trained reward model, and update the language model with the reward information on the new responses.
> ==Despite challenges in hyperparameter optimization for PPO, we found that, with optimal hyperparameter settings, the online RL methods yielded comparably strong results.==

We now know that DPO at least needs weird hyperparameters (eg low learning rates, like 5e-7) and may need special data like [[UltraFeedback]] to work.

Recent work by Nvidia "shows" that DPO is less performant than PPO (sight edge to RLHF+PPO on MT-Bench and a few other benchmark).

Most of the evidence in the early literature of an extremely empirical field like RLHF is not conclusive -- we need to build a web of more direct comparisons with shared code and detailed parameters.

Questions on DPO:
- Is the engineering abstraction separating reward model training from policy training important to scalable and stable RLHF? Leading RLHF labs do this, but it isn't possible in DPO.
- Does the implicit reward they define as a log-prob ratio behave like the classifier-based rewards (direct scalars) from a standard reward model?
- Do DPO algorithms have an advantage in the synthetic preference data regime (where GPT4 labels the score/choice)?
- Will it be important to be able to train reward models with different preference rankings and apply RL to them?
- Does bootstrapping pairwise preferences from scalar reward functions like code-based execution scores, toxicity, and other clever operations translate to dPO? If not, we'll always need an RL-based alternative.
- How do DPO and offline RL algorithms overcome the lack of feedback that generating samples give, where samples have some feedback of the policy in the form of a scalar reward that goes into updates?


----

Do we have the right RLHF objective?
- It's a good time to remember when debating all of this about DPO and PPO if the objective/problem specification we're using for RLHF is even the right one (a KL-constrained reward).
	- There may be "objective mismatches" (nato paper on this) between reward model training, policy training, and evaluation. See: "The Alignment Ceiling"