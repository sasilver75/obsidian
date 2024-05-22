March 20, 2024
[[Allen Institute|AI2]], lead author[[Nathan Lambert]]
[RewardBench: Evaluating Reward Models for Language Modeling](https://arxiv.org/abs/2403.13787)
...
Takeaway: ...

HuggingFace Space: https://huggingface.co/spaces/allenai/reward-bench

---

Reward models are critical to RLHF; the whole DPO craze in open alignment has distracted people, but in big labs, reward models still matter a lot, and we should be evaluating them. 
- The reward models are where the human preferences and values are being encoded, so they're interesting things to measure.

RL has these feedback loops; RL is designed around the environment providing some stable reward function; in RLHF, we're sort of co-designing the agent and the reward, so we need to be careful we're not messing things up -- it's easy to lie to yourself... The reward model both encapsulates the reward and the environment.

There's models that want to refuse everything, models that want to answer everything, and models that refuse things we want them to refuse, and answer things we want them to answer (which is what we actually want like the lovely [[Zephyr]] models).



Abstract
> Reward models (RMs) are at the crux of successful RLHF to align pretrained models to human preferences, yet there has been relatively little study that focuses on evaluation of those reward models. Evaluating reward models presents an opportunity to understand the opaque technologies used for alignment of language models and which values are embedded in them. To date, very few descriptors of capabilities, training methods, or open-source reward models exist. In this paper, we present REWARDBENCH, a benchmark dataset and code-base for evaluation, to enhance scientific understanding of reward models. The REWARDBENCH dataset is a collection of prompt-win-lose trios spanning chat, reasoning, and safety, to benchmark how reward models perform on challenging, structured and out-of-distribution queries. We created specific comparison datasets for RMs that have subtle, but verifiable reasons (e.g. bugs, incorrect facts) why one answer should be preferred to another. On the REWARDBENCH leaderboard, we evaluate reward models trained with a variety of methods, such as the direct MLE training of classifiers and the implicit reward modeling of Direct Preference Optimization (DPO), and on a spectrum of datasets. We present many findings on propensity for refusals, reasoning limitations, and instruction following shortcoming

# Paper Figures

# Non-Paper Figures

![[Pasted image 20240521183010.png]]
- A ton of data in here across a bunch of different subsets, but you can see we evaluate chat, chat hard (trick questions) safety (whether they refuse what they should), reasoning (math code), prior sets (anthropic [[Helpful and Harmless|HH]], learning to summarize [[TL;DR]], [[Stanford Human Preferences]], etc.)