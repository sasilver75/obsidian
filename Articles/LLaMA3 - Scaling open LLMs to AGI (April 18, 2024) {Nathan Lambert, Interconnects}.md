#article 
Link: https://www.interconnects.ai/p/llama-3-and-scaling-open-llms

--------

Another criticism of the open LLM ecosystems is about whether they can continue to scale their models along with hyper-scalers like Google or OpenAI.

We've got access to multiple open-weight models over 100B params in 2024, and we've learned that we'll take a step to 400B parameters with the largest [[LLaMA 3]] model, assuming that it's open-source.

Continuing this to next year, we could easily see an open model with parameter counts similar to GPT-4 at 1 trillion.

Today, LLAMA 3 was released
- Base models in 8B, 70B, with the 400B one still training

The most important things about this release are ==scaling== and ==more of the same LLM model fatigue==. The other details are interesting, but mostly are just technical details.

Meta is leaning into *vibes* (400B) and *distribution* (it will be everywhere!)

## Summary

Released so far are:
- meta-llama/Meta-Llama-3-8B
- meta-llama/Meta-Llama-3-8B-Instruct
- meta-llama/Meta-Llama-3-70B
- meta-llama/Meta-Llama-3-70B-Instruct
- meta-llama/Meta-Llama-Guard-2-8B
	- The safety classifier for building open, safe LLM systems

Architecture:
- Largely the same as Llama 2, but accommodates a tokenizer change. ==The fact that the models *are not MoE models* is notable in the context of recent trends.== Most model providers have shifted to MoE in the last year.

Upgraded tokenizer
- ==Vocabulary size from 32k to 128k.== 
- Llama 3 uses fewer tokens, so a bigger model is similar at inference time. Built on OpenAI's ==[TIkToken](https://github.com/openai/tiktoken)== tokenizer. Similar to GPT-4 tokenizer with around 100k vocabulary.

Training compute
- Trained on 24k GPUs (max 16k concurrently) on 15T+ tokens

Training data (not released!)
- Most of the gains come from data and scale.
- Most details on the data are not released.
- ==Only 5% of the dataset is non-English== or code from 30 languages.

Expanded context window
- ==Increased from 4k to 8192 context length==

Wild alignment workflow
- Uses all of:
	- [[Instruction-Tuning]]
	- [[Rejection Sampling]] (As done in Llama 2)
	- [[Direct Preference Optimization]] (DPO)
	- [[Proximal Policy Optimization]] (PPO)

DPO is the only addition from Llama 2, but most improvements came from cleaning the data rather than specific training approaches.

Is it open source?
- Largely the same as Llama 2 -- it's not an open use model without the data, and changes push the "built with Llama 3" as the Meta brand strategy.

Other items released:
- ==torchtune==: a PyTorch native library for LLM fine-tuning
- New security tools focused on cybersecurity, child safety, and fostering community
- Responsible AI approach: A substantial document saying they did their due dilligence, didn't mess up false refusals again, and more -- seems like policy hedging.



# Pretraining, data, and basic evals

The core part of Meta's release: ==pushing a ton of tokens through base language models because most players in the open LLM community don't have enough compute to do so==. The three models are training on a ton of data.

The 8B model stands a lot less among its peers (because the 8B space is quite crowded) -- the 70B and pending 400B are the most important parts of this release.

Another missing piece is a smaller model, say ==1 B parameters==, given that other major players like Mistral are yet to release a model here.

The ==8B model is likely the best model for local LLM users, but it's not obvious that Meta is above average at all in terms of compute efficiency.==

The numbers that Meta highlights in the blog post look pretty decent for the 8B model, where they compare to open models.

If you look at MMLU, the models largely track in a per-compute sense (measuring FLOPs by multiplying training tokens by active parameter sizes).

![[Pasted image 20240423134526.png]]

Meta's models, and the commitment to better inference time performance at the cost of training costs, completely go against any notion of [[Chinchilla]] optimal training dataset size.
((It's hard to think about which models within a vertical slice are "more data-efficient" or just have some test set leakage))

Both our 8B and 70B parameter models continued to improve log-linearly after we train them on up to 15T tokens

==Training on so many tokens makes the idea real that you should train on all the data you have, or you'll be behind the competition.==

![[Pasted image 20240423134726.png]]
LLaMA 3 way off the Chinchilla graph

A few details on the dataset

***LLM filtering of pretraining data***
- We found that ==previous generations of Llama are surprisingly good at identifying high-quality data, hence we used Llama 2 to ***generate*** the training data for the text-quality classifiers that are powering Llama 3.==

This seems like an offline way of what Microsoft's [[Rho-1]] models do (via a reward model) and fit in with the trend that we've heard about regarding large companies *pretraining* on synthetic data itself!

They had an aside on how they pack their batches with parts of documents so that sequences  themselves don't cross document boundaries in a strange way. They use masking tokens instead to get sequences the same length.

Long context behavior is very important, so it'll be interesting to see if Meta extends the context length of these models by training on a few more hundred billion tokens of long-context.

Many fine-tuning recipes people use online to extend context lengths don't get anywhere near close to the quality of models like ChatGPT has, so they maybe just haven't cracked the technical challenge yet at meta.

# Alignment and Human Evaluations
- The fine-tuning section of the blog post leaves a lot to be speculated; in the [[LLaMA 2]] paper, this is the most impressive section.

> Our approach to post-training is a combination of [[Supervised Fine-Tuning|SFT]], [[Rejection Sampling]], [[Proximal Policy Optimization|PPO]], and [[Direct Preference Optimization|DPO]]. The ==quality of the prompts that are used in SFT and the preference rankings that are used in PPO and DPO have an outsized influence on the performance of aligned models.==

> Some of our ==biggest improvements in model quality came from carefully curating this data== and performing multiple rounds of quality assurance on annotations provided by human annotators.

Recall their staged training from the LLaMA2 paper; this is almost surely something similar, or that they used some methods for one of the models, and not for the other.

We all know Meta was rushing this out, so it could be that each of the 8B/70B models only uses one of either DPO/PPO after rejection sampling.
- Talking with the authors after LLaMA 2's release [[Rejection Sampling]] was their simplest method to get working in a stable manner.
- ==Rejection sampling is taking all your model's completions from the InstructionFineTuning distribution, ranking via a reward model, and then training on the top N% of data as normal instruction-tuning.==
	- ((Curious what "completions" means here; I think it's the supervised labels from IFT?))


![[Pasted image 20240423140038.png]]
Above, from the LLaMA 2 paper:
- We hope that we'll see some similar plots for the staged training of LLaMA 3
- There are two ways to look at this:
	1. Ordering based on some performance
	2. Trying lots of things on short timeframes before human data is collected (working with Scale.ai isn't easy!)

Regardless, we'd expect the order of operations to be for each model to be: ==IFT -> Rejection Sampling -> DPO -> PPO==
- At each step, the model gains a little more leeway to explore.

==He guesses they did DPO on their own model generations, and not human data or GPT 4 data, which makes it *very different* from the other open models (and closer to paper like [[Self-Reward]])==

There's also a very familiar data comment from the blog post when compared to LLaMA 2:

> The fine-tuning data includes publicly available instruction datasets, as well as over ==10M human-annotated examples==. Neither the pretraining nor fine-tuning datasets include Meta user data.

The data for Llama 2 Chat was also a mix, but the private human data is doing the work that could make these fine-tunes score more highly on ChatBotArena than other fine-tuned models we've seen recently.

![[Pasted image 20240423145655.png]]

It's also cool that Meta is releasing [Open-Weight Safety Filters](https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B) (`Meta-Llama-Guard-2-8B`) so that people can better understand the multi-stage process of ensuring a safe LLM application.
- ==It's a classifier that says which type of risk can emerge from a given piece of text, so moderation can be automated by categories.==


### Same Llama license (mostly)

> If you use the Llama Materials to create, train, fine tune, or otherwise improve an AI model, which is distributed or made available, you shall also include “Llama 3” at the beginning of any such AI model name.

> The final point people will be upset about is that **we still only can use Llama outputs to improve Llama models**. Here’s a diff of the licenses (tap to zoom).
- ==This is very interesting!==
















