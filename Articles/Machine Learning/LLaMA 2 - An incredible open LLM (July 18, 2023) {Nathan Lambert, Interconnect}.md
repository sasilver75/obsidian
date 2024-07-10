#article 
Link: [Link](https://www.interconnects.ai/p/llama-2-from-meta)

----
# Overview

In short, [[LLaMA 2]] is a continuation of the [[LLaMA]] formula, with substantial technical expansion in terms of:
- Data quality
- Training techniques (including novel research artifacts)
- Capabilities evaluation
- Safety training
- responsible releases.

The base model of LLaMA 2 is very strong (beyond [[GPT-3]]), and the fine-tuned chat versions of the model seem to be about on par with [[ChatGPT]]. It's a huge leap forward for open-source.

- What is the model?
	- Multiple models (7B, 13B, 34B, 70B, with chat variants for each size)
	- Compared to last time, Meta increased the size of the pretraining corpus by 40%, to ~2T tokens, and doubled the context length (to 4k), and adopted [[Grouped Query Attention]].

Cost an estimated $25M in preference data, and a very large team.
This org is seemingly distinct from [[Yann LeCun]] and everyone in the original FAIR.

New method for multi-turn consistency -- ==Ghost Attention== (GAtt), inspired by [[Context Distillation]].
- These methods are often hacks to improve model performance until we better understand how to train models to our needs.

---
Ghost Attention
- The model has a cool trick that helps it be useful at following multi-turn directions (eg so it doesn't respond like a pirate like you asked, but then forget in a few turns).
- GAtt is similar to context distillation (training a model on a long prompt, and then running supervised learning onto the output with a shorter system prompt)

The method works as follows:
- Concatenate this first-turn character-style instruction to all user messages of the conversation, with hidden tokens.
- Sample from this synthetic training style from the latest chat model -- pass in the modified prompt. This entails a set of hobbies, language styles, personas for training data in RLHF.
- Use this more-heavily-prompted data for IFT (multi-turn). In training, we set the loss to 0 for intermediate turns with added data, which isn't that well-explained in the paper. It seems they ignore the impact on the gradient of intermediate turns when fine-tuning with the autoregressive prediction loss on the synthetic data chain, so they're fine-tuning on an instruction with "act as a chartacter" and a bunch of filler text before a final turn.

Essentially they're adding more prompting at inference time for training, then using that data and removing the prompt in the future. They comment that it makes long conversations much more consistent in terms of instruction-following.

----


Reward models: Uses *two reward models* (helpful, harmless?) to avoid the safety-helpfulness tradeoff identified in Anthropic's work

Data controls: A ton of commentary on distribution control (as Nato has said is key to RLHF); hard to reproduce.

RLHF process: Uses a two-stage RLHF approach, starting with [[Rejection Sampling]], then doing [[Rejection Sampling]] + [[Proximal Policy Optimization|PPO]]. Indicates RLHF as extremely important.

Almost half of the paper is safety evals, and detailed context distillation and RLHF for safety purposes. 

# Philosophy
- An important starting point is trying to dissect what the goals of the model are, and how that differs from what they can put in the paper.
	- Meta is in a tenuous political position; Meta tried to make it exceedingly clear that they don't use user data to train their LLaMA models.
	- LLaMA 2 feels like an incredible double-down on the original LLaMA formula.

# Base Model
- The model is very similar to the original LLaMA in architecture, and most of hte changes are to the data and training processes, other than the aforementioned 4k context length and the [[Grouped Query Attention|GQA]].
- Most of the paper is on evaluation and finetuning, rather than on the dark magic of creating a great base model.

![[Pasted image 20240426014118.png]]

# Preference Data
- A big takeaway is that Meta agrees that ==the reward model is the key of RLHF==, and ==RLHF is the key to models==. 
- To get good reward models, Meta had to push hard on gathering preference data extremely upgraded from what the open-source community is working with

Preference Data Details:
- Collected ==binary comparisons, rather than other fancier feedback==... also information like *"significant better, better, slightly better, neglibgibly better/unsure"*
- Used ==multi-turn preferences==, where model responses are taken from different model checkpoints with varying temperature, to generate diversity between pairs.
- Focus collection on ==helpfulness and safety== (as opposed to *honesty*), using separate guidelines at data collection time for each vendor (safety is often a much more deceptive prompting style).
- The team added additional ==safety metadata== to the collection, showcasing whcih responses are safe from the models at each turn; when this is passed to the modeling phase, they don't include any examples where the chosen response was unsafe, while the other response *was* safe; "we believe safer responses will be better/preferred by humans, in the end."
- They don't detail additional metadata being logged, but it's likely they did, in order to identify potential bugs and data issues (confusing prompts, requiring tools to solve, etc.)
- Deployed ==iterative collection for distribution management==: "Human annotations were collected in batches on a weekly basis; as we collected more preference data, our reward models improved, and we were able to train progressively better versions for LLaMA 2-Chat"

![[Pasted image 20240426020753.png]]
Above: The scale of preference data was impressive, likely costing $8M+ on dat alone, with way more turns than were often available at that time.

# Reward Modeling

This section can be summarized by two important details:
1. ==Two reward models are trained== to separate the goals of *helpful* and *safe*.
2. The iterative deployments and scaling laws of how much preference data is used/need.

To start, the paper says that they train two separate reward models, with one optimized for helpfulness ("==Helpfulness RM=="), and another for safety ("==Safety RM==").
- These are both built on the base language model, with an additional linear *regression layer* replacing the normal language model head.
	- ((I'm curious about the regression head, since I thought that the preference data that *humans* collected was binary -- does that mean 0/1, or does that just mean "between two prompts?" I know that they attached "significantly better", "slightly better", etc. information to decisions; is that like a 1-5 scale?))
- They don't indicate which of the pretrained model checkpoints the model is from, in terms of size, but they almost alway use the most recent chat model.

The more interesting points are the following:
- Adds a margin term to the reward model loss function (*proportional to the confidence of preference*), improves helpfulness.
- The preference delta between model comparisons decreases over batches from the data vendor as the model converges in their setup.
- The authors compare their reward model to using GPT4 as a reward model, and find that they beat it, but reward models trained only on open-source data don't beat GPT4.

Paper says:
> *"We note that reward model accuracy is one of the most important proxies for the final performance of LLaMA 2-Chat"*

This makes it even more annoying when we realize that no one is open-sourcing strong reward models for investigation of potential issues and utilization.


# RLHF and Fine-Tuning
- Meta came out swinging, showcasing how they used RLHF to improve their model.

![[Pasted image 20240426130848.png]]
==Meta iteratively trains 5 RLHF versions with progressing data distributions ,and shows how the RLHF process shifts the generated texts towards higher rewards.==

Meta says that open-source instruction datasets are "meh" -- the most recent trend in the open-source community is filtering datasets and the notion of "uncensored data," which all probably happened well-after Meta did the LLaMA 2 SFT.

The amount of annotation data used by Meat (27,540) is similar to what Anthropic and OpenAI are rumored to have used (~10k order of magnitude), which is a win for reproducibility.

We also observed that different annotation platforms and vendors can result in markedly different down-stream model performance, highlighting the importance of data checks even when using vendors to source annotations.


> We've found that outputs sampled from the resulting SFT models were often competitive with SFT data handwritten by human annotators., suggesting we could reprioritize and devote more annotation effort to preference-based annotation for RLHF

A missing piece could be what filtering they used to identify strong data; Until these are common knowledge, open-source training of instruction models likely will still be behind.

Focusing then on the illusive RL component:
> Our findings underscore that the crucial determinant of RLHF's success lies in the synergy it fosters between humans and LLMs throughout the annotation process. Even with proficient annotators, each individual writes with significant variation; a model finetuned on SFT annotation learns this diversity, including the tail-end of poorly-executed annotations. The model's performance is called by the writing abilities of the most-skilled annotators.


> Throughout the RLHF stage, the accumulation of iterative reward modeling data in parallel with model enhancements is crucial to ensure the reward models remain within distribution.

This type of thing is why Nate says that an effective RLHF requires a moderately sized team; this type of work requires contracts and close connects with external companies, which is always a bit of a time sink due to culture and communication mismatch.

The RLHF baselines the authors use are [[Proximal Policy Optimization|PPO]] and [[Rejection Sampling]] fine-tuning (similar to Best-of-N sampling). 
- PPO is the algorithm that's most popular for online RL (trial and error learning, so to say) -- specifically because *it's the most popular.*
- [[Rejection Sampling]] is the idea that you sample a batch of K completions from a language model policy, and then evaluate them across a reward model, returning the best one. If you retain the best few outputs via the reward model, your policy can improves.

# Evaluation (capabilities)
- LLaMA 2 is way better than any other open source model at all scales on the HuggingFace Open LLM leaderboard.
- The model scores better on automatic and less flashy benchmarks like [[MMLU|MMLU]].
- The base model capability is what others rely on; substantial data efforts are likely the biggest factor in these "basic" evaluations; then, RLHF makes the model easier to use and makes that knowledge available.

In terms of performance, these models go beyond ChatGPT after RLHF3
![[Pasted image 20240426132023.png]]
Some highlights of reward model testing
- They calibrate reward model scores to human labelers of preference between a pair; they were able to get a straight line correlation, which is probably hard to get.
- They compared their reward models to those trained on open-source datasets, and win; shows how things are possible in the open-source space.

Some highlights of human/model evaluation
- Evaluate models on both ChatGPT and LLaMA-2-Chat outputs to avoid models being biased by their own style enhancing their own results (they vary the LM judge, it sounds like?)
- Interesting use of inter-rater reliability metrics like Gwet's AC1/2, which are properly designed statistical tools for the job.
- Acknowledge limitations of human eval; Large evaluation prompt set doesn't cover all real-world uses.


# SAfety
- Lots of details on how safety relates to various training and evaluation steps.
- Shows performance against 2,000 different adversarial/red-teaming prompts.
![[Pasted image 20240426132401.png]]
Above: See the crazy increase in model refusal with "borderline datasets" -- what are "borderline asks," though?