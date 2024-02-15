Link: https://cameronrwolfe.substack.com/p/the-history-of-open-source-llms-imitation
See also:
- [[The History of Open-Source LLMs - Early Days (Part One) (Jul 2023) {Cameron Wolfe, PHD}]]
- [[The History of Open-Source LLMs - Better Base Models (Part Two) (Jul 2023) {Cameron Wolfe, Deep Learning Focus Newsletter}]]

-----

![[Pasted image 20240214175911.png]]

A majority of prior research on open LLMs focused heavily on creating better pre-trained base models -- but these models necessarily haven't undergone any fine-tuning or alignment, so they'll ultimately fail to match the quality of top closed-source LLM (eg [[ChatGPT]], [[Claude]]) do to their lack of alignment.
- Paid models are aligned extensively using [[Supervised Fine-Tuning]], [[Reinforcement Learning from Human Feedback]] ((and now, [[Direct Preference Optimization|DPO]])), which greatly enhances their usability.

This overview will take a look at recent research that aims to improve the quality of open-source LLMs via more extensive fine-tuning and alignment.

We'll go from initial models like OPT to the incredibly high-performing open-source LLMs that we have today (eg LLaMA-2-Chat).

![[Pasted image 20240214180312.png]]

The alignment process
- This overview will study the fine-tuning and alignment process for open-source LLMs. Prior to studying research here, we need to understand **what alignment is, and how it's accomplished!**
	- After pre-training, the model is able to accurately perform next-token prediction on arbitrary documents from its training corpus, but its output, if resampled over and over, might be repetitive and uninteresting.
	- Therefore, the model needs to be fine-tuned to improve its ==*alignment*, or its ability to generate text that aligns with the desires of a human user (e.g. follow instructions, avoid harmful output, avoid lying, produce interesting or creative output, etc.)==

![[Pasted image 20240214182227.png]]
SFT
- Alignment is commonly accomplished via two fine-tuning techniques: Supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF)
	- ((SFT is just more training, using specially selected "golden data," it seems. In the sense that I think it's just the same next-token prediction task on a higher-quality set of data.))
	- ((In comparison, RLHF is about two parts; First, we have a prompt, and we generate a number of possible outputs from it. The outputs are ranked by a labeler model (often a human), and the ranked outputs are used to train a *reward model*. In the second step, that reward model is used to scale this process, stepping in for the human in much the same way -- a prompt is selected, the policy generates an output, the reard model calculates a reward for the output, and the reward is used to update the policy using (eg) [[Proximal Policy Optimization|PPO]]))

[[Supervised Fine-Tuning]]
- Simply fine-tunes the model using a ==standard language-modeling objective over examples of *high-quality prompt and response pairs, often curated by human experts.*==
- Incredibly simple and effective, but requires careful curation of a dataset that captures "correct" behaviors.

[[Reinforcement Learning from Human Feedback|RLHF]]
- Trains the ==LLM directly on feedback from human annotations -- humans identify outputs that they like, and the LLM learns how to produce more outputs like this.==
- First, we obtain a set of prompts and generate several different outputs for the LLM for each prompt. We use a *group* of human annotators to score/rank these responses, based on their perceived quality. These scores can then be used to train a *==reward model==* (i.e. ==just a fine-tuned version of our LLM with an added regression head==) to predict the score of a response.
- Then, RLHF fine-tunes the model to maximize this score by using a reinforcement learning algorithm called PPO.
- ==Typically, the highest-performing LLMs are aligned by performing both SFT and RLHF in that order.==


# Imitation Learning











