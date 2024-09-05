---
aliases:
  - ReST
  - ReSTEM
---
December 11, 2023
[[DeepMind]]
Paper: [Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](https://arxiv.org/abs/2312.06585)
#zotero 
Takeaway: A simple, two-step (Generate, Improve) self-training method for when you don't have a large amount of data, but you DO have access to (eg automatic) *binary* feedback on generations (eg math problems, where you can verify correctness). Involves generating samples, filtering them using binary feedback, fine-tuning on the remaining generations, and repeating the process.
1. `Generate (E-Step)`: The language model generates multiple output samples for each input. Then, we filter samples using a binary reward, yielding a training dataset.
2. `Improve (M-step)`: The original LM is SFT'd on the training dataset from the previous Generate step; the fine-tuned model is then used in the *next* Generate step.
- (Repeat!)

Gulchere et al. (2023) proposed Reinforced Self-Training; Deepmind authors make some modifications and call their approach $ReST^{EM}$ , which can be viewed as applying expecation-maximization for reinforcement learning.

Question: What sort of decoding do they use for this? I imagine that it would matter? Because greedy decoding while eating your own shit would just pump the maximum probability tokens, right?... Temperature matters too.

- ==Authors don't really talk about the reward model that's used in the generate step, which is... the driving force behind this whole methodology.==

----

Notes:
- They make some modifications to the previously proposed ==Reinforced Self-Training== (Gulchere et al, 2023) algorithm, and call their approach $ReST^{EM}$, which can be viewed as applying expectation-maximization for RL. This algorithm alternates between the *expectation* and *maximization* steps:
	1. `Generate (E-Step)`: The language model generates multiple output samples for each input. Then, we filter samples using a binary reward, yielding a training dataset.
	2. `Improve (M-step)`: The original LM is SFT'd on the training dataset from the previous Generate step; the fine-tuned model is then used in the *next* Generate step.
- $ReST^{EM}$ and its various adaptations have had success in enhancing LMs across diverse domains, including:
	1. Machine translation
	2. Semantic parsing
	3. Preference alignment
	4. Elementary reasoning
- Our work aims to complement these efforts and investigate the effectiveness and scalability of model-generated synthetic data compared to human-generated data in the challenging domains of mathematical problem-solving (MATH) and code generation (APPS).
- For mathematical reasoning and code-generation capabilities, we notice when applying $ReST^{EM}$ to [[PaLM 2]] models of varying scales that models fine-tuned on model-generated synthetic data exhibit remarkably larger performance compared to those trained on human-written data.

Other interesting related work that is mentioned, which the paper tries to describe/formulate in the terms of $ReST^{EM}$. Worth looking at this section before/after reading these papers; See Table 1 in the figures section for a graphical comparison between methods: 
- ==Expert Iteration== (ExiT; Anthony et al., 2017)
	- Alternates between expert improvement and policy distillation. 
		- In the E step, we combine a base policy with a search procedure ([[Monte-Carlo Tree Search|MCTS]]) to generate samples from a better, expert policy.
		- In the M step, we use these expert sample to train the base policy in a supervised way, improving it to match the expert policy.
- ==Self-Taught Reasoner== (STaR; Zelikman et al., 2022)
- ==Rejection Sampling Fine-Tuning== (RFT; Yuan et al., 2023)
- ==Iterative Maximum Likelihood== (IML)
	- Optimizes a policy using a reward-weighted log-likelihood objective on self-collected data. 
	- In IML, the learned policy can significantly diverge from the initial pretrained model, which can manifest as task-specific overfitting, losing the ability to effectively generalize to other tasks or domains.
- ==Reward Weighted Regression== (RWR; Peters and Schaal, 2007)
- ==Reward Ranked Fine-Tuning== (RAFT; Dong et al., 2023)



# Paper Figures

![[Pasted image 20240506174135.png]]
Above: See that we're able to improve performance on a Math benchmark and on a Coding benchmark! These are both examples of domains where automated binary reward is possible.

![[Pasted image 20240506175738.png]]
Above: This is the $ReST^{EM}$ algorithm, which iteratively applies Generate and Improve steps to update the policy, improving reward.

![[Pasted image 20240506180833.png]]
Comparison 