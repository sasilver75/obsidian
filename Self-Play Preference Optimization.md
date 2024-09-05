---
aliases:
  - SPPO
---
June 14, 2024
UCLA
[Self-Play Preference Optimization for Language Model Alignment](https://www.youtube.com/watch?v=Q-81DStL7do)
#zotero 
Takeaway: ... (Honestly, I didn't understand this paper. It's very mathematical.)

----

## Introduction
- Most existing approaches to [[Reinforcement Learning from Human Feedback|RLHF]] rely on either explicit or implicit reward models.
	- In [[InstructGPT]], a reference policy $\pi_{ref}$ is first established from a pretrained or SFT'd model. An explicit reward function is then obtained by training a reward model on human preference data (employing the [[Bradley-Terry Model]]). This reward model provides some "reward score" r(y; x) for the given response y and prompt x. Subsequently, RL algorithms like [[Proximal Policy Optimization|PPO]] are used to fine-tune the reference LLM $\pi_{ref}$ by maximizing the expected reward function.
	- More recently, methods like [[Direct Preference Optimization|DPO]] have been introduced that forego the training of a separate reward model, instead using the log-likelihood ratio between a target and reference policy.
- Models like [[Bradley-Terry Model]] fall short of capturing the complexity of human of human preferences -- these models presume a monotonous and transitive relationship among preferences for different choices, but empirical evidence for humans suggests otherwise.
	- Human preferences do not always adhere to a single value-based hierarchy, and can even appear irrational, exhibiting loops in preference relations!
	- *Munos et al 2023* empirically showed that directly predicting a pairwise preference can achieve higher accuracy than predicting the preference via a BT-based reward model.
- In some recent work, the we consider a two player game between two LLMs that output responses that aim to maximize the probability of being preferred over its opponent. The goal is to identify the Nash equilibrium or von Neumann winner of the two-player constant-sum game:
![[Pasted image 20240718195447.png]]
- This minimax strategy (commonly used in two-player zero-sum games) means that for each possible $\pi$, we consider all possible $\pi'$ and find the worst-case scenario (the minimum), and then among all these worst-case scenarios for different $\pi$, we choose the $\pi$ that gives us the *best* worst-case scenario outcome (the maximum of the minimums).
- ... SPPO basically solves the above two-player constant-sum game...
- Despite using only 60k prompts from [[UltraFeedback]] and forgoing any prompt augmentation, our method achieves performance comparable to GPT-4 on the [[AlpacaEval]] 2.0 win-rate.

----
Aside: ==Symmetric Loss Functions== vs ==Asymmetric Loss Functions==
- ==Symmetric Loss Functions==
	- Treats errors in both directions equally -- in the context of preference learning, it means the loss function handles the chosen and rejected responses in a balanced way.
	- In [[Direct Preference Optimization|DPO]], the loss function is typically structured to increase the probability of the chosen response and decrease the probability of the rejected response by the same amount.
- ==Asymmetric Loss Functions==
	- An asymmetric loss function treats different types of errors/outcomes differently -- in preference learning, it might handle chosen and rejected responses in different ways.
----


## Related Work
- RLHF with Explicit/Implicit Reward Model
	- [[Reinforcement Learning from Human Feedback|RLHF]] as originally proposed is a methodology that first learns a reward model reflecting human preferences, and then uses RL to maximize that reward.  The reward model assumes a parametric model such as the [[Bradley-Terry Model|BT Model]], which assigns a "score" representing how preferred a given response is.
	- [[Direct Preference Optimization|DPO]] is claimed to be more efficient and stable, yet still implicitly assumes such a reward model that specifies the "score."
	- [[Kahneman-Tversky Optimization|KTO]] derived a different loss function from the Kahneman-Tversky human utility function, which implicitly denotes a score of the given response.
	- [[Rejection Sampling Optimization]] (RSO) utilizes a preference model to generate preference pairs with candidates sampled from the optimal policy, then preference optimization is applied to the sampled preference pairs.
	- [[Odds Ratio Preference Optimization]] can perform supervised fine-tuning and preference alignment in one training session without maintaining an intermediate reference policy.
- RLHF with General Preference Model
	- Human preference isn't strictly transitive, and can't be sufficiently represented by a single numerical score.
	- Azar et al. 2023 proposed a general preference optimization objective based on the preference probability between a pair of responses instead of a score of a single response.
		- They further propose a learning objective based on the identity mapping of the preference probability, called IPO ([[Identity-Mapping Preference Optimization|IPO]]).
- Self-Play Fine-Tuning
	- Most works above consider a single optimization procedure starting from some reference policy. 
	- The same procedure can be applied repeatedly for multiple rounds in a self-play manner! In each round, new data is generated by the policy obtained in the *last round*; the new data is then used to train a new policy that can outperform the old policy.
	- References ([[Self-Play Fine-Tuning]] (SPIN), *Beyond human data* (2023), [[Self-Reward]]) as examples of iterative improvement of language models.

## Preliminaries
- Skipping, this is a lot of math


## Self-Play Preference Optimization (SPPO)
![[Pasted image 20240718203256.png]]
![[Pasted image 20240718203028.png|500]]
- In each round, we generate K responses $y_1, y_2, ..., y_k$ according to some policy $\pi_t$ for each prompt $x$.
- Then, a preference oracle $\mathbb{P}$ will be queried to calculate the win rate among the K responses.
- Certain criteria can be applied to determine which response should be kept in the constructed dataset D_t and construct the prompt-response-probability triplet (x, y, $\hat{P}(y \succ \pi_t|x)$).
- A straightforward design choice is to include all K responses into D_t and each $\hat{P}(y \succ \pi_t|x)$ is estimated by comparing y_i to all K responses. 
	- In total, O(K^2) queries will be made.
- Then the algorithm will optimize on the dataset D_t.

...
- In practice, we use mini-batches of more than 2 responses to estimate the win rate of a given response, while DOP and IPO loss focus on a single pair of responses.
- "When there are plenty of preference pairs, DPO and IPO can ensure the policy will converge to the target policy, but when the preference pairs are scarce (eg one pair for each prompt), there's no guarantee that the estimated reward of the winner a and the estimated reward of the loser b will decrease"

## Experiments
...


## Conclusions
- By integrating a preference model and employing a batched estimation process, SPPO aligns LLMs more closely with human preferences and avoids common pitfalls such as “length bias” reward hacking.


Abstract
> Traditional reinforcement learning from human feedback (RLHF) approaches relying on parametric models like the Bradley-Terry model fall short in capturing the intransitivity and irrationality in human preferences. Recent advancements suggest that directly working with preference probabilities can yield a more accurate reflection of human preferences, enabling more flexible and accurate language model alignment. In this paper, we propose a self-play-based method for language model alignment, which treats the problem as a constant-sum two-player game aimed at identifying the Nash equilibrium policy. Our approach, dubbed Self-Play Preference Optimization (SPPO), approximates the Nash equilibrium through iterative policy updates and enjoys a theoretical convergence guarantee. Our method can effectively increase the log-likelihood of the chosen response and decrease that of the rejected response, which cannot be trivially achieved by symmetric pairwise loss such as Direct Preference Optimization (DPO) and Identity Preference Optimization (IPO). In our experiments, using only 60k prompts (without responses) from the UltraFeedback dataset and without any prompt augmentation, by leveraging a pre-trained preference model PairRM with only 0.4B parameters, SPPO can obtain a model from fine-tuning Mistral-7B-Instruct-v0.2 that achieves the state-of-the-art length-controlled win-rate of 28.53% against GPT-4-Turbo on AlpacaEval 2.0. It also outperforms the (iterative) DPO and IPO on MT-Bench and the Open LLM Leaderboard. Starting from a stronger base model Llama-3-8B-Instruct, we are able to achieve a length-controlled win rate of 38.77%. Notably, the strong performance of SPPO is achieved without additional external supervision (e.g., responses, preferences, etc.) from GPT-4 or other stronger language models. Codes are available at [this https URL](https://github.com/uclaml/SPPO).


# Paper Figures