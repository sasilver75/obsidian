#article 
Link: https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo

------

![[Pasted image 20240209223909.png]]

Recent AI research has revealed that reinforcement learning (RL), specifically [[Reinforcement Learning from Human Feedback]], is a key component of training large language models (LLMs).
- But many AI practitioners avoid the use of RL due to several factors, including a lack of familiarity with RL, over a preference for familiar supervised learning techniques.

This is an overview of RL, part of the series with:
- [[Basics of Reinforcement Learning for LLMs (Sep 2023) {Simon Wolfe, Deep Learning Focus Newsletter}]]
- [[Policy Gradients - The Foundation of RLHF (Oct 2023) {Cameron Wolfe, Deep Learning Focus Newsletter}]]

Now, we'll dive into the algorithm that lays the foundation for language model alignment -- [[Proximal Policy Optimization]], or PPO.
- PPO works well and is easy to understand and use, making it a desirable algorithm from a practical perspective.
	- This is why OpenAI used PPO in the RLHF implementation to align InstructGPT and ChatGPT.


# Background Information
- We want to learn exactly how RLHF works, so that we aren't afraid of it, and so that we have a new tool in our toolbelt.
![[Pasted image 20240209224822.png]]
- In this section, we'll:
	1. Briefly cover the RL algorithms we've learned so far, focusing on their limitations.
	2. Overview the problem setup of RL for language model alignment, which we'll use as relevant context when learning about new algorithms.
	3. Learn about [[KL Divergence]], a useful concept for both RL and Machine Learning in general.

### What have we learned so far?
![[Pasted image 20240209225304.png]]
- We've mostly focused on fundamental concepts within RL including:
	1. Basics of RL for LLMs: Problem setup (re: LLMs) and basic algorithms like Deep Q-Learning
	2. Policy Optimization: Understanding policy gradients - the class of optimization techniques used by RLHF - and basic algorithms in this space.

We'll build on these basic concepts by diving into two RL algorithms that are more directly related to RLHF:
1. ==Trust Region Policy Optimization== (TRPO)
2. ==[[Proximal Policy Optimization]] ==(PPO)

Similarly to the vanilla policy gradient algorithm we saw in a prior overview, both of these are based on policy gradients -- but PPO (an extension of TRPO) is more commonly used for RLHF.

Recap:
![[Pasted image 20240209225741.png]]

But wait, why do we need these two new algorithms, rather than either [[Deep Q-Learning]] or the Vanilla Policy Gradient Algorithm?
- These have notable limitations when we want to use them to solve complex problems!
	- [[Deep Q-Learning|DQL]] can only be applied in simple environments (eg Atari); it's effective for problems with discrete action spaces, but struggles to generalize to continuous action spaces, where it's known to fail at solving even simple problems!
	- Vanilla Policy Gradient Algorithm has poor data efficiency and robustness -- we need to collect *tons* of data from our environment to eliminate noise in the policy gradient estimate and train the underlying policy.

What we want is an RL algorithm that is:
1. Generally applicable (To both discrete and continuous problems)
2. Data efficient
3. Robust (i.e. works without too much tuning)
4. Simple (i.e. not too difficult to either understand or implement)

TRPO satisfies the first two points above, but PPO satisfies all four! ((Sort of like RAFT)), Due to its *simplicity and effectiveness*, PPO is widely used and has become the go-to choice for aligning language models via RLHF!

# Better Algorithms for Reinforcement Learning



# Takeaways


































