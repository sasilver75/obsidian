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
1. ==[[Trust Region Policy Optimization]]== (TRPO)
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

==TRPO satisfies the first two points above, but PPO satisfies all four==! ((Sort of like RAFT)), Due to its *simplicity and effectiveness*, PPO is widely used and has become the go-to choice for aligning language models via RLHF!

# Better Algorithms for Reinforcement Learning

### (1/2) Trust Region Policy Optimization (TRPO)
- Recall: Vanilla Policy Gradient algorithm (VPG) is limited by the fact that it can only perform a single policy update for each estimate of the policy gradient that's derived. It would be nicei f we could perform multiple (or larger) updates -- but in practice, it often leads to destructively large policy updates!
- [[Trust Region Policy Optimization|TRPO]] aims to solve this problem
	- ==At each step of the optimization process, we find the *largest possible policy update that still improves performance!*==
	- TRPO thus allows us to learn faster by finding a reliable way to make larger policy updates that don't damage performance.

TRPO Formulation
- We update the policy under a constraint based on the [[KL Divergence]], which captures the distance between policies before and after the current update.
- Considering this constraint allows us to find a balance between the update size and hte amount of change to the underlying policy.

![[Pasted image 20240409172501.png]]
Above: The TRPO update rule
- Here, the [[Advantage Function]] is computed using some advantage estimate technique, like Generalized Advantage Estimation

Intuitively, this formula looks pretty similar to the Vanilla policy gradieint update rule, below:
![[Pasted image 20240409172624.png]]

There are some differences, though:
1. The terms in the expectation are modified slightly to express the probability of a given action $a$ as a ratio between old and updated policies.
2. The update has an added constraint based on the KL Divergence between the old and updated policy.
3. instead of performing *gradient ascent*, we're instead solving a constrained maximization problem to generate each new policy.

How do we compute TRPO's update in practice?
- We actually find an approximation to this equation that works quite well and can be computed efficiently!
![[Pasted image 20240409172748.png]]

Using TRPO in practice
- The implementation of TRPO is similar to that of VPG.
	- We allow our current policy to interact with the environment and collect data.
	- From this observed data, we compute the approximate update for TRPO as described above
	- We continue collecting data and performing updates until we arrive at a policy that performs quite well.

Because we are *using* the actual policy being trained to collect the data used ot train it, TRPO is an [[On-Policy]] RL algorithm.


On the KL Divergence constraint:
- What's the KL Divergence constraint's purpose in the TRPO update equation?
	- Recall: The VPG algorithm is based on gradient ascent, which ensures that updates to policy's parameters $\theta$ are not too large. In particular, we use a learning rate to perform updates with VPG, which controls the size of the update in the parameter space.
		- Small changes to $\theta$ can drastically alter the policy -- ensuring that policy updates are small in the parameter space doesn't provide much guarantee on changes to resulting policy -- as a result, we're constrained to very small updates.
	- TRPO sidesteps this issue by considering the size of our policy updates from an *alternative* viewpoint -- we compare the updated and old policies using the KL divergence (easuring the difference in probability distributions over action space produced by two policies).
		- ==This approach compares policies based on the actions they'd take, rather than by their underlying parameters $\theta$ .==
		- ==In this way, we can perform large policy updates while ensuring that the new policy doesn't produce *actions that are significantly different from the old policy!*==


### (2/2) Proximal Policy Optimization (PPO)
- TRPO has improved data efficiency, stability, and reliability compared to the VPG algorithm, but there are still limitations that need to be addressed!
	- The TRPO algorithm is complicated
	- The TRPO algorithm can only perform a single update each time new data is sampled from the environment
	- The TRPO algorithm is only applicable to certain problem setups.
- [[Proximal Policy Optimization|PPO]]: Another policy gradient algorithm that alternates between collecting data from the environment and performing several epochs of training over this sampled data.
	- Shares the reliability of TRPO, but is much simpler, data efficient, and generally applicable.

![[Pasted image 20240409173449.png]]
Above: Reformulation of the TRPO surrogate objective

Recall that during each policy update in TRPO, we maximized a surrogate objective to get the new policy -- this surrogate objective being solved by TRPO can be reformulated as shown above.

We simplify our originaly expectation over actions/states sampled from a policy with the subscript $t$ , which represents time steps along different trajectories sampled from the environment.

The objective has two terms:
1. The probability ratio
2. The advantage function

the expression for the probability ratio is as below:
![[Pasted image 20240409173616.png]]

If we were to maximize this objective without constraints, it would lead to a policy update that's *too large* -- this is why we leverage the KL divergence constraint!

The PPO surrogate objective
- Similar to TRPO, we perform policy updates according to a surrogate objective -- but this one has a"clipped" probability ratio as per the formula below:
![[Pasted image 20240409173659.png]]

The surrogate objective for PPO is expressed as a minimum of two values -- the first value is the same surrogate objective from TRPO, while the second is a "clipped" version of this objective that lives in a certain range.
- In practice, this means that there is no reward for moving the probability ratio beyond the interval of $[1 - \epsilon, 1 + \epsilon]$ 

==In other words, *PPO has no incentive for excessively large policy updates!==*
- By making the minimum of the clipped and unclipped versinoso f the surrogate objective, we only ignore excessive changes to the probability ratio if they *improve* the underlying objective.

While TRPO sets a hard constraint to avoid policy updates that are too large, PPO simply formulates the surrogate objective such that a penality is incurred if hte KL divergence is too large.
- This approach is much similar, as we no longer have to solve a difficult constrained optimization problem -- we can just compute PPO's surrogate loss with only minor tweaks to the VPG algorithm.

![[Pasted image 20240409174206.png]]

PPO is able to perform multiple epochs of optimization via stochastic gradient ascent over the surrogate objective, which improves data efficiency.

Computing the estimates of the advantage function (e.g. via GAE) typically requires that we learn a corresponding *value function* -- in PPO, we can train a joint network for policy and value functions by just adding an extra term to the loss function that computes the MSE between estimated and actual value function values:
![[Pasted image 20240409174405.png]]

# The Role of PPO in RLHF
- PPO was a useful advancement in mainstream RL research -- this algorithm also had a massive impact on the space of language modeling!
- [[InstructGPT]] was aligned via a three-part framework including both SFT and RLHF using PPO.
	- InstructGPT popularized this framework, and it was used for ChatGPT, GPT-4, LLaMA-2, and Sparrow.

How does this relate to PPO
- Due to its ease of use, PPO was the RL algorithm that was originaly selected for use in RLHF by InstructGPT.





# Takeaways


































