https://www.youtube.com/watch?v=AKbX1Zvo7r8&list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0&index=3

----

Recap:
- We looked what MDPs and exact solution methods were (e.g. value iteration)
- We looked at Deep Q Learning (and DQN) which could solve larger MDPs

Now: We'll look at some alternative methods for solving large MDPs using [[Policy Gradient]] and [[Advantage Function]] estimation.

![[Pasted image 20240708175415.png|400]]
The whole of this lecture is going to be pretty mathematical, unfortunately!
- Foundations of policy gradient methods through derivation
- Take advtange of temporal decomposition (TD) to be more data-efficient
- Looking at Baseline subtraction, which reduces the variance of our policy gradient method
- Value function estimation can further reduce our variance
- Advantage estimation can further improve our policy gradient, and bring them closer towards actor-critic methods.

---
# (1/5) Policy Gradient Derivation

In policy gradients, our agent is represented by a policy $\pi_{\theta}$ that outputs a distribution over actions, given a state.
Our policy is parametrized by a parameter vector $\theta$, which are the weights in the network, and we want to learn the set of weights that maximize expected reward.
![[Pasted image 20240708180451.png]]
Deterministic policies can be optimal, why should we consider stochastic policies that output a probability distribution? The answer is that it smooths out the optimization landscape! We want a smooth optimization surface to roll over!
- Also, as the agent is learning, it needs to collect data to learn from; stochasticity can help us *explore* the world.

![[Pasted image 20240708180633.png]]
- A simple state-value function V doesn't prescribe actions, and it's not clear what action to take if you don't have a dynamics/environment model to help you with transitions!
- A state-action function Q isn't always efficient to solve for the argmax in every state... 

So how do we compute our policy gradients?
![[Pasted image 20240708180742.png]]
The methodology: ==Likelihood Ratio Policy Gradient==
- We define $\tau$ as a trajectory sequence of state-actions.
- The reward for tau is the sum of rewards for state, action pairs.
- We want to optimize our utility by selecting the parameters theta that maximize our expected reward over all trajectories we could encounter.
	- A sum of rewards of trajectories, weighted by the probability of that trajectory... optimized by changing theta, which changes the probability of various trajectories by changing the policy, which outputs a distribution over actions, given state.

![[Pasted image 20240708182845.png]]
When we look at this... we have a sum (over the gradient?) over all trajectories, but it's not weighted by the probability... We really want a weighted sum, so that we sample from the distribution to compute these things.

So we multiply and divide by the probability of the trajectory, and reorganize...
Now we have a weighted sum!
![[Pasted image 20240708183121.png]]
This looks like [[Importance Sampling]] to me.
Now we can use a sample-based approximation (so we don't have to enumerate all trajectories)... we can take a sample-based estimate.

![[Pasted image 20240708183304.png]]
Note that $(\nabla p) / p$ can be rewritten as $\nabla log(P)$ , so we rewrite the above slightly
And then take an empirical estimate for m different sample paths ([[Monte-Carlo]]). For each rollout, we compute the grad log of the probability of the trajectory under teh current parameter setting, times the reward collected under that trajectory.

![[Pasted image 20240708183359.png]]
Interestingly, when we look at this, this equation can be used no matter what our reward function is -- because the gradient is with respect to theta, and our reward doesn't depend on theta, only on the trajectory that results from it.
- But how do we solve for the first term (grad log prob of trajectory) here? We'll look in a moment!

![[Pasted image 20240708183649.png]]
We collect some trajectories, and look at the grad log probability of these trajectories (and the reward).... what's gonna happen? A trajectory with high reward and high log probability will have a strong impact on the $\nabla U$ , whereas trajectories with high reward and low probability, or low reward and high probability, won't have as much of an effect. Because we promote high probability, high reward trajectories, we implicitly demote/dampen low probability/low reward trajectories.


# (2/5) Temporal Decomposition

![[Pasted image 20240708184154.png]]
- We can decompose paths into states and actions
	- A trajectory is a product of probabilities; the probability of next state, given current state and action, and the probability of an action, given a state.
	- Using the identity log(AB) = Log(A) + Log(B), we can get the second line
	- The gradient of a sum, is the sum of the gradients, so we can move the gradient inside the sums
		- We can drop the first term, because there's no theta in it (it won't have a gradient contribution)
	- W can move the gradient operator inside the summation because of the linearity property of derivatives, which states ***that the derivative/gradient of a sum is equation to the sum of the derivatives/gradients.***

Notice that there's no dynamics required; we just need to consider the $\pi_{\theta}$
- Interesting that we started with wanting to increase/decreases the log probability of a a trajectory that was experienced (where a trajectory consists of a policy and a dynamics model)... because the dynamics model doesn't include/relate to theta, all we're left with is a part with the policy
- To increase the P(trajectory), we increase the log probability of actions along that trajectory (and vice versa).

![[Pasted image 20240708184515.png]]
We now know that this first term can be computed as the sum of grad log probability of actions given state, along a trajectory. 
So now we can start computing gradients!
- We roll out our current policy
- We know the state, action, rewards along the trajectory (from the policy)
- We backpropagate for each state/action experienced; we compute the grad log probability of the action, given state... and then accumulate these over the trajectory, and then multiply by the reward of the trajectory.

So we can solve the above problem!

![[Pasted image 20240708184830.png]]
It's sample-based, so it's noisy.
We'll introduce some fixes:
1. Baseline
2. More temporal structure
3. Later: Trust Regions and Natural Gradients to further improve our updates.

So what was the intuition, again?
![[Pasted image 20240708184857.png]]
- We want to increase the probability of trajectories with high reward, and decrease the probabilities of trajectories with low rewards.
	- Really, intuitively, we want to increase the probability of paths that are above average, and decrease the probability of trajectories that are below average. 
	- ==Idea:== But if we're in an MDP with positive rewards only, then no matter how the trajectory performs, the above equation would have a contribution saying "let's increase the log probability of what I did" (sometimes by a little, sometimes by a lot). Really, we'd prefer to, for the bad ones, just bring the probabilities *down!*
Can we do this?
- ==Something called Baseline Subtraction will get us what we want!==

# (3/5) Baseline Subtraction

![[Pasted image 20240708185209.png]]
We introduced a Baseline b to our reward term.
- Things above average now have an increase in their probability
- Things below average now will have a decrease in their probability

Can we do this? Is it still an okay gradient estimate?
- If you work through the math on the left, you see that, on expectation, the extra term b will be zero (interesting -- why do we care about it, if it doesn't contribute? On expectation it is, but when we have finite samples, the estimate will actually have a reduction of variance effect... and you'll get a better gradient estimate (math not shown))

What else can we do?
![[Pasted image 20240708190217.png]]
We already did the temporal decomposition for the trajectory probability (left term, resulting in the $\nabla_{\theta}log\pi_{\theta}(u_t|s_t)$ term)
We can do the same thing for the reward term (right term)
And we should ask: Should all of these terms participate? Is it meaningful when I think about the grad log probability of an action given state... that it should be multiplied with a reward from the past? No! Actions I take now only influence the future.
- So we can split the right term into rewards from the part and rewards from the future.
- Rewards from the part are not relevant.  Indeed the expected value coming from rewards from the past is actually zero, so we can just remove it without a loss in accuracy.
![[Pasted image 20240708190516.png]]
We're left with this practical policy gradient equation
- We have a bunch of rollouts; m rollouts
- Each rollout has steps in it, up to H
- For each rollout, we accumulate the grad log probability of the action we took given the state at time t, and we multiply it with the reward from then onwards -- the action 

![[Pasted image 20240708191349.png]]
- Optimal Constant baseline: A minimum variance baseline that he hasn't actually seen people use in practice; rather than take an average, take a weighted average. When you average reward along a trajectory, you weight it by the square of the grad log probability. If you work through the math, this works out to a lower-variance policy gradient estimate.
- In a time-dependent baseline: Very popular; when we have finite-horizon rollouts. At a alter time there's less reward left than at an early time. This is a nice way to capture that
- State-dependent baselines:  for a specific state and time, how much reward do we still expect; this is basically a value state value function under the current policy.
	- Increase the logprob of action proportionally to how much betters its returns are than the expected return under the current policy.
	- But how do we get this value function to use as a basleine?

# (4/5) Value function Estimation
- There's many ways to estimate it!
- ![[Pasted image 20240708191825.png]]
- We could initialize a neural network parametrized by phi, and collect trajectories, and just do a regression against the empirical return (this becomes a supervised learning problem). This is often the natural thing you would try first. It's relatively simple, just a supervised learning problem.

But we could also do bootstrapping using the Bellman equations! In this case for policy evaluation.
![[Pasted image 20240708192122.png]]
We collect data, but it into a replay buffer, and do what's fitted value iteration:
- We have targets (reward plus value at next state)
- We have the NN we're currently fitting
- We do some gradient updates and maybe some regularization to keep our NN parameters phi close to the previous iteration's phi so we don't jump too far.
![[Pasted image 20240708192303.png]]
- For each timestep we compute the return from that time onwards
- And the advantage estimate, which is the different between that time onwards and the baseline: "How much is this better than average"
- We then do some learning
	- refit the baseline by having the NN that approximates the value function in this case.... minimize the difference between its predictions and the rewards we got during our rollouts.
	- Update our policy, using our policy gradient estimate

People often think of the bootstrap estimate as being less stable but more sample efficient
- So often get started with the Monte Carlo version, then try the bootstrap version

# (5/5) Advantage Estimation
- At this point, our policy gradient method has this term of grad log probability of action given state , multiplied with an advantage term (future rewards experienced minus the expected average return of that state).
	- Using the Value in the advantage thing is kind of interesting.... but we're still using a single sample estimate when using R(s,u) -- can we reduce the variance here?
![[Pasted image 20240708192726.png]]
We do it by discounting/function approximation.

![[Pasted image 20240708192735.png]]
Instead of using sum of future rewards, we use a discounted sum of future rewards.
- We're looking at the grad log probability of an action, given state, and seeing how good that action was.
- Think: That action might have more influence on things that are nearby in time, than those that come later... so maybe we shouldn't look at things in the future with equal weighting (though it IS true that sometimes things now can have future impact)
	- So the discounting is because the effect of actions *tends* to decay over time
		- ((This is kind of a pretty boneheaded prior. It's mostly true, but there's no modeling of the world/problem that goes into it))

Can we do more?
![[Pasted image 20240708193027.png]]
A3C uses one of the above estimates.... some mix of reality and value function estimations. The more reality, the less bias and the more variance (because reality is noisy). The less reality and the more estimation, the less variance but the more bias. Somewhere in between might be the best place for estimating advantage.

![[Pasted image 20240708193136.png]]
In GAE, they use a lambda exponentially-weighted average of all of the above!
- This is very similar to [[TD-Lambda]]/[[Eligibility Trace]]s

![[Pasted image 20240708193414.png]]








