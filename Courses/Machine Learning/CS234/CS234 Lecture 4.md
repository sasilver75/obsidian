
This lecture covers [[Deep Q-Learning]], which came out in 2014 or so, and it was a big buzz at NeurIPS because [[DeepMind]] had learned to use it to play Atari games from pixels.

-----
![[Pasted image 20250112121927.png]]

Deterministic, since it's a max.

False. If our policy is greedily taking the action that maximizes Q(s,a) at every state s, then we won't get data about the other actions that don't maximize the Q(s,a), since our policy is deterministic and doesn't involve exploration -- we only ever take $\pi_{i+1}(s)$ in our state, which is deterministically one action. So we don't get information about other actions that are off-policy.

Agenda:
Last Time: Policy evaluation with no knowledge of how the wold works (MDP not given)
This time:
- Control (making decision) without a model of how the world works
- Q-Learning with a DNN --> [[Deep Q-Networks]]
Next time: Policy Gradient algorithms

![[Pasted image 20250114140129.png|350]]
Today:
- [[Monte-Carlo]] and [[Temporal Difference Learning|Temporal Difference]] -based control
- [[GLIE|Greedy in the Limit with Infinite Exploration]] (GLIE)

We're going to start by thinking about staying in the tabular land and trying to learn how to make optimal decisions in that case.

Table of Contents:
- [[Generalized Policy Iteration]]
- [[Monte-Carlo]] Control with Tabular Representations
- [[GLIE|Greedy in the Limit with Infinite Exploration]] (GLIE)
- [[Temporal Difference Learning|Temporal Difference]] methods for Control
- [[Policy Evaluation]]
- Monte Carlo Policy Evaluation
- Temporal Difference TD(0) Policy Evaluation
- Control using General Value Function Approximators
- [[Deep Q-Learning]]


----------

### Generalized Policy Iteration

### *Model-Free Policy Iteration*

If the policy is deterministic, we can't compute the $Q(s,a)$ for any action that is not the policy $\pi(s)$. And if we can't evaluate $Q(s,a)$ for every $a$ that isn't the $\underset{a}argmax Q(s,a)$ , then we can't really evaluate the policy.

We're going to be using an estimated Q, because we'll be estimating Q from data, directly from experience.

#### The problem of exploration
- We can only learn about the things we try in the world:

![[Pasted image 20250114140948.png|400]]

The downside of exploration is that we spend less time using our knowledge to make decision
- If we act randomly always, we might learn a lot about the world, but we don't spend a lot time using that knowledge to get a high reward during training.

This is the ==Exploration-Exploitation Tradeoff==

There are some deep questions around here about how we quantify our uncertainty in our knowledge, and how we propagate that to our decision making process downstream.

So how do we balance between exploration and exploitation?

One of the simplest things we could imagine doing is [[Epsilon-Greedy]], where we spend some of the time doing things *randomly* (with probability $\epsilon$, e.g. 0.01), and the rest of the time ($1 - \epsilon$) behaving in a way that you think is optimal.

![[Pasted image 20250114141351.png]]

This is a pretty simple strategy, but can be pretty effective!


Recall
- We proved that policy iteration using given dynamics and reward models was guaranteed to monotonically improve.
	- This proof assumed that policy improvement outputted a deterministic policy, though
- It turns out that epsilon-greedy policies also get this monotonic improvement!

We're either going to have Monte-Carlo approaches of Temporal Difference approaches, which more directly try to use the Bellman or Markov structures.
### Monte-Carlo Control

Recall:
![[Pasted image 20250114142022.png]]
- We sample states and actions under a specific policy. 
- We compute the return from each step until the end of the episode.
- For the first time we visit a (s,a) tuple, we update our Q value by a weighted average between our old estimate and the weighted target of the difference between the sum of rewards we ACTUALLY got and our predicted state-action reward).
	- This actual return sample is an unbiased estimator of the expected reward that we would actually be getting.

Now let's talk about ==Monte-Carlo Online Control!==
![[Pasted image 20250114142243.png|400]]
After I do an episode, I'm going to potentially change my policy. 
For each state s, the policy for s is doing to be $\pi(s) = \underset{a}{argmax}Q(s,a)$ with probability $1-\epsilon$, else $random(a)$.


![[Pasted image 20250114143727.png]]
We've got some decay in our epsilon, so it will converge to just $\pi$... 

It's helpful to think about how much time we're spending doing policy improvement versus policy evaluation
- This says that we spend one episode and then do policy evaluation
- Then we're going to change our policy
- And then play single episode, evaluating our policy 

Some of those games might be really long, while others might be really short. 
- We're just trying a policy out for ONE ROUND, and using that as a way to evaluate the policy. THat's just not very much data to do an evaluation... and the Q(s,a) we end up with are some ==weird weighted average== over all the policies that we've had and all the episodes that we've experienced under those policies.

The second thing: We're only doing *one rollout* to try to evaluate a policy; there might be a lot of stochasticity involved, so even with the same strategy you might get a bunch of different answers every time.

So we can't guarantee that this will converge to Q*

But there are some ways we can make thsi work!

![[Pasted image 20250114145814.png]]
If you can ensure that all state-action pairs are visited an infinite number of times *in the limit*... AND the behavior policy (the one we're using to act in the wold) converges to a greedy policy...
- See that the epsilon is reduced to zero with the following rate: $\epsilon_i = 1/i$
That's why we're being ==Greedy== ==in the limit== of ==infinite exploration== (you're being [[GLIE]])!

As long as you have an e-greedy strategy, you will visit all states and actions.

![[Pasted image 20250114150135.png]]

-----

Now let's look into Temporal Difference Methods for Control!

There are two types of algorithms that we'll focus on for TD for control:

The idea in these settings is that we alternate between:
- Policy Evaluation: Computing $Q^\pi$ using [[Temporal Difference Learning|Temporal Difference]] updating using an [[Epsilon-Greedy]] policy.
- Policy Improvement: Same as Monte Carlo policy improvement: Just set $\pi$ to $\epsilon$-greedy $Q^\pi$.

What are we trying to evaluate, and what are we improving with respect to?

[[SARSA]]
- State, Action, Reward, State', Action'
	- And these are the tuples we need to do an update!
- SARSA is an [[On-Policy]] algorithm
	- Meaning that it computes an estimate of the Q value of the policy that we're using to act/make decisions with in the world.

![[Pasted image 20250114153402.png]]
- We're going to iterate. Our loop is going to be such that:
	- We start in some state $a_t$ and take an action $s_t$, observe the reward $r_t$, end up in a next state $s_{t+1}$, and then we LOOP!
	- We take the next action according to the same policy. We update our Q function given our tuple (SARSA). 
	- This update looks similar to what we saw before:
		- $Q(s_t,a_t) = Q(s_t, a_t) + \alpha(r_t + \gamma Q(s+{t+1}, a_{t+1}) - Q(s_t, a_t))$
			- Note that $\alpha$ is the learning rate that tells us how much we update in the direction of the new error.
	- This looks somewhat similar to what we saw in TD(0), where we have a ==target== (here, $r_t + \gamma Q(s+{t+1}, a_{t+1})$ ) and an ==error== (here,  $(r_t + \gamma Q(s+{t+1}, a_{t+1}) - Q(s_t, a_t))$)
	- ==Note that we're plugging in the ACTUAL ACTION that we would take/took at the next state!==
	- "What is the expected discounted sum of rewards of Q(s,a)? One estimate is the immediate reward that I'd get, plus the discounted Q value of the action that I'd take in the state I reached". This is why it's on-policy.

Next, we do policy improvement:
- $\forall s$ , $\pi(s) = \underset{a}{argmax}Q(s,a)$ with probability $1-\epsilon$, and a random action otherwise.
- Note: We decay this epsilon between iterations, e.g. $\epsilon=  1/t$

![[Pasted image 20250114155310.png]]
- result is built on stochastic approximation
- Relies on step-size decreasing at the right rate
- Relies on the Bellman backup contraction property
- Relies on bounded rewards and value function
So it's not very clear that this should work, but it does (a bunch of beautiful papers out there from the 1990s)

Empirically, it often does quite well, but [[Q-Learning]] is more popular. 


----
![[Pasted image 20250114160255.png]]


now let's see if we can do [[Off-Policy]] learning, estimating and evaluating a policy using experience gathered from following a *different* policy. We'll still be in a [[Model-Free]] scenario, where we don't have access to reward/transition models.

So we're acting in a certain way and using that to estimate the value of and improve an alternative policy.

![[Pasted image 20250114160558.png]]
See that there are basically two changes in the Q-Learning algorithm:

![[Pasted image 20250114160642.png]]

In the bellman equation, we'd have something like 
$\sum_{s'}{p(s'|s,a)V^*(s')}$ 

But we don't have access to the transition model $p(s'|s,a)$ in a model-free scenario

In Q-Learning, we approximate that via this max:
$\underset{a'}{max}Q(s_{t+1}, a')$

Which is different from what SARSA does, because SARSA is on-policy and considers the actual action you would take:
$Q(s_{t+1}, a_{t+1)}$

Q-Learning doesn't care about the actual action you would take, it cares about the best thing you *could* have done there, because this gives a better estimate of the maximum discounted sum of rewards that you could get from that state in time.

So we get this algorithm for [[Q-Learning]]:

![[Pasted image 20250114161630.png|500]]
- We gather data under our current epsilon-greedy policy
- We use that data to update our Q value
- Because we don't know what the actual Q function is, we do this weighted approximation between our current estimate of the Q function and the target that we calculated.

==Even if you act completely randomly, you can learn Q* because in your Q* estimate, you're always doing this maxing. That's an important difference with SARSA, and it's what enables Q-learning to do off-policy learning==

==But over time, we want to become greedy over time to allow us to exploit our better-understood Q function. We decay our epsilon over time.==



-----

Now let's talk about 

# Model-Free Function Approximation
- Policy Evaluation
- Monte Carlo Policy Evaluation
- Temporal Difference TD(0) Policy Evaluation













 






























