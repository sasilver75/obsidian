![[Pasted image 20250111111250.png|400]]
1. True
2. True. Value Iteration is guaranteed to converge if gamma is < 1, but it can require more iterations. In policy iteration, there can only be A^S, since you only go through each policy once, but in value iteration there can be more! Consider a silly MDP where there's one state and one action -- there's only one state and one policy, so policy iteration takes one round. But Value iteration will keep going until the value function stops changing. In value iteration, r(s,a)= 1, gamma=0.9, and we initialize V_0(s) = 0. If you then use a geometric series, the V star of this state is 1/1-gamma, but after the first iteration of value iteration, v_1(s) = 1...

So even though both Value Iteration and Policy Iteration are guaranteed to converge to the same thing in the limit, they can have different behavior before then.

![[Pasted image 20250111112257.png]]

This class, let's talk about policy evaluation when we DON'T have a dynamics model (meaning we don't always know P(s'| s, a) for every action) like we did last time.

Let's get into policy evaluation!

## Evaluation through Direct Experience
- Estimate the expected ==return== of policy Pi, only using data from the environment (direct experience)
	- This is important: WE want to measure and make better policies!
		- Will be important for [[Deep Q-Learning]] and [[Policy Gradient]] algorithms that we'll cover later.
	- What properties do we want from policy evaluation algorithms?

We'll cover:
- [[Monte-Carlo]] Policy Evaluation
- [[Temporal Difference Learning]]
	- "[[Q-Learning]] is the control version of [[Temporal Difference Learning|TD-Learning]]."
- Certainty Equivalence with Dynamic Programming
- Batch policy evaluation

![[Pasted image 20250111112553.png]]
Above: [[Return]], [[Value Function|State-Value Function]], [[Q-Function|State-Action Value Function]]

## Recall: Dynamic Programming for Policy Evaluation
- When we do have acacess to the models... When someone gives us a function for the reward and function for the dynamics model.
- We do a [[Bellman Backup]] for a particular policy
	- (Diffrent from a [[Bellman Equation]] becaues ther's no max -- we just take whataever action is specified by the policy.)

![[Pasted image 20250111112741.png]]
(Above: For a deterministic policy; we'd need some additional average for a non-deterministic policy)

Bootstrapping: WE plug in one estimate to help us do another estimate...

In [[Monte-Carlo Policy Evaluation]] Policy Evaluation:
- We DON'T have a model of how the world works! so we just simulate and act in the real world instead:

![[Pasted image 20250111112908.png]]
(For most of today, assume Pi is deterministic!)
- Simple idea: The value function is an average over the returns that you get from following the policy from the state.
	- Therefore the value is just the means of the returns.
	- We do things a bunch of times and just average them.

Maybe if we want to know how good a treatment is in medicine; we have 100 people go through that treatment for a year, and then average their outcomes. We execute the policy for many different episodes, and then we just average the returns.

The trajectories may not all be the same length, that's fine -- just average over them -- and that's your state value function!

### Monte Carlo (MC) Policy Evaluation
- Does not require that your system is Markov; you're just doing trajectories and averaging them. In the MC Policy Evaluation, we just roll out the policy many times and average; we don't care.
- Can only be applied to ==episodic MDPs==, meaning you episode has to *end* in order for you to see what the total return is.
	- If you have horizons that last for a year, that's fine -- you can do that. But if you want to have episodes that last forever, you can't use MC Policy Evaluation.

![[Pasted image 20250111122318.png]]
N(s): Count of times we've updated our estimate for hte value of state s
G(s): Initialized to zero; we haven't seen any returns for this state.
- Increment the total return, counts of starts from this location, and update the state value estimate (as the average of the total return stemming from this point).

Example:
![[Pasted image 20250111122805.png]]

![[Pasted image 20250111122829.png]]
==Incremental== Monte Carlo policy evaluation: Having a running estimate and smoothly updating it for every case. 
![[Pasted image 20250111122931.png]]

![[Pasted image 20250111123657.png]]
We approximate these full expectations by taking a number of samples.
==We use a SAMPLE of the return to APPROXIMATE an expectation!== We do this many times, averaging over many such returns.
- This is good when the transition function isn't known, so we can't just integrate/sum over the possible states that we can end over, since we don't know p(s'|s,a).

Notice that MC isn't doing any form of bootstrapping -- there's no DP going on here. We're using samples of trajectories and their returns as means of evaluating our policy.

![[Pasted image 20250111123718.png]]
- Yes, the estimate should converge to the true value of the policy (law of large numbers).
- We'd like to know how quickly these things converge as we get more and more data.

![[Pasted image 20250111123947.png]]
As we get infinite data for our estimator, we'd like the difference between the estimator and the true estimator to be ~0.

![[Pasted image 20250111124006.png]]


![[Pasted image 20250111124307.png]]
In generally it's a high-variance estimator even though it's unbiased and consistent... but we have to wait until the end of the episode to update our estimate.
- WE might want to use the data that we already have in (eg) a long episode to update the behavior of the agent.
- If we have some evidence that our self-driving car isn't working well within a single episode, we might want to change how we do steering. We'll cover these later.

[[Monte-Carlo Policy Evaluation]] summary slide:
![[Pasted image 20250111124403.png]]
- Even if we know the true dynamics model, we still might want to do this!

q: "If we don't know the reawrd model, how do we calaculate the reward for hte trajectory?"
a: "We assume something is giving us these rewards. We might not have an explicit parametric representation of the reward model, but we ARE getting rewards. Maybe a customer is buying something or a basketball is making it into the hoop."

----------

[[Temporal Difference Learning]] (TD-Learning)
- Specifically, TD-Zero

![[Pasted image 20250111124443.png]]

A way to construct estimators for both control and policy evaluation. 
- The idea is to combine between ==sampling== to approximate expectations and ==bootstrapping== to approximate future returns!
- ==Model-free==, meaning you don't have to have a parametric representation of the dynamics model
- Can use in episodic settings or in infinite-horizon non-episodic settings.

Critically: ==We update our estimate of $V^{\pi}$ IMMEDIATELY after each (s,a,r,s') tuple== (Rather than at the end of the episode as we did in Monte-Carlo Policy Evaluation)

![[Pasted image 20250111124745.png]]
We're in a state, take an aciton, get a reward, and end up in a new state. Instead of playing that trajectory out to completion and using the Return, we instead just use the value function for the s' that we end up at as a way to short-circuit (sam's words) the entire playing-out of that trajectory.

So we don't have to wait, we can do this immediately, as soon as we reach s'!

![[Pasted image 20250111124950.png]]
We're taking our old estimate, and we're shifting it a little bit by our learning rate towards our target, which is our reward plus our discounted sum of future rewards, using our estimate of the rewards from s_t+1. 

![[Pasted image 20250111130243.png]]

![[Pasted image 20250111130253.png]]
We go back to our tree in which we expand possible futures
- TD updates its value estimate using a sample of $S_{t+1}$ to approximate an expectation...
	- In reality with Dynamic Programming, you'd want to do a weighted expectation of all states we can reach, but TD just samples one of those. That sample is an approximation of our expectation.
So TD does both sampling to approximate expectations as well as bootstraps by using our existing estimate of the value function.
- (The bottom equation is what we do in TD, the top is what we'd do in DP)

![[Pasted image 20250111130437.png]]
F
T
T
T

---------
## Certainty Equivalence

![[Pasted image 20250111131528.png]]
Once you have this model, a certainty equivalence model... Once we have this maximum likelihood MDP model, we can just compute the value of a policy using the same metrics we saw last week... because we have a dynamics model and a reward model.

One of the benefits of this
- It's really data efficient!

![[Pasted image 20250111131625.png]]
This computes the dynamics/reward model for all s,a and then tries to update all of them!
- It computes a value for every single state

Downside: Now we're doing policy evaluation with a full model, so it's computationally expensive but very data efficient, because as soon as you reach any state for which you get a positive reward, you can kind of propagate that to any other state that you know is possible to reach from there.

Q: "This seems similar to MC; is the difference just that we're estimating probabilities instead of rewards?"
A: "Its pretty different actually! we'll see in a second. WE use the data to compute models and propagate information but we'll end up making some interesting different decisions."

--------

## Batch Policy Evaluation

If I have some data, how do I best use the data that I have? This comes up often in the real world, when you're dealing with expensive-to-get data and you already have some.

Imagine we have a set of K episodes, and we want to do policy evaluation with that data

Given a set of K episodes
- Repeatedly sample an episode frmo K
- Apply Monte Carlo or TD(0) to the sampled episode

If we do this, what will MC and TD(0) converge to, in terms of the evaluation of the policy?

![[Pasted image 20250111132003.png]]
Two states example, Gamma = 1 so no discounting, with 98 episodes of experience.
- Started in A, got a reward of 0, transitioned to B, got a reward of zero.
- In other episodes, we started in B and got a reward of one.
- In one episode, we started in state B and got a reward of 0

Imagine running TD updates over data an infinite number of times.
What would V(B) be? 0.75.
What about Monte Carlo? Still, V(B) = .75 (6/8)

![[Pasted image 20250111132454.png]]
What about in V(A)? (Note these are consistent in a case with infinite data, but we have a finite amount of data here. Will they converge to the same thing?)
- TD V(A): .75 (Our V(B) is 0, even though in the situation where we start in A we don't actually get a reward!)
- Monte Carlo V(A): 0 (In the case where we start in A, we don't get any reward)





































