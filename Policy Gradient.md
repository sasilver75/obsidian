---
aliases:
  - PGM
  - Policy Gradient Method
---


References:
- Video: [Mutual Information's Policy Gradient Methods](https://youtu.be/e20EY4tFC_Q?si=1bTNBR08Qu9PqExQ)
- Video: [RitvikMath's Policy Gradient](https://youtu.be/k5AnU_zFkac?si=xmEND6QFEYxzPHRo) 

-------

We'd like to determine a policy that would achieve high reward.
We're still in the world of [[Function Approximation]], where we have a hopelessly large state or state-action space. 
In the past, we made the assumption that nearby states have similar value function and learn to approximate our $Q_{\pi}(s,a)$ using some parametric model with parameters $w$. We then used this to determine a policy which we in an epsilon-greedy way.
But INSTEAD, with [[Policy Gradient]] methods, we 're going to go completely around these value estimates, selecting a parameter $\theta$ which directly determines our policy! It specifies the probability of every action in every state!
![[Pasted image 20250115114945.png|400]]

The [[REINFORCE]] algorithm is the first policy gradient algorithm we'll cover!
- It's a [[Monte-Carlo]] algorithm, waiting till end of the episode to make updates to our policy.
- Need to specify upfront:
	- Functional form of the policy (e.g. NN, Linear Model)
	- Initial $\theta$
	- Step size $\alpha$, which dictates how aggressively we apply our rule of updating the parameters in a direction that minimizes error. This is problem-dependent.
- For each episode:
	- Sample a trajectory of states, actions, and rewards until the episode terminates. This trajectory depends on the policy dictated by $\theta$
	- Apply some update to theta $\theta \leftarrow \theta + \alpha(???)$ 
		- Since we're adding it to theta, we can say that it's a vector of the same length as theta, and it should be something that makes high rewards more likely.

Let's say that we have a game world where we can only go left, right, and down.
We need a function that will map our state to a probability vector that has our probabilities for taking actions (left, down, right). We'll be mapping from a theta vector of length 2.

![[Pasted image 20250115115925.png|300]]

![[Pasted image 20250115120352.png]]

 The gradient is the direction to nudge thetas such that the probability of action a in state s is maximally increased. This is modulated by the return that we observe. Increase the probability of a positive-return action in proportion to the return. If it's negative, decrease the probability. 

But there are [problems] with this, so we use this:
![[Pasted image 20250115120614.png]]
this scales updates to account for the frequency of their application. 
This can be equivalently written like this:
![[Pasted image 20250115120638.png]]


![[Pasted image 20250115121410.png]]
Baseline function can be almost any function without breaking the algorithm's convergence function (but it *cannot* depend on the action). It's chosen to improve speed.
A natural choice for the baseline is an estimate of the value, aka the expected return of being in state s.
This makes our g_t = vhat positive for actions that generate returns above the expected value of being in state s, and negative otherwise
So we evaluate actions relative to what is expected in the state where they were taken.
![[Pasted image 20250115121601.png]]
Now let's bring it together
- The solution is to rewrite our update rule to this:
![[Pasted image 20250115121619.png]]
The $\nabla \ln{\pi(a_t^m|s_t^m, \theta)}$ is the direction to nudge theta to make an action more likely in a certain state, and so $(g_t^m - b(s_t^m))$ needs to be positive for good actions and negative for bad actions, where "good" and "bad" actions are judged relative to some baseline.

So for "REINFORCE with Baseline"
- We'll use a new update rule
- And we'll have to learn a value function for the baseline
![[Pasted image 20250115121804.png]]
- So we have two functional forms: One for the policy and one for the value function
- We have two step sizes to choose, and the choice for one impacts the best choice for the other

We need to discuss the ==Policy Gradient Theorem==
- What are we trying to do when selecting $\theta$?
	- In the episodic case, it's the $v_{\pi_\theta}(s_0) = \mathbb{E}[G_0|S_0 = s_0]$, meaning we want to optimize the value of the starting state.
	- We'd like to choose theta so that we get a high expected return, of course! We never have access to this, only estimates of it.

Theorem:
Tells us about the gradient of the state value with respect to theta:
![[Pasted image 20250115122358.png]]
- It says that it's *proportional* to something, which is:
- mu(s) gives us the probability of being in a particular state if we bounce around according to the policy. We take the average of {something} weighted by these state probabilities, and the {something} is a weighted sum over actions in a particular state, where the weights are the true action-values, and the things we're weighing are the gradients -- the direction to move theta if we want to make an action more likely in a given state. So think of this whole thing as the weighted sum direction to move theta if you want to make high-return actions more likely when in state s, and then the entire action is this averaged over states, weighed by the probability of being in each state! It's just a big weighted average of theta directions that make high-return actions maximally more likely. 

==If an algorithm approximates this direction, it will approximately optimize the objective!==
- Any algorithm that on average nudges theta in a direction that on average equals this theoretical direction, this will move us towards high-return policies.

Why is this remarkable?
- It only involves terms we can handle!
	- We can calculate $\nabla \pi(a|s, \theta)$ exactly
	- We can estimate $q_{\pi_\theta}(s,a)$ using the methods that we've seen
	- $\mu(s)$ will naturally get factored i, since we'll be applying updates as we bounce around the state space according to this distribution.

General comments on PGMs
- [[Policy Gradient|PGM]]s always involve *smoothly-evolving* action-probabilities. Value-based methods that use (eg) epsilon-greedy will have $\pi(a|s_0)$ that can abruptly jump around, because of the greediness. The PGM approach is much smoother.
	- Gives PGMs nicer convergence properties because they can always make small updates smoothly that help
	- Can be slow and inefficient
- It solves ONLY for the policy:
	- It's doing the least it can to solve the problem, which can be bad if it ignores relevant information
		- This might ignore relevant information (that could be useful later, like learning the dynamics model of an environment)
		- But it might also ignore irrelevant complexity (e.g. environment dynamics that have no impact on the model's best policy)
- Can learn stochastic policies
- It avoids needing to argmax over actions, which we have to do in our value-based methods. In situations with many actions, this can be a slow operation; this makes PGMs friendly for high-dimensional action spaces.

---------


Instead of trying to infer the policy from a learned Q function (like in Q-Learning), we instead try to *directly* learn a policy function from the data.
	- The goal is to learn a *P(s)*, and then sample some action: *a ~ P(s)*  
	- Below: 90% of the time, we're going to go the left, 10% stay, and 0% to the right; it's a probability distribution, which is very unlike Q-learning. Here, the outputs need to sum to one, unlike in Q-learning, where the outputs are arbitrary. While Q functions are naturally suited to discrete action, spaces... with Policy gradients, we're outputting a distribution, which can be interpreted as being continuous. This could be "How fast should I move, and in what direction (ie some positive/negative float)," in the Atari example below. The idea is that we can model a continuous action space; *P(a|s) = N(mu, sigma^2)*
	- Here's how we train policy gradients:
		- Initialize an agent
		- Run a policy until termination
		- Record all states, actions, and rewards (this becomes our mini-dataset)
		- Decrease the probability of actions that resulted in low reward
		- Increase the probability of actions that resulted in high reward

Limited by the fact that it can only perform a *single policy update* for each estimate of the policy gradient that is derived. It's notoriously data-inefficient -- we have to sample a lot of data when deriving a policy update.



![[Pasted image 20240625232923.png]]



--------

https://youtu.be/aQJP3Z2Ho8U?si=wmVyQWbWLsPd2nfc&t=2767

We can modify policy gradients slightly by changing the mean of the reward, in some sense;

![[Pasted image 20250118133114.png]]
This doesn't change the expected direction of the gradients, but it *can* change the variance of the updates!


--------

From RitvikMath's [video](https://youtu.be/k5AnU_zFkac?si=xmEND6QFEYxzPHRo)

![[Pasted image 20250118170444.png]]
- You're a manager at some sort
- "More" and "Less" are our actions
- Our State, p, is the change in demand for the product
- The rewards are a function of the state of the world and the action we take!
	- Rewards are positive if e.g.
		- Demand is growing and we order more of the product
		- Demand is shrinking as we order less of the product
	- Rewards are negative in the complement cases

We have everything we need to get started thinking about what it means to solve the problem.

![[Pasted image 20250118175307.png]]
Let's have a single parameter $\theta$ that we multiply with $p$ (our state) and pipe through a sigmoid $\sigma$ to get the probability of taking the MORE action.

$\pi(MORE|p,\theta) = \sigma(\theta p)$ 

We're mapping the state of the world to the probability of taking an action in the current timestep. This [[Policy]] is something that lets us look at the state of the world and say what we're going to do next.

How should we find the best policy?
- In normal ML, we're used to define some sort of loss, and then derive the parameter(s) of our policy with respect to the loss and move such to minimize the parameters.
- Here, we're going to set up some function $J(\theta)$ which is equal to the expected total reward of the policy: $\mathbb{E}_\theta$. Since this is a measure of the expected total reward, we want to *maximize* this; we do gradient ascent!
	- Note, this expected value is over all trajectories $\tau$ that result from following our policy $\pi_\theta$ 
	- These trajectories all have different probabilities and different individual rewards, so we can turn the expectation of the rewards into this summation that we see in the bottom of the image.

If we can find $\frac{\partial J}{\partial \theta}$ , then we can use gradient ascent to figure out the ideal policy!
But how do we get the expectation of expected total reward?

![[Pasted image 20250118180253.png]]
Above: The [[Policy Gradient]] theorem, finding the expected total reward
- Statement: The expected total reward is equal to the summation overall trajectories of the probability of the trajectory of the individual trajectory, times the derivative (now brought into) of the log of the probability of the trajectory times the total reward of the trajectory.

But whoah, where did that come from? :) It's a big old proof, but let's break it down
- If I take the first part of the term, $\pi_\theta(\tau)\frac{\partial}{\partial \theta}[log \pi_\theta(\tau)]$ 
	- We see in purple that we can simplify this a little bit
		- Appealing of the definition of the derivative of a log(x) is 1/x * x', where here x is $\pi_\theta(\tau)$.
		- So we can simplify this whole thing to $\frac{\partial}{\partial \theta}\pi_\theta(\tau)$ 
		- So if we then plug that in to replace the original term  $\pi_\theta(\tau)\frac{\partial}{\partial \theta}[log \pi_\theta(\tau)]$ 
		- Then you can see that we get something pretty similar to our original formula for $\frac{\partial J}{\partial \theta} = \frac{\partial}{\partial \theta}\mathbb{E_\theta}R(\tau) = \frac{\partial}{\partial \theta}\sum_\tau \pi_\theta(\tau)R(\tau)$ ... looks pretty similar.
			- Technically here, this derivative applies to both of these terms though!
				- \pi_\theta depends on theta
				- The returns of a trajectory depends on theta as well

But why are we introducing this log thing at the top at all if we're able to simplify it to something much simpler, as we showed just above?
- If we now expand what it means to calculate the probability of a trajectory under a policy characteized by theta... we will understand!

(See blue under purple in image)
What's the probability of a trajectory under a policy?
- There's a probability of an ainitial state S_0
- For each timestep, the action we take depends only on the state of the timestep.
- At every timestep, when we take an action from a state, we have some environment that will give us both a new reward and a new state, probabilistically. 

We do a trick that we often do in such situations, where we appeal to the log... so that we can turn our products into sums of logs!
Now things are going to get really awesome... because now we can take our derivative with respect to the log..

![[Pasted image 20250118181448.png]]

![[Pasted image 20250118181623.png]]
Above: $\mathbb{P}$ is just the policy

These crossed out ones don't depend on our policy.
If we had to care about these, then we would have had to model the environment!
So to optimize our policy, we don't actually have to care about the environment, in a sense!
The environment is notoriously difficult to model.
So it's an awesome property that makes these policfy gradient techniques useful in the real world -- they're [[Model-Free]]!

It reduces to this thing that we see at the end of the blue.

Recap:
- To find the derivative our of this expected reward function J with respect to our policy parameters Theta,
- Then by the policfy gradient theorem, it's going to be this $\pi_\theta(\tau)$, our $R(\tau)$, and finally we can substitute into where our derivative $\frac{\partial}{\partial\theta}[log \pi_\theta(\tau)]$ is, instead have the value $\sum_{t=0}^T \frac{\partial}{\partial \theta}log[\mathbb{P}(A_t|S_t]$ 
	- (I think his $\mathbb{P}$ is supposed to be his policy lol)

The first term $\pi_\theta(\tau)$ is probability of the trajectory, and the rest of the term is some measure of that trajectory, and so we put the measure on the inside and we can wrap the entire thing into an expected value.

Policy Gradient methods are popularly used in a method called [[Proximal Policy Optimization]], or PPO, which is based on this same theory that says: "If I want my agent to learn an optimal policy to solve some RL problem, it suffices to take the derivative of the expected reward with respect to the parameters that define my policy..." and doing so reduces to basically just taking the derivative of the log of our policy with respect to theta.