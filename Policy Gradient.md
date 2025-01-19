---
aliases:
  - PGM
  - Policy Gradient Method
---


References:
- Video: [Mutual Information's Policy Gradient Methods](https://youtu.be/e20EY4tFC_Q?si=1bTNBR08Qu9PqExQ)

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