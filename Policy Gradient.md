Instead of trying to infer the policy from a learned Q function (like in Q-Learning), we instead try to *directly* learn a policy function from the data.
	- The goal is to learn a *P(s)*, and then sample some action: *a ~ P(s)*  
	- Below: 90% of the time, we're going to go the left, 10% stay, and 0% to the right; it's a probability distribution, which is very unlike Q-learning. Here, the outputs need to sum to one, unlike in Q-learning, where the outputs are arbitrary. While Q functions are naturally suited to discrete action, spaces... with Policy gradients, we're outputting a distribution, which can be interpreted as being continuous. This could be "How fast should I move, and in what direction (ie some positive/negative float)," in the Atari example below. The idea is that we can model a continuous action space; *P(a|s) = N(mu, sigma^2)*
	- Here's how we train policy gradients:
		- Initialize an agent
		- Run a policy until termination
		- Record all states, actions, and rewards (this becomes our mini-dataset)
		- Decrease the probability of actions that resulted in low reward
		- Increase the probability of actions that resulted in high reward

Limited by the fact that it can only perform a *single policy update* for each estimate of the policy gradient that is derived. It's notoriously data-inefficient -- we have to sample a lot of data hwen deriving a policy update.