---
aliases:
  - Bellman Expectation Equation
  - Bellman Optimality Equation
---
(Note that Bellman Expectation and Bellman Optimality have different but related meanings; they aren't synonyms, but I'm using this note as a landing page for each.)

- References
	- NOTE: [[David Silver RL (2) - Markov Decision Processes]]
	- VIDEO: [Mutual Information: Bellman Equations, Dynamic Programming, Generalized Policy Iteration](https://www.youtube.com/watch?v=_j6pvGEchWU)


There are multiple Bellman Equations



![[Pasted image 20240623200200.png|400]]
(From the Mutual Information video) 
This is the expected return (which itself is the expected/average reward over numerous episodes, so a double expectation) under a policy, given that the starting state is s_0.
- We consider all actions... probability of action times the state action function of taking that action from our state.
- Note that G_t obeys a simple recursive relationship, where G_t is equal to the reward plus the once-discounted return a step later.
	- We do some algebra, and replace G_t+1 by plugging in the randomly-determined next state into the value function. This is sneaky, because they're not equal... but in expectation, they are!
	- From here, it gets easier; we have random variables, and we're trying to calculate an expectation... so by definition it's just a probability-weighted average of values -- and we can find that information in the "go right" distribution of our problem.
The bellman equation connects all state values! If we can solve for some state values, we can solve for others -- that's huge! And these are connections for *any policy;* let's now talk about *optimal policies.* In particular, the Bellman Optimality equation for state-value functions.


Bellman Optimality
![[Pasted image 20240623201417.png|300]]
- If we had the optimal action-value for going left and right, could you tell me the optimal state-value of our state? Yeah, it's just taking the maximum of action-value functions over available actions.
- This is the fundamental of Bellman Optimality; the agent must always select the highest action-value action, or it's not behaving optimally.


For any policy/states/actions...
- ![[Pasted image 20240623202011.png|400]]
- The value of a state $s$ when following a policy $\pi$ is defined as the policy-weighted average of the action values. If we had 100 states, this would provide 100 equations.
- The action value of a state-action pair is the probability-weighted average of the reward you'll get in the next step and the once-discounted value of the next state.
==These ARE NOT the Bellman Equations==, but can be used to build them.

![[Pasted image 20240623202151.png|300]]
![[Pasted image 20240623202205.png|300]]


For the Bellman Optimality equations, theres only a tweak or two to make:
![[Pasted image 20240623202052.png|300]]
Basically only one tweak: The state value must be the maximum overall all available actions.