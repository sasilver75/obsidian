A mathematical system that undergoes transitions from one state to another, on a state space, in a stochastic/random manner. The key property of a Markov Chain is that the probability of moving to the next state depends *only* on the current state, and *not* on the sequence of events that proceeded it --  i.e. it follows the [[Markov Assumption|Markov Property]]

Markov Chains are foundational in more complex methods, like [[Markov Chain Monte Carlo]] (MCMC) methods.
# How it works

- States
	-Distinct positions/conditions that the system can be in; the set of all possible states is referred to as the *state space* .
- Transitions
	- The process moves from one state to another, and these movements are called *transitions*. Each ==transition has a probability associated with it==, determining how likely it is to occur.
- Transition Matrix
	- The transition probabilities to any possible state, from any possibloe state, can be represented in a matrix form; each element in the matrix represents the probability of transitioning from one state to another.
