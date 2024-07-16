---

---
Link: https://www.tweag.io/blog/2019-10-25-mcmc-intro1/

-------

[[Markov Chain Monte Carlo]] (MCMC) is a powerful *class of methods* used to sample from probability distributions known only up to some unknown normalization constant.

Why would we want to sampling like this?
- When we're interested in the samples themselves
	- Like inferring unknown parameters in Bayesian inference
- If we need them to approximate expected values of functions w.r.t. to some probability distribution.

Sometimes only the *mode* of a probability distribution is of primary interest; in this case, it can be obtained by numerical optimization, so full sampling isn't necessary.

==It turns out that sampling from any but the most basic probability distributions is a difficult task.==
- Inverse transform sampling is an elementary method to sample from probability distributions, but it requires knowing the Cumulative Distribution Function, which normally requires knowledge of then normalization constant.
- Rejection Sampling doesn't require a normalized distribution, but efficiently implementing it requires a good deal of knowledge of the distribution of interest, and it suffers from the curse of dimensionality.

We want a smart way to obtain representative samples from your distribution in a way that *doesn't* require knowledge of the normalization constant!

These methods date back to the first MCMC algorithm, called the ==Metropolis Algorithm== to calculate the equation of a state of a two-dimensional system of hard spheres; they wanted a general method to calculate expected values occurring in statistical physics.

We'll covert he basics of MCMC sampling in this post, and later cover several important and increasingly complex and powerful MCMC algorithms.

# Markov Chains

- At the heart of MCMC are [[Markov Chain]]s.
	- Without all the technical details, ==a Markov chain is a random sequence of states, in which the probability of picking a certain state depends *ONLY* on the current state in the chain and not on the previous history; it's a memoryless process!==
	- A Markov chain, under certain conditions, has a unique stationary distribution of states to which it converges, after a certain number of states. From that number on, states in the Markov Chain are distributed according to the invariant distribution (aka the stationary distribution; one that remains unchanged as the chain progresses over time).

==In order to sample from a distribution $\pi(x)$ , a MCMC algorithm constructs and simulates a Markov chain whose stationary distribution is  $\pi(x)$ , meaning that, after some initial "burn-in" phase, the states of the Markov chain are distributed according to  $\pi(x)$ .==
- So we thus just have to store the states to obtain samples from  $\pi(x)$ 

The key quantity characterizing a Markov Chain is the ==transition operator== $T(x_{i+1}|x_i)$ , which gives you the probability of being in state $x_{i+1}$ at time $i+1$ given that the chain is in state $x_i$ at time $i$.

Now for fun, let's whip up a Markov Chain with a unique stationary distribution!

```python
import numpy as np
import matplotlib.pyplt as plt

plt.rcParms['figure.figsize'] = [10,6]
np.random.seed(42)

state = ("sunny", "cloudy", "rainy")

# In a discrete transition space, the transition operator can be described as a matrix. Columns are rows describe to sunny/cloudy/rainy weather, with rows indicating the state the chain is in, and columns the states the chain might transition to.
n_steps = 20000
states = [0]
for i in range(n_steps):
	# Choose the next state from 0,1,2, using the probabilities in the appropriate row on the transition matrix to choose which next state to move to.
    states.append(np.random.choice((0, 1, 2), p=transition_matrix[states[-1]]))
states = np.array(states)
```

We can monitor the convergence of our Markov chain to its stationary distribution by calculating the empirical probability for each of the states as a function of chain length:
![[Pasted image 20240616204149.png|450]]

## The Mother of all MCMC Algorithms: Metropolis Hastings
- Let's go back to sampling from some arbitrary probability distribution $\pi$. 
	- It could be discrete, in which case we keep talking about a transition matrix $T$
	- It could be continuous, in which case $T$ would be a *transition **kernel***.

From now on, we'll talk about continuous distributions, but all concepts here will transfer to the discrete case too.

We can split the transition kernel $T(x_{i+1}|x_i)$ into two parts:
- A ==proposal step==: Features a proposal distribution $q(x_{i+1}|x_i)$, from which we can sample possible next steps of the chain. We can choose this distribution arbitrarily, but we should strive to design it such that samples from it as little correlated with the current state as possible and have a good chance of being accepted in the acceptance step.
- An ==acceptance/rejection step==:It corrects for the error introduced by proposal states drawn from $q$. It involves calculating an acceptance probability $p_{acc}(x_{i+1}|x_i)$ and accepting the proposal $x_{i+1}$ with that probability as the next state in the chain.

Drawing the next state is then as follows:
1. A proposal state $x_{i+1}$ is drawn from the proposal distribution $q(x_{i+1}|x_i)$ 
2. That proposal is then accepted as the next state with probability $p_{acc}(x_{i+1}|x_i)$ , or rejected with probability 1 - $p_{acc}(x_{i+1}|x_i)$ , in which case the current state is copied as the next state.

Thus we have 
![[Pasted image 20240616205734.png|300]]

A sufficient condition for a Markov Chain to have $\pi$ as its stationary distribution is that the transition kernel needs to obeyed *detailed balance/microscopic reversibility, meaning that the probability of transition from state A -> B must be equal to the probability of the reverse process.
- Transition kernels of most MCMC algorithms satisfy this condition.




















