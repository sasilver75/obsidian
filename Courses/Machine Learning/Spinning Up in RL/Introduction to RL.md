https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

--------------
## Part 1: Key Concepts in RL
- The main characters of RL are the ==agent== and the ==environment==.
	- The environment is the world that the agent lives in and interacts with.
	- At every step of interaction, the agent sees a (possibly partial) observation of the state of the world, and decides on an action to take. The environment changes when the agent acts on it, but may also change on it own.
	- The agent perceives a ==reward== signal from the environment; a number telling it how good or bad the current world state is.
	- The goal of the agent is to maximize its *cumulative reward* called the ==return==.

- A ==State== is a ***complete*** description of the state of the world. There is no information about the world which is hidden from the state.
- An ==observation== in a partial description of the state, which may omit information.
- In deep RL, we almost always represent states and observations by a real-valued vector, matrix, or higher-order tensor.
	- A visual observation might be represented by the RGB matrix of its pixel values, or the state of a robot might be represented by its joint angles and velocities.
- When the agent is able to observe the complete state of the environment, the environment is ==fully observed==; otherwise, we say the environment is ==partially-observed.==

> Note: Often times we talk about an agent choosing an action conditioned on a state; in practice, this action is conditioned on the observation, because the agent may not have access to the state.

- An ==action space== describes the set of all valid ==actions== in a given environment.
Some environments (eg Chess) have ==discrete== action spaces, while others, like robotic control, have ==continuous== action spaces (where actions are real-valued vectors, for instance).
	- This  matters! Some families of algorithms can only be applied to one case, and would have to be substantially reworked for the other.


- A ==policy== is a rule used by an agent to decide what actions to take. It can be ==deterministic==, in which case it is usually denoted by $a_t = \mu(s_t)$
- Or it may be ==stochastic==, in which case it's usually denoted by $\pi$: $a_t \sim \pi(\cdot|s_t)$ 

- Because the policy is essentially the agent's brain, it's not uncommon to substitute the word "policy" for "agent," e.g. saying "The policy is trying to maximize reward."

In Deep RL, we deal with parameterized policies: policies whose outputs are computable functions that depend on set of parameters (like the weights and biases in a neural network), which we can adjust to change the behavior via some optimization algorithm.
- These parameters are often denoted by $\theta$ or $\phi$.

The two most common types of stochastic policies in deep RL are:
- ==Categorial policies== (Can be used in discrete action spaces: A classifier over discrete actions)
- ==Diagonal Gaussian policies== (Used in continuous action spaces: HAs a NN that maps from observations to mean actions(?))

Two key computations are centrally important for training and using stochastic policies:
- ==Computing log likelihoods of particular actions==
- ==Sampling actions from the policy==

A [[Trajectory]] is a sequence of states and actions in the world.
$\tau = (s_0, a_0, s_1, a_1, ...)$

The very first state of the world $s_0$ may be randomly sampled from the ==start-state distribution==, sometimes denoted by $p_0$: i.e. $s_0 \sim p_0(\cdot)$

==State transitions== are governed by natural laws of the environment (==Dynamics Model==), and depend on only the most recent action $a_t$... they can either be entirely deterministic, or they can stochastic:
- Deterministic: $s_{t+1} = f(s_t, a_t)$
- Stochastic: $s_{t+1} \sim P(\cdot|s_t,a_t))$

The ==Reward Function== $R$ is critically important in reinforcement learning. It depends on the current state of the world, the action just taken, AND the next state of the world.

$r_t = R(s_t, a_t, s_{t+1})$

This is frequently simplified though to either just:
- A dependence on the current state: $r_t = R(s_t)$
- Or on the state-action pair: $r_t = R(s_t,a_t)$

The goal of the agent is to ==maximize some notion of cumulative reward== over a trajectory, but this actually can mean a few things.
- This cumulative reward is either ==discounted== or ==undiscounted==.

Why would we ever want a discount factor, though?
- We do, but the discount factor is both intuitively appealing and mathematically convenient.
- An infinite-horizon sum-of-rewards may not converge to a finite value, and is hard to deal with in equations. But with a discount factor and unreasonable conditions, the infinite sum converges.

==The goal in RL is to select a policy which maximizes expected return when the agent acts according to it.==

Let's suppose that both the environment transitions and the policy are ==stochastic==!

In this case, the probability of a T-step trajectory is:

$p(\tau|\pi) = p_0(s_0) \prod_{t=0}^{T-1}P(s_{t+1}|s_t, a_t)\pi(a_t|s_t)$
- Meaning the probability of the trajectory depends on:
	- The probability of starting where you start, using your start-state distribution: $p_0(s_0)$
	- Then, for every step in the trajectory:
		- The probability of taking a certain action given you policy.
		- times
		- The probability of landing in a successor state given the original state and action, given the environment dynamics model.

The expected return for this trajectory is then denoted by:

$J(\pi) = \int_\tau P(\tau|\pi)R(\tau) = \underset{\tau \sim \pi}{E}[R(\tau)]$ 
- Meaning:
	- The expected return of the policy is the average (over all trajectories) of the trajectory reward.

The central optimization problem is then expressed as:

$\pi^*$ = $\underset{\pi}argmax J(\pi)$

With $\pi^*$ being the optimal policy.

## Value Functions

It's often useful to know the ==value== of a state, meaning the ==expected return== of starting in a state or state-action pair and operating according to a particularly policy forever after.

==Value functions are used in almost every RL algorithm!==

There are four main functions of note here:

1. The ==On-Policy Value Function==: $V^\pi(s)$, which tells us the expected return from being in state $s$ and following policy $\pi$.
$V^\pi(s) = \underset{\tau \sim \pi}{E}[R(\tau|s_0=s)]$ 
In other words, the expected return of a trajectory starting at state $s$ and acting according to policy $\pi$ thereafter

2. The ==On-Policy Action-Value Function== (or "State-Action Value Function"): $Q^\pi(s,a)$, which tells us the expected return if you start in state $s$, take action $a$ , and continue according to the policy moving forward.
$Q^\pi(s,a) = \underset{\tau \sim \pi}{E}[R(\tau|s_0=s,a_0=a)]$
In other words, the expected return of a trajectory that starts in state $s$, takes action $a$, and continues acting according to policy $\pi$ thereafter.

4. The ==Optimal Value Function==: $V^*(s)$: The expected return of being in state s and acting optimally thereafter, in other words following the optimal policy $\pi^*$ thereafter.
$V^*(s) = \underset{\pi}{max} E_{\tau \sim \pi}[R(\tau)|s_0 = s]$
Here, $\underset{\pi}{max}$ means using the policy $\pi$ that gets the maximum expected reward, which is the optimal policy $\pi^*$


5. ==The Optimal Action-Value Function== (or "State-Action Value Function"): $Q^*(s,a)$: The expected return of being in state $s$, taking action $a$, and acting according to the optimal policy $\pi^*$ thereafter.
$Q^*(s,a) = \underset{\pi}{max}E_{\tau \sim \pi}[R(\tau | s_0=s, a_0=a)]$
Meaning that we want to find the policy that maximizes the expected return of trajectories that start in state $s$ and take action $a$.

### Relationship























## Part 2: Kinds of RL Algorithms
- 

# Part 3: Intro to Policy Optimization
- 