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
- ==Categorial policies==
- ==Diagonal Gaussian policies==

- ==Categorical policies==
	- Can be used in discrete action spaces
	- A classifier (eg N) over a 
- ==Diagonal Gaussian Policies==
	- 

Two key computations are centrally important for training and using stochastic policies:
- Computing log likelihoods of particular actions
- Sampling actions from the policy





## Part 2: Kinds of RL Algorithms
- 

# Part 3: Intro to Policy Optimization
- 