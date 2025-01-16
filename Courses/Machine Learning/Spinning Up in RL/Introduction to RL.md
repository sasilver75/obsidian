https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

--------------
# Part 1: Key Concepts in RL
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

### Relationship between the optimal Q-function and the optimal Action
By definition, our Q*(s,a) gives the expected return for starting in state s, taking an action a, and then acting according to the optimal policy forever after.
- So how will that optimal policy act?
	- The optimal policy in any state will select the action that maximizes the expected return from starting in $s$, since that's the definition of $V*$.

So $a*(s) = \underset{a}{argmax}Q^*(s,a)$

Note that if there are multiple actions that maximize Q(s,a), we can randomly select any of them.

## Bellman Eqautions

All four of the value function above obey self-consistency equations called the [[Bellman Equation]]s.


The basic idea is this:
> ==The value of your starting point is the reward you expect to get from being there, plus the value of wherever you end up next.==

The Bellman-equations for the on-policy value functions are:

![[Pasted image 20250115161915.png|400]]
- For the state-value function: The value of being in a state $s$ and acting using policy $\pi$ is the expected reward of taking the action prescribed by the policy, followed by the discounted state-action value of the state that you end up in (as determined by some dynamics model).
- For the action-value function: The value of being in a state and taking an action is the immediate expected return of taking an action, ending up in $s'$, plus the action-probability-weighted Q-value under our policy.

The Bellman equations for the *optimal* value functions are:
![[Pasted image 20250115162558.png]]
The crucial difference between the Bellman equations for the on-policy value function and the optimal value functions is that here we've added the $max$ over actions.
- This reflects the fact that ==Whenever the agent gets to choose its action, in order to act optimally, it has to pick whichever action leads to the highest value.==

Note: The [[Bellman Backup]] for a state or a state-action pair is the ==right hand side== of the bellman equation -- it's the ==reward-plus-next-action==.

## Advantage functions
- Sometimes in RL we don't need to describe how good an action is in an absolute sense, but only how much better it is than others on average.
- ==An [[Advantage Function]] $A^\pi(s,a)$ corresponding to a policy $\pi$ describes how much better it is to take a specific action $a$ in state $s$, over *randomly* selecting an action according to $\pi(\cdot|s)$, assuming that you act accordingly to $\pi$ forever after in either case.==

$A^\pi = Q^\pi(s,a) - V^\pi(s)$
So it's the excess value (positive or negative) of taking a specific action at a state (and then following policy $\pi$) *over* the expected value of being in the state in the first place (and just following policy $\pi$ from then on).

Later, the ==advantage function will be crucially important to policy gradient methods!==


# Part 2: Kinds of RL Algorithms

![[Pasted image 20250115164013.png]]

To make something that fits on a page and is reasonably digestible in an introduction essay, we've omitted some advanced material (exploration, transfer learning, meta-learning, etc.).

Our goals here are:
- To highlight the most foundational design choices in deep RL algorithms about what to learn and how to learn it.
- To expose the trade-offs in those choices.
- To place a few prominent modern algorithms into context with respect to those choices.

### Model-Free vs Model-Based RL

- ==Whether the agent has access to (or learns) a model of the environment==, meaning ==a function which predicts state transitions and rewards==, is an important branching point.

- If we have a model of the environment, ==it allows the agent to plan by thinking ahead,== seeing what would happen for a range of possible choices, and explicitly deciding between its options.
- ==Agents can then distill the results from planning ahead into a learned policy.==
	- This is a form of [[Expert Iteration]]
	- This is what is done in in [[AlphaZero]]; when this works, it can result in a substantial improvement in sample efficiency over methods that *don't* have a model.

The main downside is that ==a ground-truth model of the environment is usually not available to the agent==, so if it *wants* to use a model in these situations, the agent ==has to learn the environment model purely from experience==, which has several challenges.
- Model learning is fundamentally hard and can fail to pay off.

Algorithms which use a model are called [[Model-Based]]
Algorithms which don't use a model are called [[Model-Free]]

While model-free methods forego the potential gains in sample efficiency from using a model, they tend to be ==easier to implement and tune==.

### What to learn

Another critical branching point is the question of **what to learn**:
- Policies, either stochastic or deterministic
- Action-Value Functions (Q Functions)
- Value Functions (V Functions)
- Environment Models

### ... in Model-Free RL:

There are two main approaches to representing and training agents:

##### Policy optimization
- Methods in this family represent a policy as $\pi_\theta(a|s)$; they optimize the parameters $\theta$ either directly by gradient ascent on the performance objective $J(\pi_\theta)$, or indirectly by maximizing local *approximations* of this $J(\pi_\theta)$ .
- This is ==almost always performed on-policy==, meaning that each update only uses data collected while acting according to the most recent version of the policy.
- Policy optimization also usually involves learning al approximator $V_{\phi}(s)$ for the on-policy value function $V^\pi(s)$, which gets used in figuring out how to update the policy.

Examples:
- A2C/A3C, which perform gradient ascent to directly maximize performance.
- [[Proximal Policy Optimization|PPO]], whose updates indirectly maximize performance by instead maximizing a *surrogate objective function* which gives a conservative estimate for how much $J(\pi_\theta)$ will change as a result for the update.

##### [[Q-Learning]]
- Methods in this family learn an approximator $Q_{\theta}(s,a)$ for the optimal action-value function $Q^*(s,a)$.
- Typically they use an objective based on the Bellman equation.
- This optimization is ==almost always performed off-policy==, meaning that each update can use data collected at any point during training, regardless of how the agent was choosing to explore the environment when the data was obtained.

The corresponding policy is obtained via the connection between $Q*$ and $\pi*$
- The actions taken by the Q-learning agent are thus given as:

$a(s) = \underset{a}{argmax} Q_\theta(s,a)$

Examples:
- [[Deep Q-Learning]], which launched the field of DeepRL
- C51, a variant that learns a distribution over return whose expectation is Q*.


==Tradeoffs between Policy Optimization and Q-Learning==
- Policy Optimization is nice because you're *directly optimizing for the thing you want,* which tends to make them ==stable and reliable==.
- Q-Learning methods only INDIRECTLY optimize for agent performance, by training $Q_{\theta}$ to satisfy a self-consistency equation. Many failure modes, so tends to be ==less stable==... but have the advantage of being ==more sample-efficient== when they do work, because they can be substantially more sample-efficient when they *do work*, because they can reuse data more effectively than policy optimization techniques.

##### Interpolating between Policy Optimization and Q-Learning
- There are a range of algorithms that live between the two extremes, carefully trading off betwen the strengths and weaknesses of both sides:
	- DDPG
	- SAC

### What to learn in Model-Based RL
- There aren't a small number of easy-to-define clusters of methods for model-based RL.
- We'll give a few examples, but the list if far from exhaustive.

Background: ==Pure Planning==
- To most basic approach ==never== explicitly represents the policy, and uses pure planning techniques like [[Model-Predictive Control]] (MPC) to select actions.
	- Each time the agent observes the environment, it compute a plan which is optimal with respect to the model, where the pan describes all actions to take over some fixed window of time after the present (with future rewards beyond the horizon being considered by the planning algorithm through some type of learned value function).
	- The agent then executes the first plan of the action and immediately discards the rest of it, computing a new plan each time it prepares to interact with the environment, to avoid using an action frmo a plan with a shorter-than-desired planning horizon.

==[[Expert Iteration]]==
- A straightforward follow-on to pure planning involves using and learning an explicit representation of the policy $\pi_\theta(a|s).$
- The agent uses a planning algorithm like [[Monte-Carlo Tree Search|MCTS]] in the model, generating ***candidate actions*** for the plan by sampling from its current policy.
- The planning algorithm produces an action which is better than what the policy alone would have produced, hence it is an "expert" relative to the policy.
	- Later, the policy is updated to produce an action more like the planning algorithm's output.
- [[AlphaZero]] is another example of this.


==Data Augmentation for Model-Free Methods==
- Use a model-free RL algorithm to train a policy or Q-function, but either:
1. Augment real experiences with fictitious ones in updating the agent
2. Use ONLY fictitious experience for updating the agent
Examples: MBVE and World Models

==Embedding Planning Loops into Policies==
- Another approach embeds the planning procedure directly into a policy as a subroutine, so that complete plans become side information to the policy, while training the output of the policy with any standard model-free algorithm.
Examples: I2A



# Part 3: Intro to Policy Optimization 

Here, we'll discuss the mathematical foundations of ==Policy Optimization== algorithms, connecting the material sample code.

We cover three results in the theory of [[Policy Gradient]]s:
- The ==simplest equation== describing the gradient of policy performance with respect to policy parameteres
- A rule which allows us to ==drop useless terms== frmo that expression
- A rule which allows us to ==add useful terms== to that expression

At the end we'll tie these together and describe the [[Advantage Function]]-based expression for the [[Policy Gradient]] -- the version we use in our Vanilla Policy Gradient implementation.

#### Deriving the simplest policy gradient


#### Implementing the simplest policy gradient



#### Expected Grad-Log-Prob Lemma


#### Don't Let the Past Distract You



### Implementing Reward-to-Go Policy Gradient



#### Baselines in Policy Gradients


#### Other forms of the policy gradient

