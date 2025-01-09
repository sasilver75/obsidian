Link: https://arxiv.org/pdf/2412.05265
From [[Kevin Murphy]]
December 9, 2024


--------------------------------------------

# Chapter 1

## Sequential Decision Making 
[[Reinforcement Learning]] is a class of methods for solving various types of ==sequential decision making tasks== in which we want to design an ==agent== that interacts with an external ==environment==.
- The agent maintains an internal ==state== $s_t$ which it passes to its ==policy== $\pi$ to choose an ==action== $a_t = \pi(s_t)$.

### Problem Definition

The goal of the agent is to choose a policy $\pi$ so as to ==maximize the sum of expected rewards==.

$V_{\pi}(s_0) = \mathbb{E}_{p(a_0, s_1, a_1, ... a_T, s_T|s_0, \pi)}[\sum_{t=0}^TR(s_t, a_t)|s_0]$  

Where:
- $s_0$ is the agent's initial state
- $R(s_t, a_t)$ is the ==reward function== that the agent uses to measure the value of performing an action in a given state
- $V_{\pi}(s_0)$ is the ==value function== for policy $\pi$ evaluated as $s_0$

We define the optimal policy $\pi^*$ as:

$\pi^* = \underset{\pi}{argmax}\mathbb{E}_{p_0(s_0)}[V_\pi(s_0)]$ 

In other words, pick the policy that maximizes the expected value function of the initial state.
Picking a policy to maximize the sum of expected rewards is an instance of the maximum expected utility principle.

### Universal Model

A generic representation for the sequential decision making problems that we often use is a controlled [[Markov Chain|Markov Process]] with a hidden state which gets updated at each step in response to agents' actions.

To allow for non-deterministic dynamics, we write this as $z_{t+1} = W(z_t, a_t, \epsilon_t^z)$, where W is the environment's ==state transition function== (which is usually not known to the agent), and $\epsilon_t^z$ is random system noise.

The agent does not see the world state $z_t$ but instead sees a potentially noisy and/or ==partial observation== $o_{t+1} = O(z_t+\epsilon_{t+1}^o)$ at each step, where $\epsilon_{t+1}^o$ is random observation noise.
- For example, when navigating a maze, the agent may only see what is in front of it, rather than seeing everything in the world all at once -- or the current view might even be corrupted by sensor noise.

The agent uses these observations to incrementally update its own internal ==belief state== about the world, using a ==state update function== $s_{t+1} = SU(s_t, a_t, o_{t+1})$; this represents the agent's beliefs about the underlying world state $z_t$, as well as the unknown world model $W$ itself.

The agent can then pass its state to its ==policy== to pick actions, using $a_{t+1} = \pi(s_t+1)$.

We can further elaborate the behavior of the agent by breaking the state-update function intro two parts:
1. First, the agent predicts its own next state using a ==prediction function==: $s_{t+1|t} = P(s_t, a_t)$
2. Then, it updates this prediction given the observation using the ==update function==: $s_{t+1} = U(s_{t+1|t}, o_{t+1})$ 

The ==state update function SU is then defined as the composition of the predict and update functions==: $s_t+1 = SU(s_t, a_t, o_{t+1}) = U(P(s_t, a_t), o_{t+1})$ 

