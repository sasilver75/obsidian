Link: https://arxiv.org/pdf/2412.05265
From [[Kevin Murphy]]
December 9, 2024


--------------------------------------------

# Chapter 1

[[Reinforcement Learning]] is a class of methods for solving various types of ==sequential decision making tasks== in which we want to design an ==agent== that interacts with an external ==environment==.
- The agent maintains an internal ==state== $s_t$ which it passes to its ==policy== $\pi$ to choose an ==action== $a_t = \pi(s_t)$.

The goal of the agent is to choose a policy $\pi$ so as to ==maximize the sum of expected rewards==.

$V_{\pi}(s_0) = \mathbb{E}_{p(a_0, s_1, a_1, ... a_T, s_T|s_0, \pi)}[\sum_{t=0}^TR(s_t, a_t)|s_0]$  

Where:
- $s_0$ is the agent's initial state
- $R(s_t, a_t)$ is the ==reward function== that the agent uses to measure the value of performing an action in a given state
- $V_{\pi}(s_0)$ is the ==value function== for policy $\pi$ evaluated as $s_0$




