**Stanford** Reinforcement Learning
https://youtu.be/WsvFL-LjA6U?si=2sGxUfRzguhMdOuU

RL
Exploration
Delayed Consequences
Generalization
Where RL is powerful

![[Pasted image 20250109001632.png]]

----------

A [[Markov Reward Process]] (MRP) is a [[Markov Chain]] plus rewards

MRP:
- S: A finite set of states (s in S)
- P: a dynamics/transition model that specifies $P(s_{t+1} = s'|s_t = s$
- R: A reward function $R(s_t=s) = \mathbb{E}[r_t|s_t=s]$
- Discount factor $\gamma \in [0,1]$ 

Note: No actions
If finite number (N) of states, can express R as a vector

![[Pasted image 20250109011453.png|500]]
Above: rewards arae the +1 and +10 in the terminal states


Horizon (H):
- Number of time steps in each episode
- Could be infinite or finite
	- Otherwise called finite Markov reward process

==Return== ($G_t$) for an MRP:
- *Discounted* sum of ==rewards== from time step t to horizon H.
$G_t$ = $r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + .... + \gamma^{H-1} r_{t+H-1}$ 
(Above: The return is just the discounted sum of the series of rewards from current/future states)

==State Value Function== ($V(s)$) for an MRP:
- *Expected return* from starting in state $s$.
$V(s) = \mathbb{E}[G_t|s_t=s] = \mathbb{E}[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... +  \gamma^{H-1} r_{t+H-1} | s_t = s]$ 
(Above: The state value function is jus the expected return from that state)

Why use a discount factor?
- Nice to not sum to infinity (avoid infinite returns)
- Humans often act as if there's a discount factors.
- If episode lengths are always going to be finite, we can always just use $\gamma=1$ to not implement discounting.














