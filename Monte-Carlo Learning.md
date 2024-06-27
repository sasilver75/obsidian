References:
- Video: [Mutual Information's Monte Carlo and Off-Policy Methods](https://youtu.be/bpUszPiWM7o?si=1KKXNTpHpd7U2T-w)
- Lecture: [[David Silver RL (4) - Model-Free Prediction]]
- Lecture: [[David Silver RL (5) Model-Free Control]]

A [[Model-Free]] method of RL in which we don't have knowledge of the environment.
- We don't know how many cards are in the deck, or how the dealer works.
- We don't know which direction the wind will blow us when we take an action.s

We merely get trajectories after running some policy through the MDP.
MCL uses averages to approximate expected values.

==Monte Carlo Evaluation==
- Given samples under $\pi$, estimate $q_{\pi}$. We can express our $q_{\pi}$ as $v_{\pi}$ estimation -- because evaluation treats a policy as fixed object, we can treat it as part of the environment and pretend there's *no policy,* and proceed with value estimation.
	- This lets us study our MRP problem as a [[Markov Reward Process]], which has trajectories like {$S_0, R_1, S_1, R_2, S_2, ..., R_T, S_T$}
	- If you come across a state-action estimation method, know that it can be used for value estimation.
- Our data in Monte Carlo evaluation is M trajectory samples of an MRP.
- The main idea of MC is to use *averages* to approximate $v_{\pi}(s)$ -- the expected Return at state $s$, under policy $\pi$.
	- ![[Pasted image 20240625172152.png|300]]
	- We sum over all M trajectories, at each timestep within the trajectories. The indicator function $\mathcal{I}$ selects out state $s$ whose value we want to estimate.
	- This whole thing is the sum of all returns in the data that follow state $s$, divided by $C(s)$, the number of times the state $s$ is visited.
	- This *estimate* of expected reward is often called $V(s)$
- Issue: To calculate this, we need to process a batch of M trajectories; it would be nice if we could apply an update rule after *each* trajectory, like this:
	- ![[Pasted image 20240625172250.png|300]]
	- Here, $C(s_m^t)$ counts all visit of the state up to time t, for the m'th trajectory.
	- We can change this to a constant between 0 and 1 -- replacing this with $\alpha$. 
		- This is called ==constant-alpha MC==, where $\alpha$ is called the step size. If $\alpha$ is small, the estimate will take larger to get to the right place, but is accurate when it gets there. If it's large, it will get there quickly, but it will be a noisy estimate.
- We initialize our V(s) for all s to some arbitrary value. We sample a trajectory, and apply our update rule. We do this iteratively.

