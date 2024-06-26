A model is anything the agent uses to predict the environment's response to its actions.
- This might be an estimate of $p(s',r|s,a)$, the MDP distribution function.

Model-free methods are those without a model; they merely learn an association between states, actions and high rewards. Most of the time, we *don't* have a good model of our system, in reality.

The major dichotomy in model-free RL is between *==Gradient-based==* and *==Gradient-free==* methods.
- If we can parametrize our policy $\pi$ by some variables $\theta$, we might be able to take the the gradient of our reward/value function to these parameters directly, and speed up the optimization. Gradient-based is often fastest/most efficient method, but we don't always have a problem setup where we have gradients information available to us (if we're just playing Chess or Go).

Within the world of *==Gradient-free==* control, there's an idea of being [[On-Policy]] or [[Off-Policy]]. If we're playing a bunch of games of chess, we're trying to learn an optimal policy/value function by iteratively playing chess and refining our estimations of $\pi$ and $v_{\pi}$.
- ==On-Policy==: In on-policy, we're *always going to play our best game of chess.*
	- eg [[SARSA]], [[Temporal Difference Learning|TD-Learning]], [[Monte-Carlo Learning]]
- ==Off-Policy==: Maybe we'll try some things! Maybe the policy that I have right now is suboptimal, so I'll occasionally do (eg) random moves that might be really valuable for learning information about the system.
	- eg [[Q-Learning]]

Within the world of ==Gradient-Based== control, we're essentially updating the parameters of our policy, value function, or q-function directly by using some sort of gradient optimization.
- eg [[Policy Gradient]] Optimization

![[Pasted image 20240626000943.png]]

