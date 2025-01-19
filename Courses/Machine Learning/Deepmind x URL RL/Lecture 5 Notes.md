https://www.youtube.com/watch?v=eaWfWoVUTEw&list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&index=5
## Model-Free Prediction
Back with my homie ==Hado van Hasselt==

-------------------


Last time, we covered planning by dynamic programming to solve known MDPs, but in this lecture we do model-free prediction to ==estimate== values in an unknown MDP! Next lectures, we'll talk about model-free ==control== to optimize values in an unknown MDP, function approximation, deep RL, off-policy learning, etc.

## Monte-Carlo Algorithms

- We can use samples to learn without a model
	- We call direct sampling of episodes [[Monte-Carlo]] sampling.
	- MC is [[Model-Free]]; we don't need knowledge of the MDP directly, we only need the ability to sample trajectories from it.

Let's look at the multi-armed bandit, where we were trying to optimize action values:

![[Pasted image 20250118200057.png]]

The true action value is given on the right side, which is the expected reward given an action; the estimates at timestep t is just the average of rewards given that you've taken the action.
We can update this incrementally too, using this $\alpha$ step size. If you choose $\alpha$ to be 1/N for that action, it's exactly equivalent to the first equation taking the "flat" average above.

- Note that we typically implement the timestep immediately after we've taken the action in RL, so we interpret rewards @ t+1 from action @ t arriving at the same time as our next observation s @ t+1.


Now let's extend this to also consider bandits with states!
- Episodes are still one-step long
- Actions still don't affect state transitions; these new states don't depend on actions
	- So there are multiple different states but they don't depend on your actions,  so there are no long-term consequences to take into account.

![[Pasted image 20250118200348.png]]
These are called [[Contextual Bandit]]s in the literature 

Now let's take an orthogonal step to briefly talk about function approximation

#### Function Approximation

Talking about [[Value Function]] approximation
- We've so far mostly considered lookup tables where every state (for v) or every state, action (for q) has their own entry in a table.
- The problem is that for large MDPs, there might be too many states and actions to store these effectively in memory, or it might take too long to learn the value of each of these states, if their values are being learned independently.
	- Furthermore, individual states are sometimes not even fully observable... 

![[Pasted image 20250118202138.png]]
With [[Function Approximation]], we hope to learn value functions (v or q) by updating some parameters $w$ from data; goal is to have a compact-ish number of parameters that can meaningfully generalize to yet-unvisited states.

Recall, agent-state update function:
![[Pasted image 20250118202231.png]]
- When the environment state isn't fully observable, we use an ==agent state==, which is a function of the previous state, action, and current observation of the environment (which is going to be a partial observation of the true environment, e.g. camera angle).
