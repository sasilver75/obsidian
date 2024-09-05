https://www.youtube.com/watch?v=Psrhxy88zww&list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0&index=2

----

Recap of Lecture 1
![[Pasted image 20240706144617.png]]
- Given an MDP with states, actions probabilistic transition model, reward, discount factor, and horizon, find the optimal policy!
	- Value Iteration
	- Policy Iteration
- There are exact solution methods, but they have limitations -- they require having access to the dynamics model/transitions model P -- you need a model of the environment. (So let's look at sampling-based approximations where the agent collects experiences and learns from it.)
- Requires looping/iteration over all states and actions, which is impractical for large MDPs (we can't have a table of states and actions for a combinatorially large state! Let's try to use function approximation instead of tabular approaches!)


Agenda
- Q-Learning
- Deep Learning / Neural Networks refresher
- Deep Q Networks (DQN)
	- The approach used by DeepMind for the 2013 DeepMind Atari breakthrough

![[Pasted image 20240706145342.png]]
Q*(s, a) is the expected return (sum of discounted rewards) that an agent will accumulate if it starts in state s, commits to action a, and there-onwards acts optimally.
- Q-value iteration came down to essentially recursively computing values 
	- From Q0, we could find Q1, from Q1 we could find Q2, etc.


In (Tabular) Q Learning... we have to do these updates... every iteration visiting every state/action and computing an updated value.
- It assumes access to the transition model (problem) and the ability to iterate over every state/action (problem).
We can rewrite this as an expectation, where Q_k+1 is the expected value of instantaneous reward plus future rewards summarized in the Q value of the next state.
- ==Once we have an expectation, we can approximate expectations by sampling!==

[[Q-Learning]]
![[Pasted image 20240706150034.png]]
The new target comes from only one sample of reality (getting a reward, and then considering the Q value of the place we end up at)
We use an exponentially moving average, so we mix in new targets over time, and we still get Qs close to the actual expectation.

 ![[Pasted image 20240706150242.png]]
 Simple! :) 

But how do we sample actions, in the above example?
- We can choose the action that maximizes the Q_k(s,a) in the current state (Greedy)
	- This can work
- More popularly, we use [[Epsilon-Greedy]] sampling, where we select greedily, but with probability $\epsilon$ select a uniform random action.

![[Pasted image 20240706150505.png]]
We need to decay our LR otherwise we'll hope around too much with every update, and we won't converge. But we don't want to decrease it too quickly, or we won't learn enough.

![[Pasted image 20240706150558.png]]
The sum of the learning rates that you use over time has to sum to infinity (meaning if you start at any future time past zero, it still sums to infinity); you always have enough juice left in your LR to correct for bad past experiences
- But to make sure that *variance is bounded*, the sum of squares of the LR has to be bounded.

 ![[Pasted image 20240707002520.png]]
 We can't really scale tabular methods to enormous state spaces; we don't want to store tables this large.
![[Pasted image 20240707002540.png]]
It's just not practical to work with tables that have an entry for each (state, action).

What can we do instead of storing a table?
- Instead, we can have a ==*parametrized Q function* $Q_{\theta}$(s,a)== that learns, through parameters $\theta$ , to predict the Q value at a state, action.
- This function could be a linear function with some features that we determine (eg how far am I from the wall, what's my speed, etc.), or it could be a ==neural net== (most common in practice), decision tree, etc.
![[Pasted image 20240707002746.png]]
We determine our new idea of what the Q value should be at state (s,a) (I don't know why he calls it target of s') by taking a specific action, landing in a state, getting a reward, and then behaving optimally thereafter.
Then we update our parameters slightly (alpha) opposite the direction of the gradient of the objective function with respect to theta.

![[Pasted image 20240707004851.png]]
[[Deep Q-Networks|DQN]] training algorithm (from the DeepMind paper) with [[Experience Replay]]
- Recall that we have an experience from the agent (s,a,s') that we're going to use as part of our target. Instead of using it once, we're going to have a replay memory D that we'll use multiple times in our Q updates.
- See that we initialize *two* Q functions; two learning the ~same thing but slightly out of phase turns out to stabilize the learning.
- As our agent is acting in the Atari game, its' getting a sequence of frames as observation, and that sequence of frames is preprocessed into a stack frame phi1 for time1. So we work with these stack frames phi (since a single frame doesn't have enough information in it; we need velocities, etc.)
- See our epsilon-greedy action selection.
- Execute an action and receive a reward, and get our preprocessed phit+1.
- Store phit,at,rt,phit+1 in our replay buffer; that's an experience we can use to generate a target
	- Phi is basically our state (its a preprocessed state)
- Sample some random minibatch from transitions (A bunch of past experiences), and for each experience (sars'), we'll compute a target value (differently depending on if it's a terminal state), See that we use $\hat{Q}$ , which is what we use for the target Q valeus
- We bring the Q function we're learning closer to the targetevalues, ys.
- Periodically, we set the Qhat equal to the Q we're learning
	- The Qhat lags behind the Q that we're learning and choosing actions with; the reason this is done is to stabilize it. If the Q we're using to generate tanrgetsn changes too much, we have instability problems. We want our targest to come from this stabilized lagging function.
![[Pasted image 20240707010509.png]]
A huber loss is a parabola at the center, but then becomes linear. Any single target can only contribute so much with respect to how you update your weights in your neural network.
- We anneal the exploration rate $\epsilon$
- Use of [[RMSProp]] instead of Vanilla SGD.
![[Pasted image 20240707010629.png]]

![[Pasted image 20240707010710.png]]
When we take hte max over actions in our target calculation, our Q functions become overestimates because if randomly some action is initialized with a high Q value, it's favored in this max.
- We counterbalance by using one of the two Q networks to see which achieves the max, and use the other one to see what value *it has* for that action. So there's some independence between choosing of the action, and the effect (?). Helps stabilize learning, makes it faster.
- All DQNs these days are Double DQNs.


![[Pasted image 20240707010940.png]]
We have a buffer of past experiences; are each equally valuable?
- We keep track of the Bellman Error...  how different the target value for a sars' is different from what the q function predicts. If they're different, there's a lot to be learned, so you get a higher priority.
