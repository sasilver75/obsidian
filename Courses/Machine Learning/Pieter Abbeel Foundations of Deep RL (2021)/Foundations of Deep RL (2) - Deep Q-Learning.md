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

