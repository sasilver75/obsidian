https://youtu.be/zSOMeug_i_M?si=wwBdvlB8BehXwzLO

Note: Lecture 2 notes were distributed amongst the various notes for topics that were covered. Wasn't sure if I wanted to continue with the course, but I do!

## Markov Decision Process and Dynamic Programming

------

Recall:
- RL is the science of an agent learning to make decisions in an environment
- Agents learn (any of):
	- Policy
	- Value Function
	- Model
- The general problem involves taking into account ==time== and ==consequences==
- Decision affect the ==reward==, ==agent state==, and ==environment state==

Last time we saw an agent acting in an environment with one state but multiple actions; let's now talk about problem with a full ==sequential structure==, and in which we assume the ==true model== is given.


![[Pasted image 20250118193341.png]]

![[Pasted image 20250118193349.png]]
[[Markov Decision Process]]
- (They're being really inconsistent lecture-to-lecture regarding their use of variables. For instance here we have $p$ representing the mode, which gives probabilities of next-states and rewards given actions and states.)

- See that from the given model $p$, we can  marginalize out/extract the state transition probability or expected reward individually.
	- First looks at the transition to the next state
	- Second looks at all of the rewards that we could expect from a transition starting from state s and taking an action a.
		- Takes into account all of of the states that we could end up in, and then the reward of being in each of those states.

Alternative definition of the MDP, which is quite common in the literature:
![[Pasted image 20250118193711.png]]
Sometimes you'll see $p$ and $r$ broken out into two separate functions.


![[Pasted image 20250118194002.png]]
[[Markov Assumption|Markov Property]]

![[Pasted image 20250118194534.png]]
MDP Example
- With probability $\alpha$ we get into a state of high battery from a state of high battery.
- With probability $1-\alpha$, we transition from a state of high battery into low battery, at which point we should recharge.

![[Pasted image 20250118194737.png]]
The full MDP table
- See that the MDP is a tuple of ${S,A, p, \gamma}$ 
	- The joint transition kernel $p$ can be read off from this table.



==THIS LECTURE KIND OF SUCKS, I'M GOING TO SKIP IT. ALSO 4 BECAUSE IT'S BULLSHIT==


