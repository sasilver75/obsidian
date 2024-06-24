https://www.youtube.com/watch?v=Nd1-UUMVfz4&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=4
---


Last time we talked about formalizing the RL problem as MDPs, and now we'll talk about solving them.

One of the most fundamental/old ways of solving these types of problems is the idea of Dynamic Programming

- Introduction
- Policy Evaluation: If someone gives me a policy, how good is that policy?
- Policy Iteration: Use Policy Evaluation to build a method to find the optimal policy for an MDP
- Value Iteration: Works directly on value functions; not in the policy space. Makes the value function better and better by applying the Bellman Equation iteratively until we arrive at the optimal solution to the MDP.
- Extensions to Dynamic Programming

----

## 1) Introduction

What is Dynamic Programming?
- Dynamic: Having a sequential or temporal aspect/component to the problem
- Programming: Optimizing a "program," i.e. a policy. *Mathematical* programming (line linear programming). Here a program ~= a policy.
- Together, it's an optimization method for sequential problems! 

It's a method for solving complex problems by doing two things
- Breaks the problem into subproblems
- Solve for each of the subproblems, and then reassemble a solution

Problems need to have two properties for [[Dynamic Programming]] to be applicable:
- ==Optimal Substructure==
	- The *principle of optimality* applies; basically tells you that you can solve an overall problem by breaking it into (eg) two pieces, solving for each of those pieces... and that the optimal solution for those pieces will tell us the optimal solution for the overall problem. 
	- Canonical example: Shortest path from A -> B
- ==Overlapping Subproblems==
	- The subproblems which occur... occur again and again and again when solving a higher-level problem. Meaning that we have something to *gain* by solving a subproblem, and that we can *cache and reuse* our solutions to subproblems.
==Markov decision processes satisfy both of these properties!==
- The [[Bellman Equation]] gives recursive decomposition: It tells us how to break down the optimal value function into two pieces:
	- The optimal behavior for one step
	- The optimal behavior after that step
- The [[Value Function]] gives us the caching/reuse; we can think of the Value Function as a cache of all of the good information we've figured out about the MDP; It tells us for any state, we've figured out the solution (or rather, the maximum reward) for that state, onwards.


Planning (is what we're going to address today)
- ==Dynamic Programming assumes full knowledge of the MDP== (how the environment works). Someone tells us the transition/reward structure, and we want to solve the MDP.
- Two cases for planning in an MDP:
	- ==Planning to solve the prediction problem== (Policy Evaluation):
		- Given: an MDP(S,A,P,R,$\gamma$) and policy $\pi$
			- Or: MRP(S, P^pi, R^pi, $\gamma$)
		- Output: A value function $v_{\pi}$ that tells us how much reward we're going to get from any state in the MDP.
	- ==Planning to solve the control problem:== (The full optimization, when we're really trying to figure out the best thing to do in the MDP)
		- Given: MDP(S,A,P,R,$\gamma$) ... 
		- Output: Optimal value function $v_*$  (What's the best thing that people can do in the MDP? What's the best mapping from states to action that will achieve the best reward with the MDP), and, ultimately/equivalently, the optimal policy $\pi_*$  that tells us the action that gives the best reward for every state.

We'll start with prediction (figuring out how much reward we'll get from a given policy), and then use that in the inner loop of some optimization to figure out how to attack the control problem.


Dynamic Programming is widely used -- for much more than planning/MDPs
- Scheduling algorithms
- String algorithms (eg sequence alignment)
- Graphs algorithms (eg shortest path algorithms)
- Graphical models (eg Viterbi algorithms)
- Bioinformatics (eg lattice models)
These are all things where there's:
- Optimal substructure+Principal of optimality
- Overlapping subproblems


## 2) Policy Evaluation
- This is when someone tells you an MDP and tells you a Policy, and we want to learn how to estimate the reward of the model. This is a key piece in learning how to maximize the reward (basically, it's the inner loop).
- How do we do it? We use the [[Bellman Equation]]!
	- There are different flavors of the Bellman Equation
		- ==Bellman Expectation Equation== (here, for Policy Evaluation)
		- ==Bellman Optimality Equation== (later, for Control)
- Problem: Evaluate a given policy $\pi$
- Solution: Iterative application of the Bellman Expectation Equation. We turn it into a system for iterative update; we start some arbitrary initial value function $v_1$ (which tells us the value of all states in the MDP; we might start with 0 everywhere). Then we plug in our Bellman Equation and figure out a *new* value function $v_2$. We iterate this many times, and we end up with the true value function of the policy, $v_{\pi}$
	- v_1 -> v_2 -> .... -> $v_{\pi}$
- The way we do this is by using *==synchronous backups==*:
	- At each iteration k+1, we consider all states in our MDP (sweep over all states in each iteration).
	- We apply our iterative update to all states in our value function to produce a completely new value function for all states.
	- We'll discuss "*asynchronous backups*" later.
![[Pasted image 20240624145344.png|300]]
- We take our bellman equation we had before... The value of the root is given by a one-step lookahead, where we consider the actions we might take, and all the places we might end up, and we look at the valeu of those successor states. We back that all up, and sum it weighted by the probabilities of each leaf, and we get teh value of the root.
- In Dynamic programming, we 

## 3) Policy Iteration


## 4) Value Iteration


## 5) Extensions to Dynamic Programming

