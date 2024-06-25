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
- We take our bellman equation we had before... The value of the root is given by a one-step lookahead, where we consider the actions we might take, and all the places we might end up, and we look at the value of those successor states. We back that all up, and sum it weighted by the probabilities of each leaf, and we get the value of the root.
- In Dynamic programming, we turn this equation into an iterative update!
	- We define the value function at the next iteration $v_{k+1}$ by plugging in the *previous* iteration $k$'s values at the leaves (take the value function v_k and plug in the values we had at the leads), and back those values up to compute a new value for the next iteration at the root.

![[Pasted image 20240624145929.png|400]]
Starting in some state and taking a random walk, which is the simplest policy we can think of. We want to know how long it's going to take (where a step gives -1 reward, so the number of steps = -G).
![[Pasted image 20240624150131.png|400]]
- We start with our $v_{k=0}$ estimate, which let's just say is 0s everywhere
- Next, we want to apply one step of iterative policy evaluation; we'll apply the bellman expectation equation to all states, and end up with a new value at each state
	- We look at each state in turn, and say: Whichever direction we go in, we get -1 reward, and then we look at the value according to our OLD estimate from $v_{k=0}$ (which said we'd get 0 everywhere), so we'll plug in a new value of (-1+0) = -1, and the same logic applies everywhere but in the terminal states (which stay 0). This resulting value function is our $v_{k=1}$ value function.
	- We iterate, and keep doing it! Look at the -1.7 function, and note that we're doing the Bellman *expectation* function, and not the Bellman *optimality* function -- so we consider all possible actions, with weighting based on the probability of taking that action.
![[Pasted image 20240624151205.png|300]]
As we increase k=inf, the value function converges to the true value function of our policy $v_{\pi}$ . So these numbers on the k=inf graph show the expectation of how many steps we're going to take in a random walk before hitting one of the terminal states (noting that reward for all states is -1 and gamma is 1, so there's no discounting)

This also tell us how to make a better policy:
- What if I were to pick actions, not randomly, but according to the actions in these grids? EG if we were to act greedily according to the values of reachable states?
- It turns out the value function helps us find better policies!

Any value function can be used to compute a better value function (if we define our policy as greedily going to better value states, it will improve).

## 3) Policy Iteration
([[Policy Iteration]])
- In 2), we were evaluating a fixed policy. Now in 3), we want to find the best possible policy in the MDP.
- Let's start with a given policy $\pi$ ... how can we find some new policy that we can say for certain is better than the one we had before?
	- If we had this process, we could apply it iteratively, and eventually find the best policy for our MDP.

Let's break this into two steps:
1. Evaluate the policy; figure out a value function for the policy, which tells us how much reward we'll get in expectation from any start state, given a policy in an MDP.
	- $v_{\pi}(s) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + ... | S_t=s]$ 
2. Improve the policy; We do this by acting greedily with respect to our value function $v_{\pi}$.
	- $\pi' = greedy(v_{\pi})$   ... We look ahead in each direction and pick the one that's best.

 Still, it's true that you can go around this process iteratively (evaluate, improve, evaluate, improve), and eventually this policy will always converge to the optimal policy $\pi^*$

![[Pasted image 20240624155001.png|300]]
We're going to use this idea of repeated improvement many times:
- We start with inputs to the process (here, an arbitrary value function of all zeros, and a policy)
- We start by evaluating the policy (on our up arrows), and then improving the policy (on our down arrows). The diagram illustrates that  that we converge to the optimal value function and policy.
We've seen one way of doing policy evaluation (bellman expectation expectation being iterated to find the value of a policy), and one way of doing policy improvement (Acting greedily with respect to our value function)

![[Pasted image 20240624155953.png]]
An example from Sutton and Barto
A car rental place with 2 locations, with a maximum of 20 cars at each. The rate of rental and return at each location is different; we ant to move cars in the optimal way overnight so that customers aren't met with an empty car park. So what's the optimal policy for shifting cars around so that we maximize our income?

We can formalize this as an MDP by considering the states as how many cars we have in the two locations, the actions are how many cars from -5...+5 we move from location A to B each night, the reward being $10/car rented, and the transitions being the actual return/request process (some poisson distribution characterizes this in the example)

![[Pasted image 20240624160249.png]]
Just looking at the policies to begin with; these 5 squares represent the policies we're considering.
Y axis: # of Cars at the first location
X axis: # of Cars at the second location
These diagrams indicate what the policy tells us the best thing to do is, at any state.
See how the policy changes.
At each state, we build up a value function that looks like the surface. We then act greedily according to that value function, giving us a new policy. We repeat this, and see how the policies change over iterations.
We can see that we're pretty much converging after our third policy.

(A more formal definition of what it means to do policy improvement)
![[Pasted image 20240624161531.png|300]]
We've shown so far that if we pick this greedy policy, the total reward G_pi of the policy is at least as much as the policy *before* we greedified it.
![[Pasted image 20240624161753.png|300]]
If this process of improvement ever stops, it must satisfies the Bellman Optimality Equation, and so it's an optimal policy.

Q: Is there some sort of intuition to explain why you can't stop at a local maximum?
A: Yes, there's a contraction argument, but it's a little hard to understand because it's a little formal. When you apply the Bellman equation to the value function, you bring the value function closer and closer together in value space ((?)) and ... uh... yeah, I'd need more time.

![[Pasted image 20240624162442.png]]
Above: Modified Policy Iteration
- The idea is just to stop early; eg having a stopping condition 

## 4) Value Iteration
([[Value Iteration]])
- Recall that an optimal policy can be subdivided into two components:
	- An optimal first action A_*
	- Followed by an optimal policy from successor state S'
- This gives us a theorem ==Principle of Optimality== for policies.
- ![[Pasted image 20240624163722.png|400]]
A policy is optimal if, for each state the wind might blow us to, that policy would behave optimally from that state onwards. This is the requirement for optimality
- (I think this is the same thing we said before where, when comparing two policies A and B, A >= B only if A >= B in all situations)

So we can use this to build a value iteration algorithm -- you can think of it as a backwards induction algorithm -- think of the value function as caching our solutions to all of our subproblems.
- The wind's gonna blow us to some state s', and we'll just assume we have the correct solution to that $v_*(s')$. 
	- The question, is how do we use that information to build an optimal value function in the previous step?
	- We just do a one-step lookahead tree, this time using the ==bellman *optimality* equation==. All we need to do is this one-step lookahead, look at the leaves of our tree, and back them up.
	- We start with the inductive premise, and bakc this up the tree, and maximize over all the things we might do (actions), giving us  the optimal at the root. Instead of someone telling us the optimal solution from s' v_*(s'), we start with an arbitrary value function, and iteratively improve it.
	- ![[Pasted image 20240624164117.png|400]]
	- The intuition is that we start off with the end of the problem; imagine someone tells us what the final reward is... and then work backwards, figuring out more of the optimal path, working backwards.
		- We'll just do this by looping over our entire state space (Which is nice because it works with loops in our MDPs, or with stochasticity... but the intuition of working backwards from the solution is still there).

So let's try to understand [[Value Iteration]] now:
- Problem: We want to find the optimal policy of some MDP $\pi$ (we're trying to do planning, still; we're not solving the full RL problem here; someone tells us the dynamics of the system -- the probabilities of ending up in next states, and immediate rewards we'll get).
- Solution: Given that someone's given us this MDP, we want to solve it by finding the optimal value function, and we're going to do it via iterative application of the Bellman Optimality backup (whereas before we were iterating the Bellman Expectation backup over and over).
- v1 -> v2 -> ... v*  (we iterate over whole sweeps of our state space, updating our value function everywhere in our state space)
- Using ==*synchronous* backups==: (at each iteration, we consider each state in turn, doing one-step look-aheads, using our previous iteration to seed values)
	- At each iteration k+1
	- For all states s in S
	- Update $v_{k+1}(s)$ from $v_k(s')$ 
- Unlike [[Policy Iteration]], there is no explicit policy being built at every step, we're just working directly in value space.
	- Before, we were alternative between value functions and policies, using value functions to build improved policies to get new value functions to get better policies. Here, we go from value function to value function to value function.
- Intermediate value functions may not correspond to any policy (they might not be achievable by any real policy). But at the end of the process, we know we have the optimal value function + policy.

![[Pasted image 20240624183844.png|400]]
The same slide we saw earlier for the Bellman Expectation iteration in value iteration, and it applies for the Bellman Optimality iteartion in policy iteration.
- We start with our old value function v_k, we put in our old value funciton on the leaves... and now, each state gets a turn to be the root in a diagram, and we do ones-step lookahead, and we maximize over all the things we might do... and take a maximum, backing it up to give us one new value in our new v_k+1. And we do this for all states.
- And then we restart the iteration for v_k+2, using v_k+1 at the leaves, and backing it up again.
We're turning our Bellman Optimality equation into an iterative update.
Q: Should there be a around the rest of the stuff after max for all a in A, in the second one?
A: Yes; it's max of the whole thing, over all As.


![[Pasted image 20240624184706.png]]
So there are different problems that we're trying to solve
- We've talked about synchronous dynamic program (complete sweeps, update everything)
- In all cases, the MDP is given, and we're trying to solve the MDP
- There are two types of Planning problems
	- Prediction: What's the value function for our policy? We use the Bellman Expectation Equation. We take the ==Bellman Expectation equation==, iterate it, and get v_pi
	- Control: Two ways of doing control (Policy Iteration, Value Iteration) which are two families of algorithms that address the problem of how to "solve" your MDP, getting v_* and pi_* 
		- In ==Policy Iteration==, we again use the ==Bellman Expectation Equation== to evaluate our policy, but alternate that process of evaluation with a process of policy improvement.
		- In ==Value Iteration==, we use the Bellman Optimality Equation, which tells us how v_* relates to itself; how the maximum reward that you can get out of the MDP recursive relates to itself. Iteratively applying this.
		- Between Policy Iteration and Value Iteration is a spectrum, which is the Modified Policy Iteration algorithm, which can recover the Value Iteration algorithm when k=1

In subsequent lectures, we'll apply similar ideas using the action-value function q(s,a)

## 5) Extensions to Dynamic Programming

![[Pasted image 20240624184901.png]]
Do you have to look at every state, and update every state in each iteration? No! That's often wasting a lot of computation
- In Asynchronous DP, we pick any state we want to be the root of our backup; we backup for that state, and then move on immediately, without having to wait until we've updated every single state.

![[Pasted image 20240624185027.png]]
These are all basically different ways of picking which states to update during an iteration.
- ==In-Place Dynamic Programming==
	- ![[Pasted image 20240624185050.png|300]]
	- You'd probably think of this yourself if you were sitting down to do DP; if you're doing synchronous evaluation, you kind of have to store two value functions at any given time (old value function for leaves, new value function at root).
	- The idea of in-place value iteration is just to forget about differentiating old and new; instead, we sweep over it, and in whatever order we visit states, we just immediately update our value function, and for any states we're yet to visit who will reference updated states, we'll just use the new value for that state.
- ==Prioritized Sweeping==
	- The idea is to come up with some sort of measure to determine how important it is to update any given state in your MDP.
	- You can update your states in any order you like, so which states to upate first?
	- Keep a priority queue, let's look at which states are better than others, and update in some order that depends on how important we consider given states to be.
	- ![[Pasted image 20240624185858.png]]
	- Most methods here use the Bellman Equation itself to figure out which states the most important! 
	- Intuition: If before my update, I thought the value of a state is zero, and after an update, I think the value of a state is 1,000, that will really effect my results. So we use the magnitude of the error between what we think before + after (∆ in the bellman equation)... we use that magnitude to guide the selection of states to do, and order them by the things that we think will change the most.
- ==Real-Time Dynamic Programming==
	- ![[Pasted image 20240624190208.png|300]]
	- The idea here is to select states that the agent actually visits! Don't sweep over everything naively; actually run an agent in the real world, collect samples, and update around those real samples!
		- If a robot is really wandering around a certain part of the room, we care most about those states, because that's what it's encountering under it's current policy -- not the opposite corner of the room.

![[Pasted image 20240624190354.png]]
Dynamic programming considers full-width backup, meaning we consider the whole branching factor of all states we might be taken to at a given step
- This is a very expensive process! And to do this look ahead, we need to know the dynamics of the system! We need to *know* where the wind might blow us!
- If future lectures, we'll learn how to solve this problem, and we do this by sampling; instead of considering the entire branching factor, we sample particular trajectories through this.
We'll consider *sample backups* in future lectures that look like this:
![[Pasted image 20240624190554.png]]
Starting in a state, sample an action based on our policy, sample one transition according to our environment dynamics, and do a backup just based on that one sample, instead of the full branching factor.
- This "breaks" the curse of dimensionality via sampling. Because we're sampling ,we also don't need to know the dynamics of the environment (we don't need to list all possible states an action will take us to; we just pick an action and see where we land).