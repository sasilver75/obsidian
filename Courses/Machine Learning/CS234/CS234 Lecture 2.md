RL Lecture 2: https://youtu.be/gHdsUUGcBC0?si=fwH9MJ7IsjpNfaQc

Quiz: If gamma were 1, you would care about short term rewards exactly like you would long term rewards. If gamma were 0, you'd be entirely myopic and not care about future rewards at all.

Agenda:
- Making good decisions given an MDP
Next time:
- Policy evaluation when we don't have a model of how the world works

-------


Today: Given a model of the world
- A dynamics model (p(s'|s,a)) and a reward model

...

Return and Vale Function
- Definition of ==Horizon==
	- Number of time steps in  each episode
	- Can be infinite

Defninition of ==return== G_t: Discounted sum of rewards from time step t to horizon H
- G_t = r_t + gamma(r_t+1) + gamma^2(r_(t+1)) + ...
	- A particular series of rewards you might get

definition of ==State Value Funciton== (V(s)) for an MDP: Expected ==return== for an MRP startnig in state s
- V(S) = Exp(G_t|s_t=s)
	- On average, how much reawrd you'd get if you start in state s and act __


Computing the value of an infinite horizon MRP leverages the [[Markov Assumption|Markov Property]] (The future depends only on the present)

V(s) = R(S) + $\gamma \sum_{s' \in S}P(s'|s)V(s')$
- Equal the immediate reward plus the discounted sum of future rewards

A recursive decomposition of V(s)

Matrix Form of Bellman Equation for MRP
- For a finite state MRP, we can express V(S) using a matrix equatino

![[Pasted image 20250110162528.png]]
Now we can just invert this to solve for V

![[Pasted image 20250110162657.png]]
In general matrix inverses are fairly expensive... if the number of states you have is large, this can be very expensive. Also requires that the matrices are invertible.

In practice, we're usually dealing with state spaces that are far too large, and so we can't do this. So instead of doing it analytically, we'll just do it iteratively, avoiding tahat expensive matrix inverse!

We initialize the value of a state to 0 for all states in S

For k=1 until iterations:
- We make a new copy of our value function:

![[Pasted image 20250110162838.png]]
We do this over and over again until our value function stops changing
- The nice thing is that it's only S^2 for each iteration


![[Pasted image 20250110162927.png]]
We often say it's defined by this tuple (S,A,P,R,Gamma)

![[Pasted image 20250110163143.png]]
- So now our mars rover model has two dynamics models, one if we take a_1 and one if we take a_2. In this example, they're deterministic, but they could also be stochastic!
- Once we define the state space, action space, transition/dynamics model, rewards, and gamma, then you're defined your MDP.


Now let's start to think about ==Policies==: How do we make decisions depending on the state that we're in?
- We often think about policies as being ==stochastic==, but they can also be deterministic.
Policy: $\pi(a|s) = P(a_t=a|s_t=s)$

MDP + a Policy = a Markov Reward Process

![[Pasted image 20250110163422.png]]

So how do we evaluate a policy? How do we know whether it's good or not?
- We can plug in the actual policy that we'd use, and see how much reward we get.

![[Pasted image 20250110163500.png]]
- Value of s under policy pi: We consider the probability of taking each available action from our current state, under the current policy. For each of those action probabilities, we multiply it by the value of that state (which is the immediate reward of taking that action, and then the sum of the discounted probability-weighted states that we could land in, given that we took that action.)

This is a [[Bellman Backup]] for a particular policy; it specifies what the our expected discounted sum of future rewards are for a state if we follow that policy.

Soon, we won't just be interested in evaluating performance of a single policy, but finding an optimal policy! Note that optimal policies are not alway unique.

The optimal policy is one that maximizes the value of state s.
$\pi^*(s) = \underset{x}{argmax}V^{\pi}(s)$ 
The Optimal policy for an MDP in an infinite horizon policy is:
- Deterministic
- Stationary (does not depend on the time step)
- Unique? Not necessarily; may have policies with identical (optimal) values

## Policy Search
- One option is searching to compute best policy
- The number of deterministic policies is $|A|^{|S|}$
- We can evaluate all of them, but ==policy iteration== is generally more efficient!

Policy Iterate
- Alternate between having a candidate policy that might be optimal, then:
	- Evaluate it
	- Improve it
- Repeat

We initialize it randomly, so $\pi_0(s)$ is randomly initialized for all states s
- While not converged;
	- Evaluate the policy; determine $V^{\pi_i}$ for our current policy
	- Improve the policy; $\pi_{i+1}$ <- Policy improvement
	- i = i + 1

To do policy improvement, it's helpful to define the [[Q-Function]]
- The reawrd of hte immediate state and action
- Plus the sum of discounted future rewards if, after that action, we acted according to the policy.

![[Pasted image 20250110165732.png|500]]
Above: [[Q-Function]] is the "state action value of the policy"

((So it's kind of like the value function but it's conditioned on taking a specific action at the current state, rather than by taking a policy-weighted average over all current available actions?))
- i.e. "Take action a, then follow the policy \pi"

For each $s$ in $S$ and each $a$ in $A$:
$Q^{\pi_i}(s,a) = R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)V^{\pi_i}(s')$ 
- "Is there a relationship between the Q funciton and the value function, because they look similar?" "Yeah, we often call the Q function the ==state-action value function!=="

Compute new policy $\pi_{i+1}$ for all $s \in S$:
$\pi_{i+1}(s) = \underset{a}{argmax}Q^{\pi_i}(s,a) \forall s \in S$ 

![[Pasted image 20250110175631.png]]

Are we guaranteed to improve? Are there any formal properties? What can we say about it? 
Let's dive into what the policy improvement step is actually doing!

![[Pasted image 20250110175658.png]]
We take our old policy $\pi_i$ and compute the Q function for it.
- We think: What will the performance be of the new policy we extract?

We're going to be able to show that the Q function.... is better than the value of the old policy. (??)

![[Pasted image 20250110175824.png]]
So what is this saying?
- If you take your new policy $\pi_{i+1}$, defined as the argmax of the Q function in the second row... If you take it for one action and then follow Pi forever, then our expected sum of rewards should be AT LEAST AS GOOD as if we had always followed the only policy.

Every step of policy improvmement, we get a better and better policy for every state.
![[Pasted image 20250110181130.png]]
The only time we don't is if we've converged.

![[Pasted image 20250110181228.png]]
Proof of monotonic improvement in policy
"If we do policy evaluation where we compute the Q functions and take the max, you will always monotonically improve unless you stay the same."


![[Pasted image 20250110182237.png]]
[[Value Iteration]] is a different technique than [[Policy Iteration]]!
- At every iteration, we maintain the optimal value of starting in a state as if we only get to make a finite number of decisions.
Value iteration says: "Whats the optimal thing for me to do if I can just make one decision? I figure out what the optimal is (take one step here), and then I get to imagine that I can take TWO steps, and I build on what I know I can do for one step." So the interesting thing is that we always have an optimal value, but for the wrong horizon.

We keep going to longer and longer episodes with value iteration (H+1 steps, H+2 steps, building upon previous solutions using dynamic programming.)

Let's get into the [[Bellman Equation]]!
- For a particular policy, The Value function must satisfy the Bellman Equation:

![[Pasted image 20250110182604.png|400]]

The bellman backup operator... if we have a value function (BF(S))
If we take a max over all of the actions, and then get the reward for that action, and then use the given value function to get the discounted value of the next state... 

This yields a new vector of values function overa ll states s.

How do we do value iteration?
- We do this recursively, looping until convergence:

![[Pasted image 20250110182736.png]]
For policy iteration we kept going until our policy stopped changing; here, we keep going until our value function stops changing.

![[Pasted image 20250110183241.png]]

![[Pasted image 20250110183250.png]]

![[Pasted image 20250110184859.png]]








































































