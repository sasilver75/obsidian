https://www.youtube.com/watch?v=UoPei5o4fps&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=6

Today we're gonna start getting serious, and learn how to scale up RL to practical problems! By the end of today's class, we can go off and program interesting RL agents in the world that can solve problems.

See how we do policy evaluation and control, using function approximation to represent the value function.

Agenda:
1. Introduction
2. ==Incremental Methods== (We take a function approximator, like a NN, and incrementally, every step, we online take some step to update our value function)
3. ==Batch Methods== (More data-efficient methods that look at the history of things you've seen so far, and *exactly* fits your value function to things we've seen so far)

----


Large Scale RL
- We'd like to be able to use Reinforcement learning to solve LARGE problems:
	- Backgammon has 10^20 states
	- Computer Go has 10^170 states
	- Helicopters have a continuous state space; uncountably infinite states!
We can't just build a table anymore where we have a separate value for each state in our state space, anymore. We need a method that scales up and *work* -- methods that *generalize*; intuitively, the value of a state where you're at X, or one millimeter to the right of X should have very similar value functions.

In the next class, we'll look at other approaches for using function approximation for *policy-space* algorithms, rather than today's value-based algorithms.

We care, as usual, about:
- Prediction
- Control

![[Pasted image 20240627155233.png|500]]
So far, we've looked at these value functions V(s), where V represents some kind of lookup table; for every state (or every state, action pair, in a Q table), we have some value stored, and then we built algorithms to learn these tables to solve the MDP.
- For control, we picked our actions without using a model, in model free control, so we used an action-value function Q(s,a). This Q was sufficient to do control, because if we have it, we can immediately pick our actions my maximizing our Q(s,a) over all actions a available in state s. But this is again a giant 2-d table that we have to populate.

The problem is that we have too many states and actions to store in memory, and even if we *could*, we might not visit all the states in the table (and there's no generalization between cells). You need too much data!

The solution is value function approximation, where we represent the idea as:
- We consider the true value function (right) as a function mapping from s to the true value of $v_{\pi}$, and we try to build a function that estimates this thing *everywhere*
	- If we feed in any s, it gives us an estimate of $v_{\pi}$
	- We consider parametric function approximators (think: NNs), where we estimate our value function that fits this above function, across the state space.
We can do the same thing for the action-value function q(s,a), where we just consider it as a function that we want to fit that takes any (s,a) and gives us back an estimation q(s,a).

This will enable us to generalize from seen to unseen states!

Before we understand how to do MC or TD learning, let's focus on what it means to do funciton approximation.

![[Pasted image 20240627155903.png]]
Three different ways of doing function approximation (you can use a NN, but you can use any function approximator at all that you'd like.)
- If you're trying to do state-value function approximation, we pass in a state, and use some internal parameter vector(s) $w$. The model spits out a predicted value function $\hat{v}(s,w)$ for our state s.
- When we do action-value function approximation, we have two choices
	- "==Action in==": In this state here, I'm considering this action here -- how good would that be? The NN spits out this estimate $\hat{q}(s,a,w)$.
	- "==Action out==": Given just the state, the function approximator should produce the q estimate of *all* actions we can take.


![[Pasted image 20240627160300.png]]
There are a whole bunch of function approximators that we might try.
- How do we navigate this sea of function approximators?
- We're going to focus on differentiable function approximators, where it's relatively straightforward to adjust the approximator, because we know the gradient of the parameters with respect to the loss. We'll focus on linear combinations of features, and neural networks.

==Furthermore, we require a training method that is suitable for *non-stationary, non-iid data*! (eg our effective policy is going to be changing as we're learning).==
- In practice, we have a non-stationary sequence of value functions that we're trying to estimate as we estimate $\hat{v}_{\pi}(s,w)$ , as we learn different policies $\pi$.
- The data is IID, in the sense that we receive a sequence of SARs in a trajectory, where my location at 


## Incremental Methods

![[Pasted image 20240627160619.png]]
We're going to consider some differentiable function J, with respect to parameter w.
- We define the gradient vector to be the vector of partial derivatives of the function J with respect to our parameters, in turn. This tells us the direction of steepest ascent.
- We go *downhill* on this vector by some step size, adjusting our parameters in the direction to minimize our objective function.


![[Pasted image 20240627160908.png]]
What we're going to do is plug this into value function approximation.
- What if some oracle actually tells us $v_{\pi}$, like we're doing supervised learning. If we had this, then we can just minimize MSE over errors using gradient descent.
- To do gradient descent, all we need to do is move a little bet in the direction of the (error that we see in each state) multiplied by the gradient.
- The way to deal with this expectation $\mathbb{E}$ is to do [[Stochastic Gradient Descent]]; we're going to sample a state, see what the oracle says in that state $v_{\pi}$, look at our estimate, consider the error term, and then adjust our parameters (step size multiplied by error times gradient). The error tells us what we want to correct, and the gradient tells us how to correct it.

In expectation, if we visit all states, then we're going to arrive back at minimizing the MSE; even if you change things online in SGD, you still arrive at minimizing this loss term (as long as you have a reasonable step size).

==This is "cheating" though, because we're using an oracle that's just giving us $v_{\pi}$, which isn't realistic.==

![[Pasted image 20240627162810.png]]
Each element in this feature vector represents some aspect of our state (eg two meters from *this wall*, and two meters from *that wall*, etc.). If the features are good, it makes our learning problem much easier.

==Let's assume for now that someone gives us a good feature vector.==

![[Pasted image 20240627162922.png]]
How do we use these features in service of function approximation, then? The simplest idea is just a linear combination (weighted sum) of features, where we have some weight that tells us (eg) "How good it is to be a certain distance from *this wall,* or how good it is to be a certain distance from *that wall*," and we multiply these with our feature vector elements and sum.
- Equivalently, we dot product our feature vector with our weight vector.

Objective function is quadratic in parameters w (when using MSE as our objective function). This means there's some bowl or other quadratic shape representing basically our MSE; the objective function. This is an easy thing to optimize using standard optimization methods, eg gradient descent. One nice thing about using simple linear combination of features is that we never get stuck in some local optimum, we always find the global optimum if we follow our gradient long enough.


![[Pasted image 20240627163555.png]]
In an attempt to connect what we just talked about in our 
- We can build up a feature vector that's basically just a one-hot indicator function... An enormous feature vector with one feature for each state, and they'll all be zero except for the state that we're in now, which will have a one.
- If we use that as our feature vector, then when we take our dot product between feature vector and weights matrix, we see that we're just picking out a specific vector of our matrix; just one entry from our table. And now we've got just one entry for each state!


Okay, so we've cheated when we imagined the interest of this oracle giving us the true $v_{\pi}$ -- the whole point of RL is figuring this out directly from experience! So how do we do that
- We use the same fundamental methods from the last few lectures
	- [[Monte-Carlo Learning]]
	- [[Temporal Difference Learning]]
- To give us a target for a function approximator, instead of the oracle giving us that target.

![[Pasted image 20240627165006.png]]
- So far, we've assumed we were given $v_{\pi}(s)$, but we know that we can get targets that *estimate* that value function
- For [[Monte-Carlo Learning]], we use the Return $G_t$ as an estimator of $v_{\pi}$ . Wherever we saw $v_{\pi}$, we paste in our return $G_t$. So we adjust our weights a little in the direction between our *actual return* and the return predicted by our function app
- For [[Temporal Difference Learning|TD-Learning]],
- For [[TD-Lambda]],




## Batch Methods



















