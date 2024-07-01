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


# Incremental Methods

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
- For [[Monte-Carlo Learning]], we use the Return $G_t$ as an estimator of $v_{\pi}$ . Wherever we saw $v_{\pi}$, we paste in our return $G_t$. So we adjust our weights a little in the direction between our *actual return* and the return predicted by our function approximator.
	- So I'm at a state, I use my function approximator that spits out a prediction, then we see what happens by running out the trajectory and getting some actual reward, and then updating our weights appropriately.
	- So it's like we're doing supervised learning on the real returns.
- For [[Temporal Difference Learning|TD-Learning]] (TD(0)), the target that we use as our estimate of $v_{\pi}$ is our TD target! Instead of waiting until the end, we start in a state, estimate that we'll get 10 units of reward, then we take a step (get some 1 reward), and estimate (with TD) that we'll get 8 more units of reward. So then our TD Target = 1+8 = 9 units of reward, and then, just like supervised learning, we treat that as our target and adjust our function approximator.
	- Just like supervised learning, we adjust our approximator to fit these predicted values.
	- Again stepping in the direction of the error between out TD Target and the prediction from our original state, multiplied by the gradient
- For [[TD-Lambda]], we do the same thing, but we use the Lambda Return, which is the thing that interpolates between TD(0) and Monte-Carlo learning.

![[Pasted image 20240627165601.png]]
So we an think of this as a process that looks a lot like supervised learning. Think of it as building training data; we see state S1, and we run a trajectory from it, and see we get a return of G1. Then we see S2, get a return of G2. Se see the final state, and the final return.
So the agent sees all of these states, and from each of those states it gets some return.
We treat this like a using our function approximator to predict the predict the $G_t$ and estimate what the $G_t$ will be from any $s_t$. 
Monte Carlo is about as close as you can get to an oracle (since they're real returns); this will work, but it will be pretty slow.


![[Pasted image 20240627170348.png]]
We can do the same idea with TD Learning, but now we have this biased sample; we don't have the true oracle, or a noisy oracle like in MC. 
- We're at a state, use our function approximator to predict expected return, and then {take a step, get a reward, and use our function approximator again on the next state, to estimate the reward for the rest of the trajectory}. 
- So this is biased, but we can still do the same idea, creating a dataset where, for each state, we have a TD target.
Q: Do you do this after every step, or every episode? 
A: In practice, we do everything incrementally, online updates; every single step, we'll take a step, generate our TD target, compute error, compute gradient, and update weights immediately, then move on to the next step.

Even though TD-Learning is biased, there's an important result showing that linear TD-learning (using linear function approximation) still converges close to the global optimum (where close depends on things like discount factor).


![[Pasted image 20240627170428.png]]
Finally, we can do the same thing with TD-Lambda, where the dataset we're generating is each state associated with a lambda return, which is a mixture of all of our n-step returns. We're going to try to learn to make this value function fit to these lambda returns. (Where each of these n-step returns involves prediction using our function approximator, for that state at the n'th step).
- There's also a way where you can use the Backwards View to have online updates. - 
- Briefly, the Eligibility traces...  accumulate credit for the things that happen most and the things that happen recently, and we decay these values at every step.
	- ((I'm not sure how we maintain an eligibility trace table for a large state space, since we're using function approximation to avoid doing that for our Q and V? I guess it's because we only need to keep track of visited states?... But what about continuous states?))

![[Pasted image 20240627171241.png]]
Let's move from ==Evaluation==  to ==Control==!
- We'll still build on our idea of Generalized Policy Iteration... but now we're going to use *Approximate policy evaluation!*
- We start with some parameter vector that defines some value function.
- We're going to act greedily (with some epsilon exploration, for example) with respect to that value function we've defined. This gives us a new policy.
- We want to evaluate that new policy, so we run through some trajectories while updating our neural network.
- We again play this out epsilon-greedily as a new policy, evaluate it, etc.

In the most incremental case, we're updating our policy every single step, and in Monte Carlo we can do it at the episode-level.

So does this really find the right answer? Does this really get to $q_*$? Of course not! We typically end up with algorithms that oscillate around it -- but in practice they tend to get very close to the right answer.


![[Pasted image 20240627171637.png]]
So we need to do all the same things using $q$ instead of $v$.
- We approximate the value function, so that for any state or action, we can use our parameters w to predict how much expected reward we'll get from that state onwards.
- WE'll minimize the MSE between the "true" q_{pi} and the function approximator, and use SGD to find a local minimum.
- ![[Pasted image 20240627171755.png]]
What does our feature look like? 
- Now we have a vector where every element is a combination of a state and action 
	- "How far away is that wall, and I'm moving forwards," "How far I am away that *this* wall, and I'm moving sideways".
	- This gives us a feature vector telling us about the whole combined state-action space.
The simplest function approximator is going to use a linear combination of features, but we could also use a neural network. As usual, we'll optimize it with SGD.

![[Pasted image 20240627172012.png]]
This looks very similar to our slide for the previous estimation of state-value functions.

![[Pasted image 20240627172040.png]]
This mountain car problem is perhaps one of the most used in reinforcement learning. 
- The idea is that you have a car stuck in a dip, and you can see that it's kind of steep!
- The idea is that your car isn't powerful enough to get straight to its goal, so it has to roll backwards, drive forwards, roll backwards, drive forwards, until it has enough momentum to reach the goal.
- So how do we figure out this control strategy in a model-free way?
State Space
- Position of car
- Velocity of car
Think of this as a two-dimensional state space, where the value function is a surface saying the expected reward of every specific state.
- Over time, as the policy improves, we see more and more shape emerging out of our function approximator using generalized policy improvement using the [[SARSA]] algorithm, where every single step we update our value function (in this case using a linear function approximator to update q; every step, we act greedily with respect to q to choose the next action, with epsilon exploration). We update q using one-step TD Returns.

![[Pasted image 20240627172934.png]]
If you use all the way of lambda=1, it's monte carlo, which takes too long
If you use lambda=0, it's TD zero, but you could benefit from more reality
Lambda can find you better than TD learning; bootstrapping really helps, though -- so it's useful to find algorithms that are effective when we bootstrap.


![[Pasted image 20240627173201.png]]
So off-policy learning can be a little bit problematic!
New (2016) methods like :
- Gradient TD
- Emphatic TD
Fix some of the issues that TD has when it bootstraps... and get checkmarks for the whole row for both on and off policy.

![[Pasted image 20240627173332.png]]

# Batch Methods

![[Pasted image 20240627173715.png]]
Our per-instance/per-step moves are simple and appealing, but they aren't really sample efficient - they're noisy methods.
The idea of Batch methods is to try to find the best fitting value function to *all* the data we've seen in our batch. The batch in our case is the agent's experience (its life; training data).

![[Pasted image 20240627173808.png]]
What does it mean to find the best fit? One definition might be the least-squares fit.
- If we want to fit our $\hat{v}$ to our $v_{\pi}$, we can have a "dataset" of experience $\mathcal{D}$ consisting of
- ![[Pasted image 20240627173854.png]]
- The question is then what's the best-fitting value function for the whole dtaset?
- We can use a [[Least-Squares]] algorithm that finds parameter vector w that minimizes our SSE between our vhat and target values.

![[Pasted image 20240627174418.png]]
We how do we minimize this MSE/SSE over our dataset that we've seen so far?
- There's an easy way, called [[Experience Replay]]; we make this dataset a literal thing, storing/caching this dataset of experience.
- We, at every timestep, sample a (state, value) from our experience, and then do one stochastic gradient update towards that target $v^{\pi}$. This is just like supervised learning, where we've got a dataset, and we randomly sample from the dataset and update towards our target.
- This takes our non-iiid date, our whole trajectory, and decorrelates it by breaking them into random pieces and presenting them in random order... until it converges to the [[Least-Squares]] solution!

![[Pasted image 20240627174435.png]]
In lecture 1, we had this motivating example of playing Atari games. Now we can understand it! The method is called [[Deep Q-Networks]] (DQN).
- It uses exactly what we've seen so far -- [[Experience Replay]] and basically [[Q-Learning]]; it's off-policy.
- Every step, we take an action epsilon-greedily according to our function approximator representing q. 
- We remember the trajectories we've seen so far in some replay memory.
- We sample some random mini-batches of transitions (s,a,r,s') from our $\mathcal{D}$ (say, 64), and follow the gradient with respect to those 64 things, and optimize the MSE  using SGD between what our Q-network is predicting (our action-value function) and our targets. The target are something we plug in instead of our oracle, and for us, it's a [[Q-Learning]] target (which are just like our SARSA targets, but with a max). 

This method is stable with neural networks! There are two tricks that make this stable compared to naive q-learning:
- [[Experience Replay]] stabilizes these NN methods because it decorrelates these trajectories by breaking apart the trajectories into steps by sampling/shuffling.
- The second idea is to basically have a second network (we keep two different Q networks; two different parameter vectors), and we basically freeze the old network for a while, and we basically bootstrap towards our frozen targets, not our latest, freshest targets.

![[Pasted image 20240627175317.png]]
They use a convolutional neural network to produce their feature vector, in Atari.

![[Pasted image 20240627175513.png]]
The simple idea of [[Stochastic Gradient Descent]] with [[Experience Replay]] is a way of squeezing the most out of the data that we've seen so far... but are there methods where we can jump to the least-squares solution?
- The answer is yes, for the special case of linear function approximation.
![[Pasted image 20240627175604.png]]
We can solve for the least-squares solution directly!

![[Pasted image 20240627175718.png]]
![[Pasted image 20240627175949.png]]
![[Pasted image 20240627175956.png]]


















