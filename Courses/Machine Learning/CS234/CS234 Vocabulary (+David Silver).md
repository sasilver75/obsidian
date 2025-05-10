[[Return]]

[[Reward]]

[[Horizon]]

[[Value Function|State-Value Function]]

[[Q-Function|State-Action Value Function]]

[[Bellman Equation]]

[[Bellman Backup]]

[[Value Iteration]]

[[Policy Iteration]]

[[Monte-Carlo Policy Evaluation]]
- First-Visit Monte Carlo
- Every-Visit Monte Carlo
+ Incremental MC Policy Evaluation

[[Temporal Difference Learning]] (TD-Learning)
- TD(0)/TD-Zero, or "1-Step TD Learning"
- TD-Error
- TD-Lambda (Forward)
	- Backward-View TD(Lambda), Eligibility Traces

Certainty Equivalence with Dynamic Programming

Batch Policy Evaluation

[[Generalized Policy Iteration]]

[[Epsilon-Greedy]]

[[GLIE|Greedy in the Limit with Infinite Exploration]] (GLIE)
- We need to make sure that we're exploring, but asymptotically, we want to make sure that we'ren ot exploring at all, because we want to find the best policy, and the best cooked policy is one that doesn't have any exploration.
- The idea with GLIe is to come up with a schedule for exploration, satisfying:
	1. All state-action pairs are explored infinitely-many times, guaranteeing that you never miss our on anything and that every little state, action in the state space seen.
	2. The policy eventually converges on a greedy policy. It needs to become greedy because we need this optimal policy to satisfy the bellman optimality equation.
- One way to achieve this is to choose an epsilon-greedy policy and decay epsilon towards zero over episodes.

[[SARSA]]

[[SARSA-Lambda]]

[[Off-Policy]], [[Behavior Policy]], [[Target Policy]]

[[Importance Sampling]]

-----------


How can you be model-free using the state-value function V(s) to act greedily during generalized policy iteration (GPI)? The policy for a given state would be to argmax over all actions for the state, taking the reward of the action plus the value of the state we would land in. But to know the state we would land in, we need the dynamics model. But we're model-free! So to work with the state-value function V(s), we need to have the environment model. The alternative is to use action-value functions Q(s,a), which enable us to do control in a model-free setting. If we have Q and we want to do greedy policy improvement, all we need to do is maximize over our Q values! IE $\pi(s) = \underset{a \in A}{argmax}Q(s,a)$. So every step we do a MC policy evaluation, run a bunch of episodes, take the mean from each s,a pair and get the mean return to estimate Q(s,a), telling us how good a state-action pair is... and then act greedily with respect to our Q as our new policy. We run that policy for more episodes, take the mean for the state-action pairs again, iterate, etc.

![[Pasted image 20250113102746.png|400]]

If you act greedily though, you can get stuck -- you might not ever see the states that you need to see to get a correct estimate of the true Q(s,a). If we act greedily to our policies, we might not see the full state space; we need to make sure that we continue to see everything!

So how do we make sure that we carry-on exploring? The simplest possible way is [[Epsilon-Greedy]] exploration; All actions $m$ re tried with some non-zero probability.
![[Pasted image 20250113103104.png]]
Flip a coin, and with probability epsilon you choose a random action... With probability 1-epsilon, you choose the greedy action. This guarantees that you explore everything (in the limit), and that you improve your policy.

![[Pasted image 20250113104722.png]]
This actually finds the optimal state-action function. You can let this loose and it will find the correct solution, the usual GPI without exploration.

![[Pasted image 20250113105441.png]]
Using TD instead of Monte Carlo 
- TD can be lower-variance, it can be used online (can be used in continuing domains with no terminations), and can be used for incomplete sequences (it becomes particularly important to use TD in off-policy scenarios).
Natural idea: Use TD instead of MC in our control loop:
- We know we have to use Q(S,A), since we're doing model-free and don't have access
- We know we have to use epsilon-greedy to see the whole state-action space.
- Because TD learning doesn't need to wait until the end of the episode to update our value function, we can do our policy improvement at every timestep!

Introducing: SARSA
- Given a state and action
- Sample from environment to see what reward we get and what S' we end up in
- Then we sample our own policy at the next state to see A'
![[Pasted image 20250113105737.png|400]]
We start with our Q value, and move our Q value a little bit in the direction of our [[TD Target]] (our reward plus the discounted Q value of the next state), minus the Q value of where we started.
- I'm in a state and considering an action. If I actually take an action and look at the rewaard I got, and then consider the value of the next action I would take, then this gives me an estimate of the value of this policy... and I use this estimate to update the Q value that we started in.

We plug these "SARSA updates" into our generalized policyim provement framework
![[Pasted image 20250113110040.png]]
Each time we take a step, we update our value function by applying one step of SARSA (for the S,A we were juts in, we only update the state-action value function (Q(S,A).). If I end up in that state-action pair again, I want to make sure that I use the most up-to-date best information. 
We act epsilon-greedily with respect to the latest state-action value function Q(S,A).

![[Pasted image 20250113110843.png]]

We can also do N-Step SARSA
![[Pasted image 20250113111817.png]]
No we're going to take the step we took in the last class, which is to consider the spectrum between Monte Carlo and TD algorithms. We consider eligibility traces and lambda variants of these algorithms to see if we can get teh best of both worlds:
- Unbiased behavior from Monte Carlo
- Have a knob that controls the [[Bias-Variance Tradeoff]].

We consider the N-step returns, again.
We take some number of actual steps before considering the value function to bootstrap the remainder of the trajectory.
So this n-step Q return can just be defined as the generalization of this to any N.
- So N-Step SARSA... instead of updating towards the one-step estimate of our value function using SARSA, we instead use the n-step return as our target.

Now let's make a lambda version of this algorithm!
- [[SARSA-Lambda]], a very well-known algorithm.
![[Pasted image 20250113112400.png|500]]
We're going to consider just the update -- what does this look like?
- We're going to average over all of our N-Step returns! 
We average over these different returns using the weighting scheme described on the right. We average over our n-step returns in a way that's computationally-convenient and takes account of all different n.
Lambda = 1: Monte-Carlo Learning
Lambda = 0: Familiar Sarsa

![[Pasted image 20250113120057.png]]
Above: [[SARSA-Lambda]]
- Initialize our eligibility traces to zero at the beginning of each episode
- For each step of the episode, we take our actions using our policy (we're [[On-Policy]] here, picking actions from our policy, e.g. acting epsilon-greedily)
- Compute our [[TD-Error]], looking at the difference between our reward and the value of the state/action that we end up in, compared to our current value (Q(S,A)).
- We increment our eligibility trace
- For every state/action pair (not just the one we visited now), we do an update:
	- We update everything is proportion to its eligibility and the TD error
	- We decay our eligibility trace for that state/action pair.

![[Pasted image 20250113120746.png|400]]
If you set lambda to 1, these will be of equal magnitude arrows. That's what happens in MC, because in MC, you run our trajectory, see you get some eventual reward, and everyone gets updated in the direction of that reward. By tuning your lambda, you basically determine ho far back along this trajectory you look; it controls your bias-variance tradeoff.

![[Pasted image 20250113122253.png]]
What about off-policy learning, where we want to follow some *==[[Behavior Policy]]==* (mu), which is the policy that we actually follow when choosing actions in the environment, but we want to use it to evaluate some other policy (pi), a [[Target Policy]].
- But the behavior that we see (mu) is not drawn from the thing that we're wondering about (pi).

Think about batch methods where you have a big dataset of trajectories from old/other policies. How do make use of this additional data to get a better estimate of the final policy?
- Yes, if we use off-policy learning!

The best known example of when off-policy is used is this third point: We know that big issue is making sure you explore the state space effectively, but we know that the optimal behavior doesn't explore at all; so why not make use of off-policy learning to have an exploratory policy that putters around exploring... and still at the same time learning an optimal policy! 

We might want to learn about multiple policies, too! What will happen if I leave this room, how much reward will I get? What will happen if I change the topic and go back to three slides ago. Learning about multiple policies from one stream of data.

We're gonna look at two mechanisms for dealign with this:

[[Importance Sampling]]
![[Pasted image 20250113123333.png]]
- The main idea is to take some expectation... and all we do in importance sampling is multiply and divide by some other distribution (Q(X))), and how this ratio that we've got... we can say that it's an expectation over some OTHER distribution (E_{X~Q})... of the remainder of the function... and this corrects for the change in your distributions.

We can apply importance sampling to Monte Carlo learning by doing importance sampling along the entire trajectory!
![[Pasted image 20250113123731.png]]
In Monte Caro, we have to look at the hwole trajectory, and we multiply these importance sampling ratios across the entire trajectory. At every step, there's some action I took according to by [[Behavior Policy]], and there's some policy that this action would have been taken by the policy I'm actually trying to learn about, [[Target Policy]].
- Eventually, this becomes a vanishingly small probability that the return that you saw under your behavior policy gives you much information at all about your target policy 😭
- So this idea is extremely high variance, and in practice USELESS! Monte-Carlo learning is a really bad idea Off-Policy; it just doesn't work! Over many steps, your target policy and behavior policy never match enough to be useful.

And so you have to use [[Temporal Difference Learning|TD-Learning]] when you're working off-policy, it becomes imperative to bootstrap! 
![[Pasted image 20250113124008.png|500]]
We only need to do importance sampling over one step now, because we're boostrapping after one step! So we just update our value function a little bit in the direction of your [[TD Target]], but your TD Target... (what happened over one step of this action), we just correct for the distribution difference over that one step!

So you still increase the variance, but it's better than Monte-Carlo.

But the thing that works best is [[Q-Learning]]... specific to TD(0) (e.g. Sarsa(0)).
- We consider a specific case wher we make use of our Q values to do off-policy learning in a specific way taht ==doesn't require importance sampling!==

![[Pasted image 20250113124507.png]]
Next action is chosen using behavior policy
But we also consider an alternative success action that we might have taken if we were following our target policy.
All we're going to do is update our Q value for the state we started in and the action took towards the value of our alternative action... we bootstrap from the value of our alternative action. This tells us how much action we would have gotten under our target policy.
- Our Q value is updated a little in the direction of the reward we actually saw... plus the discounted value of the next state of our alternative action under the TARGET policy.
- The reward R_t+1 is the reward that we got from taking A_t according to our bheavior policy.
In other words:
- We're in S_t and take action A_t following our behavior policy
- We get a reward R_t+1 and end up in state S_t+1
- Now instead of using the action next action we would take, we consider an ALTERNATIVE action (A') that would be taken under our target policy pi.
- We update our Q-value using this hypothetical "best" action rather than the action we actually take next.
Sam:
- Let the behavior policy take a step, and then pretend that the target policy was used to play out the remainder of the episode.

![[Pasted image 20250113125235.png]]
There's a special case of this, which is the well-known [[Q-Learning]] algorithm.
- This special case is when the target policy is a GREEDY policy. We'er trying to learn about greedy behavior while following some exploratory behavior.
- In Q-learning, both the behavior and the target policy can improve; we allow improvement steps to both of them.
	- At every step, we make the target policy GREEDY w.r.t. the state-action value function.
	- But our behavior policy is only [[Epsilon-Greedy]], so still exploring a bit as well.
- When we plug this in to the previous slide, we end up seeing:
	- For our alternate action picked by the Target policy (now picked greedily, see the argmax)... that's the same as the max of our Q values.
	- So we update a litttle bit in the direction of the maixmum Q value you could take.

That looks a little like this:
![[Pasted image 20250113125527.png]]

So just to wrap up:
![[Pasted image 20250113125536.png]]
 TD we can think of as samples of the bellman expectation/optimality equations; they do a one-step sample of what would happen if you were doing dynamic programming.

-----

# Lecture 6: Value Function Approximations
Outline:
1. Introduction
2. Incremental Methods
	- Take a function approximator like a NN, and every step, when you see some data as it comes in, you update your value function.
3. Batch Methods
	- More data-efficient methods that look at whole sets of data and try to fit your value function to things we've seen so far.

Large-Scale Reinforcement Learning
- Backgammon: !0^20 states
- Go: 10^170 states
- Helicopter: A Continuous space

==So the idea of having a V or Q table with an entry for every single state is just clearly not going to work for many realistic problems!==
- We'd like some methods that can avoid us having to compute and store all of these values for each state!
- We'd like some methods that generalize across the state space; intuitively our value functions should understand that the value of being at position x or the value of being a millimeter from position x should be quite similar, and we want our value function approximators to understand that.

We'd like to understand how to achieve this generalization and have efficient methods for representing and learning ==value functions== in RL
- In a next class, we'll look at methods for function approximation for ==policy== space algorithms. But today we'll focus on Value Functions.

In particular, we want to know how to do:
- ==Prediction==/How to Evaluate a Policy
- ==Control==: Finding an Optimal Policy

![[Pasted image 20250114225628.png]]
- So far we've looked at examples where every state had its own entry in a V(s) vector/lookup-table (or, alternatively, a Q table, in model-free control, since we didn't have access to the transition/dynamics model).
- The problem is that there are too many states (or state, actions) to store in memory, and even if we could store it in memory, there's too much to reasonably explore!

We're going to consider the true value function as just some function that maps from s to the true value of $v_{\pi}$, so if you feed in any $s$, it will give you an estimate of $v_{\pi}$.
These parametric estimators have some vector of weights $w$ (e.g. the weights of a neural network), and we estimate the value function using a combination of them with the state.

Ideally, using a small number of parameters we can fit/estimate the value function everywhere in a much larger state space (so it's more ==compact in memory== and has ==generalizability==)

We could do this for the true $q_{\pi}(s,a)$ too, by approximating with $\hat{q}(s,a,w)$.

First, let's talk about what it means to do function approximation using a value function:

![[Pasted image 20250114230953.png]]
These are like three different architectures you can use.
- A NN is a canonical black-box function approximator, but you can use any architecture you'd like.
- We use some sort of internal parameter vector $w$ -- that spits out the value function at the query state $s$, which is $\hat{v}(s,a)$.
- When we do action-value function approximation, we have two cases!
	- "==Action in==": Given (s,a), spit out predicted value for that
	- "==Action out==": Sometimes it's more efficient to use a different form where we plug in the state and in a single forward pass the model spits out the values of *all* the actions we could take.
		- In a single forward pass, you get all the information you need to make a decision.
		- This was used by DeepMind in their Atari project.

Can use any sort of algorithm you want:
- ==Liner combination of features==
- ==NNs==
- Decision trees
- Nearest Neighbor
- Fourier/wavelet bases

In this course, we'll focus on the ==differentiable== function approximators above that we can optimize using gradient-based methods.

What's special about RL rather than regular Supervised Learning is that we have ==non-stationary sequence of value functions that we're trying to estimate!==  
- If we want to estimate $v_{\pi}(s)$ for our current policy, but ==our policy is changing== (And maybe starts as a random policy), then that's a non-stationary target! The shape of that true $v_{\pi}$ will change as we get better at navigating our domain!
- We also need to allow for ==non-i.i.d. data!==
	- Where I am now is very highly correlated with where I am at the next step!

![[Pasted image 20250114232303.png]]
Here's gradient descent!
We have a function J(w) that's differentiable with respect to a parameter vector w.
Our gradient vector is the vector of partial derivatives with respect to each parameter.
This gradient tells us the direction of steepest ascent.
We go in the opposite direction, moving downhill (to minimize loss, in hopes of finding a local minimum).

Let's now talk about its application in value funciton approximation!
![[Pasted image 20250114232755.png]]
Let's imagine that someone tells us $v_{\pi}$ -- that someone (an oracle) has just told us this.
- If we had this, we could just minimize the [[Mean Squared Error|MSE]] against the oracle's true value function.
- To do gradient ascent, all we need to do is move a little bit in the direction of the error that we see in each state multiplied by the gradient.
	- (This naturally falls out of applying the chain rule to the first J(w) equation on top)
- To deal with the expectation, we use Stochastic Gradient Descent
	- Randomly sample a state by seeing what we visited, look at what the oracle says in that state, look at our estimate of the v, compute our error term, and then compute the gradient and do our multiplication in that bottom $\triangle w$ bit. 

So every step, incrementally, online:
- Take a step
- Make a prediction of what the value is going to be
- Oracle tells you what the value should be
- Adjust our weights
- Move on to the next step

Stochastic approximation theory tells us that this really will ultimately let us fit the oracle's prediction.

Is that clear so far, aside from the part that we've cheated with having the Oracle?

![[Pasted image 20250114233402.png]]
You'll usually see a bunch of linear function approximation using ==features== or ==feature vectors==, which are basically just... something that tells you something about your state space -- it can be *anything you'd like it to be!* (How far my robot is from each of these landmarks).



The simplest way is to make a linear combination of these features with some weights $w$:

$\hat{v}(S,w) = x(S)^{\top}w = \sum_{j=1}^n x_j(S)w_j$

This isn't going to be a perfect representation (unless our features are very specifically engineered), it's a pretty simple function... but it's a good place to start.

![[Pasted image 20250115101452.png]]
Our Objective functino J(w) is going to be quadratic... because if we consider our parameters w, we have this squared error... we consider our mean squared error, we see that it's quadratic in w, meaning there's some bowl or other quadratic shape representing our mean-squared error, the objective funtion. THat's an easy shape to optimize using gradient descent, where we'll find the optimal mean-squared errors.

==So this is a nice consequence of using a linear combination of features, we can find the global optimum because it's convex optimization.==
- (This doesn't mean that the optimum found will be especially high-performing compared to some local optimum found by a more flexible estimator)

![[Pasted image 20250115101850.png]]
Table lookup (what we were doing before) is a special case of linear funciton approximation! We can build up a feature vector that says "If I'm in state 1, have a value of 1, else 0" (just a one-hot feature vector that' 1 for the state that we're in right now). If we use that as our feature vector, when when we take our dot product between our feature vector and our weights, then we're just picking out one of those weights!
- (This is kind of dumb, but... okay.)


But we've been ***cheating***, imagining the existence of some oracle that can tell us the true $v_{\pi}(s)$, so that we can fit against it using supervised learning! We need to figure this out from experience in practice.
- We need to ==LEARN== this using (eg) Monte Carlo and Temporal Difference methods to give us a target to use for our function approximation, instead of the oracle!
	- If we use MC learning, this target is just the return
![[Pasted image 20250115102159.png]]
 See that we just swap our the target (previous $V_{\pi}(s)$ given from an oracle) with our *estimate* of the state value function as given by our chosen technique (MC, TD, TD-Lambda)

![[Pasted image 20250115102605.png]]
We can think of this as a process that looks a lot like supervised learning
- Monte-Carlo learning can be used to build some training data, but incrementally.
	- We see state S_1, run a trajectory from it, see that we got G_1.
	- We see state S_2, run a trajectory from it, and see that we got G_2.
- Think of this as our training data.
- Now, just like supervised learning, we treat this as a dataset and we adjust our function approximator to fit our G's.

Q: How did that simplification work in that math equation on line two?
A: For ***LINEAR*** MC policy evaluation, the value function approximation is assumed to be linear in the features of x(S_t), where x is our ==feature mapping== that turns our state into features. In linear function approximation $\hat{v}(s_t,w) = w^\top x(S_t)$, and when we take the gradient of this with respect to $w$, we get $\nabla_w\hat{v}(S_t,w) = x(S_t)$  . This is because the gradient of a linear function with respect to its parameters just gives you the features/inputs. This is why we can make the substitution we did.


![[Pasted image 20250115103255.png]]
Here's a similar example using TD Learning
- If we consider again linear function approximation, we again plug in our TD target ($R + \gamma \hat{v}(S', w)$) 

In practice, every step, we'll take a step, generate our TD target (taking a step, getting a rewards, generating a new estimate of the value, computing error, and updating our weights to better estimate $\hat{v}(S,w)$).

Linear TD(0) still closes quite close to the global optimum, where "close" depends on things like your discount factor.


![[Pasted image 20250115103740.png]]
Finally, we can do the same idea with TD($\lambda$).
- The association we're making is each state with some lambda-return, which is the mixture of all of our n-step lambda returns.
- We try to learn to make our value function *fit* to these lambda return targets.
This works for either the forward view or the backward view...


----

Let's move on to control!
![[Pasted image 20250115104457.png]]
We're now going to use ==approximate== policy evaluation:
- We start off with some parameter vector w that defines some value function
- We're goign to act greedily (With some epsilon exploration) with respect to that value function that we've defined (eg some NN), giving us a new policy.
- We now want to evaluate that policy, which gives us a new value function!
- We repeat
- Note that we're not going to go all the way to the top during evaluation, wasting millions of sample of experience, to evaluate/perfectly fit the true value of our current policy.
	- Instead, we'll take some steps toward it and immediately update our policy
	- In the most incremental case, for TD for one step, we do this for every single step!
		- Take an action
		- Compute TD target
		- Update our NN once
		- Immediately act with respect to that latest NN to pick our actions

In order to do control, we took GPI and added two new ingredients
- Modeling the action-value function Q(s,a)
- Some exploration, e.g. using epsilon greedy

Now we just use a neural network to estimate our parameters Q
So what happens? Does this really get to q*?
- Of course not, it might not even be possible to represent q* using our approximation1
- We typically end up with algorithms that oscillate around some ball around q*.

![[Pasted image 20250115105025.png]]
So how do we do this with an action value function?
- Minimize the MSE between our predicted Q values and the "true" Q values for our policy (we don't have access to an oracle, so we'll use whatever method we'd like to estimate Q).
- We use SGD to find a local minimum

![[Pasted image 20250115105147.png]]
What does our function approximator look like?
- We can now build features of BOTH the state and action using our feature mapping x(S,A). This spits out some feature vector, given an (S,A) tuple.
- This feature vector can be used in a linear combination with learnable weights.

![[Pasted image 20250115105409.png]]
We can plug in the same idea, where we can basically just do the same thing, substituting our best target for $q_{\pi}(S,A)$, using the return, one-step TD-target, or Lambda return (or the backward view of TD lambda, to use eligibility traces), this time using Q rather than V, so that we can do model-free control, since we can just take the action that maximizes Q, rather than know about the environment transition dynamics and using V.

![[Pasted image 20250115110522.png]]
A one-slide summary that roughly tell us when it's "okay" to use TD, when we're guaranteed for TD to converge to something. 

![[Pasted image 20250115110630.png]]Gradient TD is a somewhat more new (in this 9 years ago lecture) method, which is a true gradient descent approach... it really follows the gradient of the projected bellman error... by just using a small additional correction term, which fixes the problem that we see in TD learning.

What about for Control?
![[Pasted image 20250115110747.png]]
($\checkmark$) means that we get chattering around the near-optimal value function
($X$) means that catastrophic divergence is possible

## Batch Methods
Motivation
- Gradient descent is simple (just move in the up/downhill direction!) and appealing
- But it's NOT sample efficient (we throw that experience away after, and move to the next one)
- Batch methods seek to find the best-fitting value function to all of the data that we've seen in the batch -- the agent's experience ("training data")
	- "Life is one big training set for an agent"

We want to find a value function that best explains all the rewards it's seen so far.

![[Pasted image 20250115111230.png]]
One type of fit is a least-squares fit
- We can define some dataset consisting of <state, value> pairs of $(s_n, v_1^{\pi})$ , where these values are given by an oracle or something.
- The question is what's the best-fitting value function for the whole dataset?
	- We can choose $w$ that minimize the sum of squared errors, giving us a best-fitting value function $\hat{v}(s,w)$
- [[Least-Squares]] algorithms find parameters vectors that minimize the SSE between our estimator and the target.

![[Pasted image 20250115111424.png]]
And an easy way to find this least-squares solution is called [[Experience Replay]]
- We actually store this dataset, we make it a literal thing -- we make our training set an explicitly-stored object, using an experience replay memory.
- At every timestep, we sample a state and value from our experience buffer $(s, v^{\pi}) \sim D$
	- This v is still an oracle return
- We then apply SGD, updating our predicted towards the target.

So we don't present things in the order they arrive (Which is strongly correlated), instead we de-correlate them, presenting them in a randomly sampled order, until we get to a least-squares solution.
$w^\pi = \underset{w}{argmin} LS(w)$ 


![[Pasted image 20250115111657.png]]
Now we can understand the use of [[Deep Q-Networks|DQN]] for Atari!
- Uses [[Experience Replay]] and off-policy [[Q-Learning]]
- We remember the transitions we've seen so far
- Every step, we take some action according to some epsilon-greedy policy with respect to our function approximator $\hat{q}$, which is just a big neural network.
- We sample some mini-batch of transitions from our dataset $D$ (say, 64 random samples from our experience-replay memory)
- Compute the gradient with respect to those 64 things ,and optimize the MSE between what the Q-network is predicting and our actual Q learning targets (this is a Q-learning target; just like our SARSA targets, but with a max over actions, if you recall).

![[Pasted image 20250115113047.png]]
Their DQN was a [[Convolutional Neural Network|CNN]] which at the end of it, output Q values for every single action.
- They used the same NN with same hyperparameters, schedule, etc. across 50 different ATARI games, and the results were great! For most games, they did better than humans (mostly for the games that are very twitchy).

![[Pasted image 20250115113256.png]]
[[Stochastic Gradient Descent|SGD]] with [[Experience Replay]] as away of getting the most out of the data that you've seen so far.
- Are there methods where we can jump directly to the least-squares solution?
	- Yes, for the special case of linear function approximation.

![[Pasted image 20250115113406.png]]
For linear value function approximation, we can use closed-form solutions rather than SGD.

![[Pasted image 20250115113433.png]]

