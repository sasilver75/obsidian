https://www.youtube.com/watch?v=KHZVXao4qXs&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=8

- Introduction
- Finite Difference Policy Gradient (A sort of naive but sometimes effective numerical approach)
- Monte-Carlo Policy Gradient (Seeing that actually, for many RL methods, there is an analytic formulation for the policy gradient.)
- Actor-Critic Policy Gradient (The most practical set of methods; combines things from this class with things from the previous class, working with both value functions and policies, trying to get the best of both worlds.)

We're talking about methods that optimize the policy directly, rather than working on value functions. How can we see an experience, and from that experience, figure how to change our policy in a direction that makes the policy better.
We'll see a variety of methods based on the simplest principle for improving a polciy: Policy Graident methods, which are about taking steps in the direction that make a policy better.

==Notes for this might be bad because he just has a recording of his screen, so I can't see what he's pointing to.==

----

![[Pasted image 20240628144658.png]]
We just had a lecture where we talked about value function approximation:
- Approximating the state-value function, telling us how much return we can expect to get if we're in a state following policy $\pi$ 
- Approximate the state-action q function, telling us how much reward we can expect to get if we're in a state and take a specific action, following policy $\pi$ thereafter.

We tried to fix functions to approximate these value functions. We didn't represent the policy explicitly, instead just using the Q function + epsilon-greedy action selection to define our policy.
- Now, let's directly parametrize the *policy*!

$\pi_{\theta}(s,a) = {\mathbb{P}}[a | s, \theta]$

This could just be a function; which action do we pick in a given state. Let's learn that function such that we solve the RL problem, maximizing the expected reward in our environment.

We'll focus in this lecture on [[Model-Free]] RL, where we just throw our agent into an environment without telling it anything, and the agent figures out how to adjust its parameters of its policy so as to maximize reward. The motivation of this is to scale to realistic (unknown environment, large state space) problems.

---

![[Pasted image 20240628145117.png]]
We'll start to understand how we can take this parametrized policy and adjust those parameters to get more reward; we'll consider using Gradient Ascent. If we can follow a gradient in a direction that improves our policy, we'll get increasing rewards.

Why would we want to work with policy-based RL, rather than value-based RL?
- There are some situations where it's more efficient to represent the policy than the value function -- imagine we're playing an Atari game alike Breakout, and the value function might be quite complicated ("Your current state is 1752 points"), but to represent the idea of "move below the ball" as a policy is simple to represent.

![[Pasted image 20240628145428.png]]
- Better Convergence Properties
	- Value-based methods can oscillate, chatter, and diverge if you do things in a weird way. Policy-based methods tend to be more stable, because we're just making little incremental changes to our policy. Tend to have better convergenec properties.
- Policy-Based methods are effective in high-dimensional or even continuous action spaces
	- For value-based methods, you have to be able to compute the *max* -- "Once we have Q, all we have to do is pick the action that maximizes over A", but what if that maximization itself is prohibitively expensive?
- Can learn stochastic policies
	- Why would we want to learn a stochastic policy? Don't we want to *maximize* reward, which sort of implies a deterministic policy?
	- There are cases where it's actually desirable to have a stochastic policy:
		- Rock paper scissors: Deterministic policies are easily exploited, and a uniform random policy is optimal.
		- When we have state aliasing: When the Markov property doesn't hold; we only use features that give us incomplete information about where we are in the environment. EG perhaps your agent is on a chessboard, but can't see black or white squres.

So what does it mean to optimize a policy?
![[Pasted image 20240628151049.png]]
We need something to optimize
- In an episodic environment, we can use the start value
	- If I always start in some state $s_1$ ... whats the total reward I'll get from that state onwards?
	- EG If I always get dropped into this start state and play out a level.
- In continuing environments, we can use the average value; If the environment goes on forever, there might not be a start state. 
	- Let's say it's the probability that we actually end up in any state under our policy pi... times the value of that state onwards.
- Or we might consider the average reward per time-step.


It turns out that, fortunately for us, exactly the same methods apply to all three of these -- the policy gradient is essentially the same (the only thing that changes is the definition of this distribution term)... they all turn out to basically follow the same gradient direction, so we don't need to worry about which of these we're following, because we get the same gradient.

![[Pasted image 20240628151817.png]]
So let's pick one of these objectives, and then say that we want to find the $\theta$ that maximize this objective function.
- Gradient-free methods (usually if you don't have access to the gradient)
- Gradient-based methods (typically give better efficiency; we'll focus here)

## Finite-Difference Policy Gradient Methods

![[Pasted image 20240628152314.png]]
In this lecture, we consider gradient *ascent*, because we want to *maximize* the reward of our system; our objective function.
- The gradient basically points us in the direction of steepest ascent, uphill. Mathematically, we just adjust our policy parameters a little bit in the direction of the gradient of the policy with respect to the objective function.

![[Pasted image 20240628152451.png]]
If we had no way to understand the gradients, we could do this numerically, too
- We perturb our parameters in each dimension
	- What would happen if I slightly perturbed my policy along this first axis? We look at how much the objective function changes.
	- We do this for each dimension separately, giving a numerical estimation of the gradient.
This is a little naive, because if we're working in high-dimensional space (eg a Neural Network with 1,000,000 parameters), we would need to do 1,000,000 evaluations of our policy.
- There are techniques that use random directions (eg SPSA) that reduce the number of samples we need to take, but they're noisier too.
==It's naive, but this works even if you have some non-differentiable policy, though.==

## How to analytically compute gradients, so we don't have to numerically compute for each of our separate dimensions

Okay, let's start with some simple approaches (no value functions yet) -- the Monte Carlo approaches.
![[Pasted image 20240628153118.png]]
Idea:
- We assume that our policy is differentiable whenever it's non-zero (when it's actually picking actions).
- We assume that we know the gradient of our policy; that we're in control of our policy (maybe its a NN).
- We use a trick called ==Likelihood Ratios==:
	- This magic gradient of the log(Policy) term will appear a lot for the rest of the course, so it's important to understand
	- It comes from this likelihood  ratio trick:
	- We want to understand the gradient of our policy (policy gradient), and we want to take expectations of that thing.
	- We can multiply and divide by our policy without changing it 
	- On the right, the gradient of the policy divided by the policy -- that's equal to the gradient of the *log* of the policy. That's some straightforward calculus.
- The way to think of the score function
	- It's the term that tells us how to adjust our policy to get more of something. So we're going to use this...
- By rewriting the gradient in this way, we can compute expectations (easier, because we have this policy here... and this is the policy we're actually following ((??)))


![[Pasted image 20240628153614.png]]
- The ==softmax policy== is basically something where we want to have some smoothly-parametrized policy that tells us how often we should choose an action, for each of our set of actions. an alternative to Epsilon-Greedy.
- We have a linear combination of features, and we consider that as some kind of value that tells us how much we want to take a particular action. And then to turn it into a probability, we exponentiate it and normalize it.
	- The probability that we take an action in a state is *proportional* to the exponentiation of the value we get when we take some linear combination of our features with our parameters.

So we would have some features corresponding to going left, and some features corresponding to going right.

We want to find the gradient of this thing to know how to update it; the score function is as we see: It's just the feature for the action we actually took minus the average feature for all of the action we *might have taken*. ("How much more of this feature do I have than usual")
- We'll often say "If a feature occurs more than usual and gets a good reward, then we want to update the policy to do more of that thing."


![[Pasted image 20240628153841.png]]
The gaussian policy is common to use for continuous action spaces.
- We basically just parametrize the mean of the gaussian, and have some variance around that mean saying "most of the time we'll take the mean action, given by, say, a linear combination of features, but sometimes I'll take a deviation from the mean characterized by some variance $\sigma^2$, which itself could also be parametrized".
Basically, we get to pick the mean, and we just add some noise to the thing to make it stochastic
- The score function tells us how to get more of a particular action... and here, again, we have something very similar to the last slide.
- $a$ is the action we took, and $mu(s)$ is the mean action. So the action we actually took minus the mean tells us how much more than usual we're doing a particular action... multiplied by the feature... and scaled by the variance.
- So again the score function is saying something like  "how much more than usual am I doing something"


![[Pasted image 20240628154102.png]]
Analytical policy gradient (special case we're considering is one-step MDPs, where we start in one state s, we get to take a step, and get a reward... and then the episode terminates. There's no sequence, it's just one step and we're done, and the goal is to pick actions to maximize that reward; might depend on the state and action. It's a type of Contextual [[Bandit]]).
- We start by picking an objective function. Our objective function here is the expected reward under our policy. We want to find the gradient here and ascend it to get the most reward.
(Math being done)

We want to do somethign similar for Multi-Step MDPs, not just One-Step ones!
![[Pasted image 20240628154656.png]]
To do a multi-step MDP, we just need to replace the immediates reward with the value function (the long term reward). This turns out to then give us the true gradient of the policy.
- If you start with some policy... For any start state... the policy gradient is given by this bottom thing
- An expectation given by the score function multiplied by the state-action value function Q.
	- {How to adjust the policy to get more or less of the action}x{How good that action was in expectation}
	- We want to adjust the policy in a direction that gets more of the good things and less of the bad things.


[[Monte-Carlo Policy Gradient]] /[[REINFORCE]] (very similar to reinforce)
![[Pasted image 20240628155026.png]]
The idea is to update our parameters by stochastic gradient ascent
- We sample this expectation... and the way we do this is... we sample this Q term by using the return as an unbiased sample of this Q
- We're in a state, start by taking an action, see the return we got, and use that as an unbiased estimate of Q, and then plug that into our policy gradient formulation to get a direction to move our parameters.
- At each step, we can update our parameters a little bit in the direction of the {score}x{reward we got from that point onwards}
- This is like a forward-view algorithm where we have to play to the end to generate a reward
	- And then for each of those intermediate steps, we can adjust our parameters, using that *actual return* which is an unbiased estimate of Q.

[[Actor-Critic Policy Gradient]]
![[Pasted image 20240628155525.png]]
A quick recap: The main idea with MC Policy Gradient is that we have these high-variance estimates of the gradient (by playing out games to completion; one one particular episode, I might get a score 1,000, and the next game a score of 0.)

The main idea of actor-critic methods is that isntead of using the return, we explicitly *estimate*  the action-value functino Q using a critic (a value function approximator).
- So we'll combine value-function approximation from our last lecture.
We plug this in to our policy gradient formulation, replacing $Q^{\pi_{\theta}}$ 

So this means that we now have two sets of parameters (this is why it's called actor-critic methods). The actor is the thing that is doing things in the world (policY), and the critic doesn't take any decisions, it just watches what the critic does and says whether things look good or bad. We combine these two things together, and the main idea is to use an approximate policy gradient instead of a true policy gradient.
- We adjust the policy in a direction which, according to the critic, should get more reward. "I think if we go in this direction, you could do better!"

We take our original policy gradient algorithm and repalce the true action-value function with our estimated action-value function.
- At each step, we move a little bit in the direction of the score, multiplied by a sample from our own function approximator.

![[Pasted image 20240628160141.png]]
But how do we estimate Q, our action-value function? Should be familiar from last lecture
- Evaluation: If I'm following some policy $\pi$, can we estimate Q_{\pi}? How much reward can we get under the current behavior? And then using that to point us in the direction that will *improve* our policy.
We can use everything we've learned so far!
- MC Policy Evaluation
- TD Learning
- TD Lambda
Use them to get an evaluation of our policy algorithm.


![[Pasted image 20240628160416.png]]
Using v for the critic parameters and v for the actor parameters.
QAC = Q actor critic
- Every step of our episode (online, at every step, using TD in our critic)
- We sample a transition, see what the reward was, and see where we ended up
- We pick an action according to our policy
- We get the TD error between the value before that step and the partially-realized value after that step
- We update our critic in the direction of the TD error, multiplied by the features (using linear TD here from last lecture, meaning using linear function approximation)
- We adjust the actor in the direction that gets us more of the things that the critic says are good.

We can think of this as another form of generalized policy iteration
- We have a policy
- We evaluate that policy using the critic
- Instead of doing greedy policy improvement, we move a gradient step in some direction to get a better policy.


----

## Tricks to make things better
![[Pasted image 20240628161447.png]]
Reduce variance by having a baseline function that we subtract from the policy gradient. This can change the variance, without changing the expectation (🤔)
- The baseline we choose to use is the state-value function
- After subtracting, we're left with this [[Advantage Function]], which tells us how good we expect it is to take action $a$ from state $s$ (V), compared to how good we expect it is to be in state $s$ generally (Q).

You don't want to end up in situations where in one moment you see a reward of 1,000,000 and in the next you see reward of 9,000... it would be much better to have like +1 or -1; this gives us a way to rescale things.

So our policy gradient can be rewritten as the *score* multiplied by the *advantage function*
- Advantage function tells us how much better than usual an action a is
- The score tells us how to adjust our policy to achieve action a
	- If the advantage is ever negative, this score tells us how to push away from policies that would assign high probability to the bad action


![[Pasted image 20240628162141.png]]
How do we estimate this advantage function? How do we do it in the critic?
- There are many ways to do it, we'll suggest one here.

Our critic could learn both Q and V, using two separate sets of parameters ... and then we take the difference between these.
- Takes more parameters, but gives a good estimate of the advantage that we can then plug back into our policy gradient formulation.


![[Pasted image 20240628162254.png]]
This is an easier and probably better way, and commonly used.
- Claim: The TD Error is a sample of the advantagefunction.
- If we knew the true value function V_{\pi}, then the definition of the TD error is above... and then if we take the expectation of this TD error, it turns out that this is the advantage function.
The TD Error is an unbiased estimate of the advantage function!
- So we just need to move in the direction of the TD error, multiplied by the score.

A couple more ideas to throw in that we can try, too:
![[Pasted image 20240628162657.png]]
A reminder of previous lectures: What about this idea of [[Eligibility Trace]]s and different time scales? The idea that we don't *always* want to go to the end of the episode, nor do we want to just take a single step -- we often want to trade off bias and variance with things like [[TD-Lambda]]; this is applicable too to Policy Gradient algorithms.
- We can make any of these the target for our updates.

![[Pasted image 20240628162946.png]]
![[Pasted image 20240628163013.png]]

![[Pasted image 20240628163023.png]]

One last idea
![[Pasted image 20240628163233.png]]
![[Pasted image 20240628163253.png]]










