https://www.youtube.com/watch?v=0g4j2k_Ggc4&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=7

---

Everything has been leading to this: Dropping our agent/robot into an unknown environment, where we don't tell the robot anything about how the environment works... and then the robot figuring out how to maximize reward, without knowing anything about the environment!

In subsequent lectures, we'll look at how to scale up and get to more realistic large-scale domains; but the core techniques all build on what we saw in the law lecture [[David Silver RL (4) - Model-Free Prediction]].

---

Agenda:
1. Introduction
2. On-Policy Monte-Carlo Control (On-Policy means learning on the job)
3. On-Policy Temporal-Difference Learning 
4. Off-Policy Learning (Off-Policy means learning while following someone else's behavior)
5. Summary

Last Lecture:
- ==Model-Free Prediction==
	- How to evaluate a given policy (find its value function) of an unknown MDP
This lecture:
- ==Model-Free *Control*==
	- *OPTIMIZE* the value function ($v_* , q_*$) of an unknown MDP, using Model-Free Prediction in the inner loop to help us optimize behavior.

----

## (1/5) Introduction

![[Pasted image 20240626121752.png|300]]
 All of these problems have MDP descriptions of the problem -- There's an environment that has some rules about how you move around in it... but that thing is unknown to us, so we have to resort to model-free control (Or, the MDP is known, but it's too complicated and large that we can't really use it directly; we have to just *sample it*, trying out actions.)

![[Pasted image 20240626122009.png|400]]
- [[On-Policy]] Learning
	- You have a policy, you follow that policy, and while you're following that policy, we learn. We follow a policy, and at the same time we're trying to evaluate that policy. The actions that we take determine the policy that we're trying to evaluate.
- [[Off-Policy]] Learning
	- We might have some robot watching some other robot walk around, and it's trying to figure out optimal behavior by watching that other robot (or a human, teleoperating). The robot learns for itself things like "What would happen if I did something other than the actions recommended by my policy?"

Recall: Generalized [[Policy Iteration]]
![[Pasted image 20240626122245.png|400]]
Iterative process where we start with a value function and policy (V, $\pi$), and every time we go "up", we evaluate our current policy to get our new value function, and every time we go "down," we act greedily with respect to that value function (perhaps Epsilon-greedily). This converges on the optimal policy and value function $\pi_*$ and $v_*$ .

So the question is: What can we slot in for these two pieces (Policy Evaluation, Policy Improvement -- our "up" and "down" movements) to allow us to do Model-Free Control?


## (2/5) On-Policy Monte-Carlo Control

![[Pasted image 20240626122504.png|400]]
(Above: This is *not* going to be our answer for how to do model-free MC-evaluation-based control)
Is it possible to just slip in Monte-Carlo policy evaluation into our "up" movement, instead of Dynamic Programming evaluation? Where we just take a bunch of samples/trajectories and then take the mean to estimate V? And then just follow greedy policy improvement?

Whats ==wrong== with this?
1. We want to be model free, but how can we be model free when we're using a state-value function V?
	- If we only have the state-value function and we want to act greedily with respect to it, we ==still need a model of the MDP== $\mathcal{P_{ss'}^a}$ to figure out how to act greedily with respect to the value function. 
	- Given a state, considering which action is best, we have to be able to roll the dynamics forward one step to then check the value function of the state that we might end up in. But we're trying to be model-free, here!
2. If, when you're running trajectories, you follow your policy greedily, then you're perhaps not going to explore the entire state space (this is a problem with policy improvement)
3. You might need to run a large number of trials/episodes to make this work (we'll improve on this my using temporal difference later).

![[Pasted image 20240626123029.png]]
Above:
- Like we said before, greedy policy improvement over the *state-value function* V(s) requires that we have a model of the MDP, because it requires some knowledge of where we'll end up in the next state, so that we can assess the value function in *that* state. This requires knowing "where the wind will blow us," which we don't have in a model-free context. 
- In contrast, if we use the *action-value function* $Q(s,a)$... we can do control in a model-free setting, because we can do policy improvement in an entirely model-free way! All we need to do is maximize Q value, over actions! 

Let's drop that in!
![[Pasted image 20240626123314.png|400]]
- We start with a (Q, $\pi$). At every step, we do a Monte-Carlo policy evaluation, running a bunch of episodes. At the end of those, we take the mean return to estimate Q(s,a), telling us how good a state-action pair is. We do that for all states and actions, giving us a new value function.
- We then act greedily with respect to our Q, which gives us a new policy.
- We iterate

==We still have one issue though==, which is that issue of ==exploration== -- we're currently acting *greedily* with respect to our policy, which means that we might not explore the state space effectively. We can introduce [[Epsilon-Greedy]] sampling of our policy, where we take some random action with probability $\epsilon$.
- Because we're not doing full sweeps of the environment like we are in dynamic programming, and we're just sampling a policy, we'd still like to make sure that we're seeing everything.

![[Pasted image 20240626124702.png]]
The simplest possible idea for ensuring continual exploration is [[Epsilon-Greedy]] exploration! There are more complicated algorithms out there, but it's hard to beat this simple technique.
- We flip a coin, and with probability $\epsilon$, we select on uniform random action, and with probability $1-\epsilon$,  we take the greedy action of argmaxxing over our Q(s,a) values.
This guarantees that we continue to explore everything, and guarantees that we'll continue to improve our policy as well.


![[Pasted image 20240626125324.png]]
One of the reasons why Epsilon Greedy is a nice idea is that it actually guarantees that we get an improved policy, during a policy improvement step.
- We start with one epsilon-greedy policy, and the new policy $\pi'$ is going to be better than the previous $\pi$.
- The proof basically says that... (over one step, and then telescoping this over other steps as the proof) if we take one step over our new policy, it's better than our original policy.


![[Pasted image 20240626125802.png]]
- For policy evaluation, we now plug in Monte-Carlo as our way to evaluate policies, using our action-value function Q instead of our state-value function V.
	- For every state action pair, we look at the mean return over many rollouts.
	- This might be inefficient for large state-action spaces; we'll see how to improve that soon.
- For policy improvement, we're using epsilon-greedy policy improvement.
	- The stochasticity in the policy ensures that we explore everything in the environment (eventually).
	- Note: This still might not *effectively* explore states that are down some strange, rare trajectory... within a reasonable time.

We saw in the last lecture that it's not necessary to *fully evaluate* your policy... sometimes you can spend a few steps to evaluate your policy, and you've already got enough information to evaluate your policy. So what does that look like in the context of Monte-Carlo?
![[Pasted image 20240626130102.png]]
In the context of Monte-Carlo:
- Why not do this every single episode?
	- We run one episode, collect all the steps along the episode, update the estimated Q value for those steps, just for this episode.
	- And then improve our policy straightaway; why wait, when we can already improve our policy?
	- We always act (epsilon) greedily with respect to the *latest* estimate of the station-action value function $Q$.

![[Pasted image 20240626130235.png]]
Natural question: How can we really guarantee that we find the optimal policy $\pi_*$ and $q_*$?
- To do that, we need to balance two things:
	- Exploring to make sure we don't leave money on the table.
	- Asymptotically, that we get to a policy where we're not exploring at all -- it's unlikely that a policy with random behavior will be optimal!
- [[GLIE]]: Greedy in the Limit with Infinite Exploration
	- Come up with any schedule where:
		- We continue to explore all states; that every state and action will be tried (eg Epsilon Greedy will have this property).
		- We want to make sure that the policy *eventually becomes greedy*, so that it satisfies the Bellman Optimality Equation (which requires an argmax over Q values).
	- One idea is to take an Epsilon-Greedy policy and just decay epsilon towards zero, e.g. on a hyperbolic schedule, where every episode we set $\epsilon_k = 1/k$


We can use this to make an algorithm! GLIE Monte-Carlo Control
![[Pasted image 20240626130613.png]]
We start off by sampling episodes from our current policy, generating S,A,R trajectories
- For each state and action, we update our action value!
	- We count how many times we've seen our state action pair, and do an incremental update to the mean. We update our existing estimate of Q a little bit in the direction of the return that we just got, and the amount we have to adjust it by to get the correct mean estimate is this 1/N term.
	- This isn't a statistical mean over some iid quantity; the policy is actually changing over time! So we're basically taking returns from better and better policies into account. We collect these statistics from increasingly-better policies to get one overall mean of how good it is.
	- The GLIE property ensures that over time, the policy gets more and more like the greedy policy; the policy that we actually care about.
- We iterate over this process, updating our Q values and improving our policy (setting epsilon to new value, and acting epsilon-greedily to our new Q values)
	- Note that we will only have new Q values along our most recent trajectory.
	- So we don't even have to store $\pi$, just $Q$ -- we just store Q and act epsilon-greedily with respect to it.
- This converges to the optimal state-action value function!


Q: How should we be initializing Q?
A: In terms of the theory, you can start with anything; it will still converge. In practice, it helps to have a good starting estimate of Q to have faster convergence.

## (3/5) On-Policy Temporal-Difference Learning

In last lecture, we started with Monte Carlo Learning for evaluation, then did Temporal Difference Learning for evaluation, and then found that there was a spectrum of methods between them (in TD-Lambda). We're going to the same thing now, but for Control!

![[Pasted image 20240626132952.png]]
- We can gain efficiency by *bootstrapping!* We want to use TD learning because:
	- It can be lower variance, run online in continuing domains, or on incomplete sequences.
	- There's an additional benefit in Off-Policy learning where it becomes important to use TD.
- Natural Idea: Let's just use the same generalized policy iteration strategy, but using TD instead of MC, in our control loop!
	- We know that we have to use Q instead of V, from our last section.
	- So let's just use TD learning to estimate Q(s,a), and continue to use $\epsilon$-greedy policy improvement!
	- Because TD learning doesn't need to wait to the end of the episode to update our value function, let's update ==every time-step==!
		- We see a sample of data, we bootstrap, and we update our value function immediately.
![[Pasted image 20240626133240.png|400]]
The general idea is called [[SARSA]] (the diagram illustrates why)
- We start in a given state with a particular action A, we sample the environment to both receive a reward and end up in a new state S', and then sample our policy again to generate A'.
	- SARS'A'

SARSA indicates an update pattern that we're going to use.
- We start with a Q value, and we move in the direction of our  {==TD Target== - Q value of where we started}. I'm in a state and I'm considering an action. If I take that action, look at the reward we got, and the value of the next action I'd take, that gives me an estimate of the value of this policy, and I use it to update the estimate of the state-action pair I started with.
	- This uses the Bellman Equation for Q.
- ==This is called a "SARSA update"==

![[Pasted image 20240626133530.png]]
- *Every single timestep* (more granular than every episode), we update our value function by applying a single step of SARSA.
	- For the state/action we're in, we *only* update Q(S,A) for that specific S,A.
We also improve our policy every timestep as well by acting epsilon-greedily to our Q.

![[Pasted image 20240626133830.png]]
Every step... we take an action, reserve the reward, observe the next state we end up in. We choose our action using our epsilon-greedy current policy, and then use our SARSA update to update our Q(S,A). Repeat!

![[Pasted image 20240626134152.png]]
Just like GLIE-Monte Carlo, this version will converge to the optimal.
We just require a GLIE sequence of policies (to explore everything, but also end up greedy), but we also need to be particular about step size (step size needs to be sufficiently large to move Q value as far as you want, but eventually, the changes to your Q values eventually have to become smaller and smaller over time).
- In practice, we sometimes don't worry about either of these (or at least, not about the step size guidance), and it seems to empirically work anyways.



Let's now consider the entire spectrum between Monte-Carlo and TD-Learning like we did in the last lecture. Considering Eligibility Traces and Lambda variants
- Our goal is to get the best of both worlds, and control the bias-variance tradeoff
![[Pasted image 20240626134859.png|400]]
- The question is how many steps of "reality" we want to take -- these n-step returns, where SARSA just takes one step, and MC takes all steps.

We can define our n-step Q-return to be the generalization of the above equations to any N.

n-step SARSA says: For each state/action pair, instead of updating towards the one-step estimate of our Q function, using SARSA, we instead use the n-step *return* as the target. We update a little bit in the direction of this target.

Like in the last lecture, we want to consider some algorithms that are able to consider multiple choices of n-s, getting the best of all possible n by averaging over them.


SARSA Lambda
![[Pasted image 20240626135745.png|400]]
- For the update
	- We average our expected returns over all of our n-step returns (following our policy in an epsilon-greedy fashion across however many steps our n is, accumulating accumulating rewards and considering the Q at the state we end up at)
- One the left side, the 1-$\lambda$ is just to normalize the whole sum to add up to one.
	- We take more account of the short-n trajectories, and progressively discount as we increase n.
- The main idea is to average over all of these n-step returns. We define a weighting, where we weight each n-step return by this factor of $\lambda^{n-1}$ (and then there's that normalizing factor $(1-\lambda)$ too).

This gives us our update rule! We update our Q value... considering how good it is to take an action... and considering some weighted sum that takes into account multiple possible future states. We update our Q a little bit in the direction of that target.

This is the ==Forward View==; there's a ==Backward View== too!
- Forward view is great to help us build this algorithm between MC and TD algorithms (lambda=1 -> MC, lambda=0 -> SARSA). We have this control to choose the bias-variance tradeoff... The problem is that we're looking forward in time. It's not an *online* algorithm, because we have to wait until the end of the episode to compute our $q^{\lambda}$. We'd like to be able to run this online, to be able to make an adjustment every single step!

![[Pasted image 20240626141943.png]]
We can build an [[Eligibility Trace]], like in [[TD-Lambda]]!
- But it's going to be for all state-action pairs. Think of this as something telling us how much credit/blame we should assign to every action we took from every state. We get a big carrot at the end of our episodes -- which states+actions are responsible for receiving that carrot? We say that the states and actions that were *most recent before getting the carrot*  and those that were taken *most frequently* along the trajectory are the ones that we should credit the most for the negative or positive reward received at the end of the trajectory.
- We can do this online:
	- Every time we visit a (s,a) pair, we can increment the eligibility trace.
	- Every timestep, for all state-action pairs, even the ones we don't visit, we decay the eligibility.
- We update our Q(s,a) for every state and action, and update in proportion to the TD-error $\delta_t$ *and* eligibility eligibility trace $E_t(s,a)$ .

Q: Is there a way to apply this to very large state spaces where we don't/can't visit every state, or a same state twice.
A: Next class we'll deal with function approximation for all the algorithms we've seen so far. 

![[Pasted image 20240626142427.png]]
We initialize our eligibility traces to zero at the beginning of the episode
- For each episode
	- We take our given action A, observing R and S'
	- We pick our action using our policy (eg acting epsilon greedily)
	- We compute our TD error, looking the difference between the return of the state/action we end up in, compared to our previous estimate.
	- We increment our eligibility trace
	- For ALL states and actions:
		- We do our Q(s,a) update, which incorporates both our TD-Error and our Eligibility Trace.
		- We decay our Eligibility trace.
	- We set our new state/action for the next step, and iterate.


![[Pasted image 20240626142742.png]]
Imagine we're in a gridworld where we're getting from a start point to our goal state.
- Let's say we start off initializing all Q(s,A) to zero. Let's indicate the Q(s,a) for a state with an arrow, and the size of the arrow indicates the magnitude of the state-action function for that state.
- When we make our update of getting to our reward, see on the right  that we have this decaying impact on Q for visited states. Our lambda parameter influences how quickly/far that information propagates back through our trajectory.

## (4/5) Off-Policy Learning

Everything so far has considered the case of [[On-Policy]] learning, where we're "playing our best game" as we explore (with some $\epsilon$ chance of taking a uniform random action). The policy that I'm following is the policy that I'm learning about.

[[Off-Policy]]
![[Pasted image 20240626145451.png]]
We want to evaluate our ==target policy== $\pi$ ... but say we're actually following ==behavior policy== $\mu$; this is the one that we're actually following in the environment.
- Why would we want to do this?
	- ==Learning from observing humans or other agents; not just using supervised learning to *copy*, but from that experience, learning how to do *better*, learning how to *solve the MDP!*==
		- "I know how you could have done better, Human! you could have gotten more reward by following this *other* path."
	- ==Re-using experience generated from *old* policies ($\pi_1$, $\pi_2$, $\pi_2$)==
		- Can we use this data more than once, going back over *old* data? "Last time I was here, I took this action and got this other reward." Batch methods where we treat these as large datasets to learn from. Can we make use of this additional data to get a better estimate of our *current* policy $\pi_{t-1}$?
	- ==Learning about *optimal* policies while following an exploratory policy.==
		- This is perhaps the best-known; we know there's a big issue in RL, which is making sure that we explore the state space effectively; at the same time, we want to learn about the optimal behavior, which doesn't explore at all!
		- We can have an exploratory policy that putters around, while still learning about the optimal policy!
	- ==Learning about *multiple* policies while following *one* policy.==
		- Perhaps the most exciting; we might have many different behaviors we want to figure out. "What would happen if I leave this room, how much reward will I get?" "What might happen if I change the topic, and go back to three slides ago." So many questions about the future are conditioned on different policies. Can we learn about these various policies while using the one stream of experience we have, called life?

Off Policy learning is a significant, thorny problem!

![[Pasted image 20240626150020.png]]
[[Importance Sampling]]
- The idea is to take this expectation... the expected future reward... and all we do is say that this is a sum over the probabilities times how much reward we got.
(Muddled Explanation)

![[Pasted image 20240626150135.png]]
Monte-Carlo learning is a REALLY BAD idea for off-policy; your target policy and your behavior policy just never really match enough for it to be useful

![[Pasted image 20240626150405.png]]
So you have to use [[Temporal Difference Learning|TD-Learning]] when working off-policy; it becomes imperative to bootstrap!
- The simplest idea is to importance sampling, but only doing it for one step, because we're bootstrapping after one step. We update our value function a little bit in the direction of our TD target...
- We take our TD Target, and we correct for our distribution (from using a behavior policy, rather than our target policy), over that one step.
It can still blow up, because we have variance, but it's much lower variance than Monte Carlo.


[[Q-Learning]] (is specific to TD-Zero) is the best option when it comes to [[Off-Policy]] learning.
![[Pasted image 20240626150755.png]]
- We're going to consider a case where we make use of our action-values Q(s,a) to do off-policy learning in a particular way that *doesn't* require importance sampling.
- We select our next action using our *behavior* policy 
- We also consider an alternative successor action that we might have taken, had we been following our target policy.
- We update the Q value for the state we started in, and the action that we actually took... and we say the value of the state and action we actually took... we update towards the value of our *alternative*  action! Because that's the thing that tells us how much value it actually got under our real *target* policy.
	- Updated in the direction of the reward we actually saw $R_{t+1}$ plus the discounted value of the next state of our *alternative action, under my target policy* (what would have happened if we were following the target policy we cared about). This is a bellman equation!

You're sampling A_t+1 from your behavior policy, and A' from your target policy. EAch time you get to a state, you consider the thing you actually took (generating real behavior; that's the trajectory you're following), but at every step we also consider the thing we might have taken, and that's what we bootstrap from.

![[Pasted image 20240626151148.png]]
If you hear about the [[Q-Learning]], it's what we'l talk about in this slide, which is the special case where the target policy is a greedy policy. 
- The special thing about Q-Learning is that both the behavior *and* target policies can improve.
- At every step, we make the target policy greedy with respect to our value function -- an aggressive, deterministic thing.
	- But our behavior-policy is only *epsilon-greedy* -- it roughly follows optimal actions, but explores as well.
- When we plug this in to the previous slide, we end up seeing that... our alternative action is now taking this argmax over a'... which is the same as the max of our Q values.
- So basically, Q learning updates a little bit in the direction of the maximum Q value we could take.


![[Pasted image 20240626151429.png|400]]
- The well-known Q-learning algorithm. He calls these SARSAMAX updates, because we're looking at SARS'A', and then the max over the A's.
- We update our Q values a little bit in the direction of the best possible Q value that we could have over the next step. Should be familiar, because it's just like the [[Bellman Equation|Bellman Optimality Equation]], and converges to the optimal Q*.

## (5/5) Summary 

![[Pasted image 20240626151843.png|400]]
- If we use the Bellman Expectation Equation for the value function $v_{\pi}$, you can use DP to evaluate your current policy... or you can just sample this thing and just take one sample, which gave us TD learning. So TD learning is just a sampling version of Iterative Policy Evaluation.
	- TD we can think of as samples of the Bellman Expectation/Optimality equations; doing a one-step sample of what would happen if we were doing DP.
- If we use the Bellman Expectation Equation for $q_{\pi}$, we can actually do SARSA now. We can use it to evaluate our state-action Q values, and plug it into generalized policy iteration to make it better and better. We can either use a policy iteration framework, or SARSA.
- Finally, we can use the Bellman Optimality Equation for $q_*$, and plug it either into Value Iteration, or we can just sample this to give us the Q-Learning algorithm.

Next week we'll talk about function approximation, and how to scale these things up to work on more practical, large-scale examples.







