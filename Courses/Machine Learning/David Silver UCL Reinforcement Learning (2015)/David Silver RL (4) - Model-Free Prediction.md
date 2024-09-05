https://www.youtube.com/watch?v=PnHCvfgC_ZA&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=5

---

Model-Free Prediction
- Taking an environment that could be represented like a Markov Decision Process... but ==no one gives us the MDP that governs the way the environment works!==
	- No one tells you that the pollution level coming out of the smoke stack depends on the torque put into the engine, etc. What do we do in those situations?

### Agenda
- Introduction to the basic ideas
- Monte-Carlo Learning (One of the two major classes of Model-Free learning; we go to the end of the trajetory and estimate value by looking at sampled returns)
- Temporal-Difference Learning ([[Temporal Difference Learning|TD-Learning]]) (The other of the major classes; they look one step ahead, and *estimate* the return based on the step ahead)
- TD($\lambda$) (A spectrum of methods that unify these two approaches)

---

## Introduction

Last Lecture:
- We covered planning by [[Dynamic Programming]] to solve a *known* MDP (maximizing the reward from any state in the *known* MDP environment with the optimal policy)
	- We did this be learning policy evaluation, and then using that in a loop to solve for the optimal policy.
	- The idea of knowing how the environment works from the get-go is unrealistic.
This lecture:
- We use [[Model-Free]] prediction to *estimate the value function* of an *unknown* MDP
	- Focusing on policy evaluation; given a policy, how much reward will we get for that policy? We'll do this model-free, without anyone telling us about the environment.
Next lecture:
- We use [[Model-Free]] *control* to *optimize* the value function of an *unknown* MDP (solving the MDP by maximizing the reward from any state in the MDP environment)


## Monte-Carlo Learning
- Not necessarily the most efficient method, but very effective and widely used in practice; the idea is to just learn from episodes of experience!
- We don't need knowledge of MDP transitions/rewards, so it's model-free.
- Learns from *complete episodes*, where we start at the beginning, and the episode always terminates. ==This only works for Episodic MDPs; you have to terminate to get a mean, below.==
- We go through all of our episodes and use the simplest idea of estimating the value function by just sampling returns and estimating the value function as being the average of individual episode/sample returns.
- ![[Pasted image 20240625135420.png|300]]

So how do we use [[Monte-Carlo Learning]] to evaluate a policy?
- We simply take that policy, use it to create a number of samples by playing through an MDP episode using the policy, and then consider the value for a state as the *expected* return (which is the average, across episodes, of total discounted rewards from then on) from that state 
- We observe episodes of experience under the policy we're interested in evaluating.
- ![[Pasted image 20240625135754.png|300]]
- We're going to look at the episodes of experiences (S,A,R,S2,A2,R2, ...) and look at the total discounted reward that we get from each timestep onwards (G_t), under some discount factor.
- We then take the value function as the expected return from time=t onwards, if we were to set our state to that particular point.
- In Monte-Carlo policy evaluation, we use this empirical mean return in place of this expected return by collecting as many samples we can from that state onwards.
	- The question is: ==How do we get the value function for *all states* using this strategy?==

==Approach 1==: ==First-Visit Monte-Carlo Policy Evaluation==
- Imagine you've got a loop in your MDP where you come back to a state repeatedly.
	- The very first time we visit that state, we say, even if we come back to that state later, we're just going to look at the first time we arrived at that state; when we reach that state for the first time, we increment the counter to say how many times we've visited that state... we add up the total return, and get the mean return from that state onwards.
	- At each episode, we look at the first time we visit that state, measure the total return that we got from that, and take the mean over those total returns from that point onwards.
		- We use the law of large numbers, which basically says that we'll converge on the true value function for our policy, as we get more samples; the only requirement is that we visit all of our states.
- ![[Pasted image 20240625140322.png]]
- So we generate trajectories along our policy again and again, and we look at the returns we get from our first time we visit our state onwards, and we average those returns (across episodes).
	- The counter persists over multiple episodes; we try to get an average from multiple episodes; the counter counts over the set of episodes, how many times we visited the state *for the first time*
		- ==((OH, so the counter is only incremented once per episode, no matter how many times within a episode we cross specific state. So the counter, which persists across multiple episodes, really counts the number of episodes in which we encountered that state.))==

Q: How do we make sure that we visit all states?
A: We'll talk more about that next class; what we care about in *policy evaluation* is following the policy that we have, and seeing the states that we'll see *under that policy*. In subsequent lectures, the question you asked is an interesting one, where we want to cover the whole state space to find the best policy, which touches on the exploration question, which is fundamental in RL.

==Approach 2: Every-Visit Monte Carlo Evaluation==
![[Pasted image 20240625143251.png|400]]
Above: See that we hanged "The first" to "Every", in this example
- To evaluate a single state *s*, if we've got a loop in our MDP, we consider *every visit* to that state, instead of just the first one.  So in one episode, we can increment a counter multiple times, and we'll also increment the return twice in the episode.
- Both of these Approaches are valid estimators.

Q: So which one is better?
A: It sort of depends on the setting; when we talk about TD(\lambda) we'll talk a little more about this.


Example:
![[Pasted image 20240625143452.png|400]]
We're going to consider this simplified version of Blackjack/21/Pontoon.
- Rules: You start with two cards, and the idea is to get as close as possible to 21 without going over 21, by asking for a "hit" or a "hold"; You have to beat the dealer, who also has two cards.

We represent this as an MDP:
- We treat the state as having three variables in the state space
	- Current sum of our cards (eg if we have 3 cards, with sum of 17, we say we're in a state with a sum of 17. We don't consider sums < 11, because in those states we just automatically ask for another card with no risk; there's no interesting decision to make)
	- What the dealer's showing (The dealer shows one of their two cards, and you can use that to decide if you ask for another card or not)
	- Do I have a "useable" ace? (meaning it can be either 1 or 11)
- Actions
	- Stick (stop), Twist (hit/draw)
- Reward
	- +1 for win, 0 for tie, -1 for lose or bust; 0 reward for a twist (draw) without going bust

If we apply Monte-Carlo evaluation to this problem, we'll see how to evaluate some naive policy: "==We're going to stick if the sum of our cards is >= 20, otherwise we're always going to twist==." How good or bad is this strategy?
![[Pasted image 20240625144144.png|400]]
- We roll out 10,000 episodes of Blackjack, and 500,000 hands, and provide the procedure we've seen
	- Every time we come across a state, we find the appropriate point in the value function space, and update that value function a little towards the eventual mean for that state.
- After 10,000 episodes, we have a pretty good estimate of the case where we *don't* have an ace, but the estimate where we *do* have an ace is still pretty noisy, because these states are rarer!
- We see that we really need 500k or so episodes to get a good estimate of the value function in those usable ace states.
It looks like our policies both have *high expected return* (the *value function*) at 21, and pretty low reward elsewhere because of our very silly policy. 
==No one told us the structure of the game of blackjack or telling us how many cards there are in the deck; this is just from playing out episodes and learning from that experience.==

![[Pasted image 20240625145052.png]]
==Incremental Mean==: This can be done incrementally too -- we'll move to online algorithms that step-by-step 

The mean can be computed incrementally; you don't have to sum everything and then divide the sum by the counter; you can do it incrementally!
==A lot of algorithms will take this form== of taking a prediction, and then correcting our prediction in the direction of the error between what we thought the (eg) mean was and what we actually observed.


![[Pasted image 20240625145507.png]]
Let's do with Monte-Carlo
- We do this now episode-by-episode without keeping track of the sums of rewards.
- For each state with some reward we've seen
- Ever time we see a state, we measure the return from that state onwards, and take the error between the value function $V(S_t)$ (our expectation of return) and the return we actually observed, and we're going to update our mean estimate for that state a little bit in the direction of the return.
- Episode by Episode, we update our estimation of the mean (here, this being our expected reward of a given state).

We want to move towards algorithms that don't need to maintain statistics, and just incrementally update themselves.
This can be done by forgetting old episodes; sometimes we don't want to take the mean, we want to have a constant step size $\alpha$ that gives us something like an exponential forgetting rate when computing our mean... gives like an exponential moving average of returns we've seen so far.

The concept is still the same:
- We have an estimate of what we thought our mean value is going to be, then we saw an actual return, and we have an error term, and then we move our value function a little bit in the direction of that error.

We're in this setup where we're looking at algorithms where we move a little towards the sample we've seen.

Monte Carlo: Run out episodes, look at complete returns we've seen, update our estimate of the mean return (value function) towards your sample return for each state that we visit.

## Temporal-Difference Learning
- TD methods learn directly from episodes of experience; from interaction with the environment.
- TD is [[Model-Free]]; we dondt need to know MDP transitions/reward structure; we don't need to know how the dealer works, or what's in the deck of cards.
- TD is able to learn from *incomplete episodes*; we don't need to run episodes to completion to learn from them. It does this by *==bootstrapping==!* In bootstrapping, we can take a partial trajectory and take an *estimate* of how much reward we think we'll get from some point to the end of the episode.
- TD updates a guess towards a guess with bootstrapping.
	- We update our guess of the value function by walking some number of steps, and then taking another guess about the rewards we'll get until completion, and then we update our original guess.
![[Pasted image 20240625150418.png|400]]
The goal is the same as before: To learn a value function $v_{\pi}$; to efficiently evaluate the expected return across all states for a given policy $\pi$.
- We're going to try to do this *online*, now; every step, we adjust our estimate of the value function, without waiting for the end.
- In Monte Carlo Learning, we were in some state, had an estimate of the value function, we got our return, we look at the error between our estimate and the actual return, and updated our value function a little bit in the direction of the error.

The simplest TD algorithm is TD(0)
- Here, we update our value V(S_t) towards our *estimated return*: R_{t+1} + gammaV(S_t+1)
	- This estimate consists of two parts, just like in the Bellman Equation
		- The immediate reward
		- plus the discounted value at the next step.
	- Now we're going to substitute this estimated return instead of the real return, but otherwise use the same algorithm we used before for Monte-Carlo.
- This red term in the screen is called the ==TD Target==; we move towards this thing. The whole error term (the difference between our TD target and our previous estimate of that state's return using the value function) is called the ==TD Error==

Why is this a good idea?
- If we're driving our Car down the road
	- if we're doing ==MonteCarlo learning==, we're driving, we see a car coming hurtling towards us, and we think we're going to crash, but we don't actually crash because the car swerves out the way and we don't crash In Monte Carlo, there's no negative reward for this crash because it didn't happen.
	- In ==TD learning==, the same thing happens, we think the car crash is going to happen, so we update the value we had before. We say "oh, that was actually worse than I thought, maybe I should have slowed down the car." We don't need to wait to die to update our value function.

In SuttonAndBarto example:
![[Pasted image 20240625151031.png|300]]
- We're driving along on the way to work... we start off by making some prediction.
	- How much time has passed so far
	- How much time from this point onward
	- The total from these two previous columns
- We leave the office, reach the car and it's raining, and maybe we update our predicted time to go, since we think it might take more time to drive home.
- Maybe a little later, we think traffic isn't so bad, so we think it will now only take 35 minutes in total, etc...

How do we update our value function based on this trajectory of experience?
![[Pasted image 20240625151246.png]]
- In Monte-Carlo
	- Each point along the trajectory... we see the difference between the actual time it took and the total predicted journey time. It actually took 43 minutes.
	- At every moment, we had a prediction of what that total time would be, and there's an error. 
	- At each step along this, we update towards the actual outcome (after waiting to finally get there)
- In TD-Learning
	- At every step, we started off thinking it'd take 30 minutes, and after one step we thought it'd take 40 minutes. And so we can immediately update!
	- At the next step, we thought it was 40 minutes but then things went quite smoothly, so we can immediately pull this guy down...
	- Eventually you get grounded by the actual outcome where you really get there, and that's the end of your episode.

Q: In this example, what's the goal and what are the actions?
A: The reward function is the time that it takes you; the reward is time. The goal is to get to work, and you get -1 per step (if you think of all of these as negative). The actions ... there are no actions. This one is a Markov Reward Process (MRP). The reason it doesn't matter is that we're just trying to evaluate the policy, evaluating how much reward we got from a particular trajectory. ==Remember that we can flatten any MDP into an MRP, given a fixed policy.==

So is TD learning a good or bad idea?
- +: ==TD can learn BEFORE knowing the final output==
	- You don't have to wait to crash and die; you can already start learning after just one step, whereas Monte-Carlo has to wait until the end of every episode to actually SEE the return, and then back it up retrospectively.
- +: ==TD can learn WITHOUT the final outcome==
	- TD can learn from incomplete sequences, whereas MC has to see the termination of an episode to get the actual Return.
	- TD can work in continuing (non-terminating) environments, whereas MC only works for *episodic* (terminating) environments.

![[Pasted image 20240625153336.png]]
One other major difference is this bias/variance tradeoff.
- The actual return is the *unbiased estimate* of the value function. The definition of value is expected return; it's just a sample from the expectation.
- What if we were to use the *True TD Target* (where we drop in the *true* value function, as if told by some oracle)? That would *also* be an unbiased estimate of the value function, because the [[Bellman Equation]] (Expectation) equation tells us this.
	- But we don't have an oracle to tell us the true value $v_{\pi}(s_{t+1})$, so we have to use our best guess so far $V(s_{t+1})$ ... this could be *anything*, it could be wildly wrong! So it's a *based estimate.*
- ==The TD target is biased, but it's much lower variance than the return!==
	- The actual return depends on the immediate reward after one step, and then whatever the environment does, it will noisily take us to some next state, and then noisily transition again... along that whole trajectory of many random variables depending on the environment and the agent's steps.
	- In contrast, in the TD target, we only incur noise from one step -- the $R_{t+1}$ -- that's the only turn that the environment takes (that one transition), and then we have our estimate.
		- The $V_{\pi}(s_{t+1})$ is biased, but not noisy. The environment is noisy (because its probabilistic) -- our estimate is just a function.

![[Pasted image 20240625153900.png]]
- So MC is high-variance, but zero bias
	- As a result of being zero bias, it has good convergence properties, and all algorithms will converge to the correct answer.
	- It's not sensitive to the initial value, because we aren't bootstrapping from some initial value.
	- Simple to grok and use.
- TD has low variance, but some bias
	- Usually more efficient than MC (in terms of # of episodes)
	- TD(0) will converge to $v_{\pi}(s)$
		- But not always with function approximation
			- ((The techniques we've seen so far aren't practical, because the assume that there's a reasonable explorable number of states; but there are many problems where that isn't the case, so we use function approximators.))
	- More sensitive to the initial values, because we bootstrap off of them.

![[Pasted image 20240625154425.png]]
Regarding the error between the true value function and the estimated value function, after some steps; see that TD is more efficient. If you're jumping around too much with your steps though, you run the risk of not converging.
- This is basically showing the effect of ==bootstrapping== on learning efficiency.

![[Pasted image 20240625155015.png]]
MC always converges to the solution with minimum MSE; it always finds the best fit to the *actual* returns we've seen.
In contrast, TD(0) converges to the solution of the MDP that *best explains the data.* Intuitively, it first fits an MDP and then solves that MDP.x


![[Pasted image 20240625155237.png]]
A Markov Decision Process *has the Markov property!*
- TD exploits this by building an MDP structure and solving that structure; it's usually more efficient because it makes use of the Markov Property; we don't need to look blindly at complete trajectories, we can understand the environment in terms of states, and we know that certain properties of those states have to hold.

![[Pasted image 20240625155534.png]]
These different updates take different forms.
- In [[Monte-Carlo Learning]], we start in a state, and implicitly there's some lookahead tree of some sort. From the actions, the we take an action, and the wind blows us somewhere. We repeat... at some points perhaps finding terminations.
- The question is how to use this lookahead tree to figure out the value function at the root node.
- Monte Carlo samples a complete trajectory along the tree.

![[Pasted image 20240625155659.png]]
In [[Temporal Difference Learning|TD-Learning]], the backup is just over one step! We sample an action, we sample the environment to get blown somewhere, and we look at the value function where we ended up, and then back up that value function to the value function at the root node.
- So we don't go all the way to the end like in Monte Carlo; we just look one step ahead, take a sample of what happens ahead a step, and back up that sample.
- Intuition: If we take one step, we're always going to be a little more accurate, because we've seen one step of reality in-between. This involves on step of real reward as well as one step of real dynamics. And then we estimate value function of where we ended up. But because we've included one step of the real dynamics and reward, we're in some sense *more accurate* than where we were before. If we take enough of our steps, we end up grounding ourselves completely in the real reward of what happened (assuming you're in an environment with terminations).

![[Pasted image 20240625155803.png]]
In [[Dynamic Programming]], we also did a one-step lookahead, but we didn't sample; we had to *know* the dynamics, and we used the dynamics to compute a full expectation.
- We *knew* the probability that the environment would take us to A or B... and so we could sum these two whites together, weight them by probabilities, take a max over out actions, and effectively do a complete full backup, giving us the value at the root.

You could alternatively do an exhaustive tree search, going all the way to the end and backing it up.


Let's tease these apart into dimensions that we understand that sort of define our algorithm space.
- ==Bootstrapping==: The idea that we don't use the *real* returns, we use our own *estimate of the real returns* (our value function) to give us an iterative update.
	- Monte-Carlo Learning doesn't bootstrap, it uses the real returns.
	- DP bootstraps (something like the Bellman Equation)
	- TD bootstraps (something like the Bellman Equation)
- ==Sampling==: Whether we sample or take full-width backups
	- Monte-Carlo Learning samples the environment; we don't need a full-width exhaustive search over all things the environment might do; instead, we just sample our policy.
	- DP does not sample, it does full-width updates; it considers every possibility exhaustively and backs them up.
	- TD samples as well

![[Pasted image 20240625160204.png]]
A unified view of [[Policy Evaluation]] techniques.

In the remaining 20 minutes, we want to talk about a spectrum of methods between the shallow and deep backups.
- There's a unifying algorithm having either end as its special case, called [[TD-Lambda]]


## TD($\lambda$) 
- [[TD-Lambda]]
- So far we've seen [[Temporal Difference Learning]], where we've taken a single step of reality, and then looked at the value function after one step... but why not take two steps of reality, or four, or n, and then use *that* value function (and accumulated reward) to back up to our original location?
![[Pasted image 20240625161817.png]]
- Above: TD Learning, or "TD-Zero" is the special case of taking one action, ending up in a new state, and updating our value function accordingly.
- But we could do that for any number of steps! We can always say "I'm going to observe some number of steps of the real world, getting real worlds, and then add on the estimate of return at the state that we end up at"... and this will be a valid estimate of the overall value function of our policy, which we can update towards.

![[Pasted image 20240625162724.png]]
Let's specifically write out what that means
- When n=1, we have the TD(0) estimator, so we just do a single step of real return, and then look at the value function of the state we ended up at.
- For n=2, we take two real steps (with discounting on the second; recall in David's formalism we get reward when leaving a state), and then a twice-discounted estimation of return of where we end up at.
- With n=inf, we end up with our Monte-Carlo estimator.
	- n=inf means that we use all of the steps of the real environment (till termination), which is exactly what Monte Carlo does.

![[Pasted image 20240625163409.png]]
We'd like to come up with an algorithm that efficiently considers all n (because they'll all perform differently for our problem)
- We could average over these n-step returns for different n

![[Pasted image 20240625163535.png]]
Yes we can, and the algorithm is called [[TD-Lambda]]!
- We use the Lambda return, which is the geometrically weighted average of all n, going into the future.
- We have some constant $\lambda$ in \[0,1\] which tells us how much we're going to decay the weighting we have for each successive n.
	- Makes all of our weights sum to one.
- When we get to the end of the episode, we give all of the remaining weight onto the final update.
![[Pasted image 20240625164000.png]]
above: "Lambda Returns"

Q: Why is it Geometrically weighted? Why not some other weighting?
A: It makes for an efficiently-computable algorithm... Geometric weightings are memoryless, which means that we can actually do this is in an efficient way that doesn't require either computing or storing something different for each of our n-step returns. We can do TD-Lambda for the same cost as TD-Learning/TD-Zero -- that's only true for Geometric weighting.


![[Pasted image 20240625164134.png]]
We've seen that we have some forward-view algorithm, where we have to wait until the end of the episode to get our returns, and only when we finish the episode can we get our lambda return.
- This has some of the same disadvantages as Monte Carlo

How can we achieve a backward view... with the nice properties of TD learning (online, every step, from incomplete sequences)?
![[Pasted image 20240625164621.png]]

Imagine that we're a Rat again
![[Pasted image 20240625164956.png]]
Did we get electrecutred because of the Bell, or the Light?
- Most people said Light!

![[Pasted image 20240625165109.png|400]]
- See: [[Credit Assignment Problem]]
- With [[Eligibility Trace]]s, we basically look over time at the states we visit.
- Graph shows Eligibility Trace for a single state
- Every time we visit a state, we increase the Eligibility Trace, and when we don't visit it, we decrease it exponentially. This combines our frequency and recency heuristics together.
- When we see an error, we can now update the value function in *proportion to the eligibility trace*.


![[Pasted image 20240625165309.png]]
We update our value function in proportion to *both* the TD error $\delta_t$ and the eligibility trace E_t(s).
- The things which we think ahve the most eligibility/responsibility for that error are updated the most.
- ![[Pasted image 20240625165359.png|400]]

How does this relate to algorithms we've seen so far?
![[Pasted image 20240625165502.png|300]]
![[Pasted image 20240625165512.png|300]]
In an episodic environment, this means that we actually end up getting the same set of updates.... the sum of the updates is the same as the forward view of TD Lambda.

![[Pasted image 20240625165634.png]]

