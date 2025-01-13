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
