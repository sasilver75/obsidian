https://www.youtube.com/watch?v=lfHX2hHRMVQ&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=2

---

Markov Processes (aka Markov Decision Processes)
Markov Reward Processes (by adding in Rewards)
Markov Decision Processes (by adding in Actions)
Extensions to MDPs (might not have a change to talk about them; partial observability, etc.)

---

[[Markov Decision Process]]es formally describe an environment for reinforcement learning. 
- We're going to start with the nice case where the environment is *fully observable* (we're *told* the state; nothing is hidden away).
- We can say that the state given to the agent completely characterizes the process.
- Almost all RL problems can be formalized in some way as a Markov decision progress
	- Optimal control primarily deals with continuous Markov Decision Process (MDP).
	- Partially-observable problems can be *converted* into MDPs.
	- Bandits (you have set of actions, you take an action, get a reward, and that's the end of the process; like serving an ad where it gets clicked or not) are MDPs with one state.

Essential idea of an MDP is the [[Markov Assumption|Markov Property]]
![[Pasted image 20240623170758.png|400]]

![[Pasted image 20240623171208.png]]
The 0th row in the state transition matrix describes being in State S_0 and (considering the cells across columns in row 0 as the S_1 states it will transition to, including S_0 again) the probability of transitioning to another state.
- We can follow through a number of steps and keep drawing from the state transition matrix and get some episodes/trajectories.

So a Markov Process (or [[Markov Chain]]) is a sequence of random States S_1, S_2, ... with the Markov Property.
![[Pasted image 20240623171322.png]]
Above: Note that there's no actions or rewards yet. The process can be described by the state space and the transition probabilities between the states. It's not something where we've got anything agentic involved.
![[Pasted image 20240623171355.png|400]]
A transition matrix describing David's course 😜
Here's the transition matrix for the above:
![[Pasted image 20240623171654.png|300]]
And some episodes/trajectories like look like
![[Pasted image 20240623171707.png|300]]

Question: These seems thing to work when the probabilities stay the same over time? What if the Facebook loop gets less additive as you keep looping around it?
A: One answer is to have a *Non-Stationary* Markov Decision Process. Or maybe if you have non-stationary dynamics, you just have a more *complicated* MDP... Which maybe has a counter telling you how long you've been in the Facebook state, for instance. Doesn't change the structure of it being a Markov process.

So far, we haven't really talked about 

## Markov Reward Processes (adding Rewards)
- Now we have a Markov Reward Process, with is like a Markov Process with value judgements of how good it is to be in a particular state/in a particular sequence.
- In our usual Markov Process/Chain, we had:
	- Our ==state space== $\mathcal{S}$
	- Our ==transition matrix== $\mathcal{P}$
- Now we're going to also add:
	- ==Reward function== $\mathcal{R}$: Starting in state s, how much reward do we get from that state? It's just the *immediate* reward. We care about maximizing the cumulative sum of these rewards.
	- Our ==discount factor== $\gamma$: Some number in the \[0,1\] range

The return $G_t$ is the total discounted reward from time-step t (to the end).
![[Pasted image 20240623172706.png|300]]
Our goal is to maximize G.
- There's no Expectation here, because we're just talking about a random sample; *one* sample from our Markov Reward Process, of the rewards that we get going through that sequence. *Later* we'll introduce expectations.

So this discount factor $\gamma$ tells us about the present value of future rewards -- about how much we care *now* about rewards we'll get in the *future*.
- 0 means myopic and maximally-short-sighted, meaning we basically zero out anything beyond our current timestamp. We prefer rewards now,.
- 1 means maximally-far-sighted, meaning we care equally about rewards in the future. We're indifferent to when rewards arrive.
 
The value of receiving reward R after k+1 time steps in $\gamma^kR$ 

Why do we use discount factors?
- To represent the fact that we don't have a perfect model of reality (future estimations things are riskier; uncertainty about the future!)
- It's mathematically convenient to discount rewards!
- Avoids infinite returns in cyclic Markov processes (Eg reward hacking)
- If the reward is financial, immediate rewards may earn more interest than delayed rewards (money now is usually worth more than money later).
- Human/animal behavior generally shows preference for immediate reward.
- It's sometimes possible to use *un-discounted* Markov reward processes (eg gamma=1), e.g. if we know that all sequences terminate.


## Value Functions
- The value function v(s) is the quantity we really care about -- it's the long-term value from being in a certain state s (until termination).
- It's just the expected return if you start in that state!

$v(s) = \mathbb{E}(G_t|S_t=s)$   

In an MRP, there's no concept of maximizing; we just measure how much reward we get (it's only once we get to MDPs that we get the ability to take actions to maximize this concept).

![[Pasted image 20240623174935.png|300]]
Take a bunch of samples and just take the average of the return G_t values (observed samples of our random process), which gives you a good estimate of the value function (which is an expectation of return).
(Recall G_t is the return from timestep t onwards)

## Bellman Equation for MRPs
- The [[Bellman Equation]] is "Maybe the most fundamental relationship in Reinforcement Learning"
- The value function $v(S_t)$ can be decomposed into two parts (and it's a recursive definition):
	- The immediate reward $R_{t+1}$ 
	- Discounted value of successor states  $\gamma v(S_{t+1})$ 

![[Pasted image 20240623175548.png|300]]
(Q: Why t+1 instead of t? There are actually different conventions about how to do this. We say the action goes into the environment, then a timestep happens, and then we receive rewards -- this is the Sutton and Barto convention. It's just a convention and doesn't effect the result.)

![[Pasted image 20240623180618.png]]
We're introducing a vector representation of the value function, which is.... us forming a column vector where each element o the column vector contains the value function of a specific state, for all states from 1...n.
- Now, the value function we start with (column vector) is equal to the reward function (also vectorized, showing the reward of exiting states 1...n) plus the discounted states we can end up in.
- "The value of this state is equal to the immediate reward plus...the transition matrix (all places we might transition to), and then the places we might end up."
![[Pasted image 20240623181140.png]]
The bellman equation is a linear equation, so it can be solved directly (assuming that our matrix is small enough to invert). The computational complexity is n^3 for n states, so it's not a good solution method for large processes. We'll look later at:
- Dynamic programming
- Monte-Carlo evaluation
- Temporal-Difference learning


## Markov Decision Processes (Adding Actions)
- This is the thing that we actually use in reinforcement learning.
- In a [[Markov Decision Process]] (MDP), we'll add in *actions!* Until now, there's been nothing to *do* -- we just get dropped into a  state and randomly sample next states using our state transition probability matrix. Now, we're adding in some agency!
- Now we're adding an *action* space $\mathcal{A}$ !
- Now we'll say that our transition probability depends on which action we take, too!
	- Where we end up depends (probabilistically) on which state we're in, and which action we take.
![[Pasted image 20240623181523.png|300]]

Now, what we're going to do is redo our student Markov process as an *MDP*, where there are decisions we can make.
![[Pasted image 20240623181601.png|250]]
In this example, the transitions are deterministic. The Pub is the only place in the graph with any randomness.

Our goal as usual is to find a path through our MDP that maximizes the rewards that we get.
To do that though, we need to formalize what it means to make decisions!

A ==Policy== $\pi$ is a distribution over actions, given states. They fully define the behavior of an agent. MDP Policies depend on the current state (not the history), so policies are *stationary* (time-independent!). The only thing they depend on are the state we're current in.

$\pi(a|s) = \mathbb{P}(A_t=a|S_t=s)$ 

We usually make this a stochastic transition matrix, which allows for exploration.

![[Pasted image 20240623182021.png|300]]
Given an MDP and a Policy...
- We can always recover a Markov Process / Markov Reward Process from our Markov Decision Process.
	- If we just draw a a sequence of states using our policy... that sequence of states we draw is actually a Markov process itself.
	- If we just look at the sequence of states and rewards we receive (once we fix the policy), that sequence is a Markov reward process.

What's central is the concept of the Value Function (which we had for our MRP, but there were decisions involved there) in MDPs. Now that we can make decisions, let's talk about the *==state-value function==!*  $v_{\pi}(s)$ tells us how good it is to be in state s, and then following policy $\pi$.

We're also going to define a second type of value function, called an *==action-value function==*. This tells us how good it is to take a particular action, from a particular state. This is the thing we intuitively care about when we talk about choosing what action to take. 
$q_{\pi}(s,a) = \mathbb{E}[G_t|S_t=a,A_t=a]$

![[Pasted image 20240623182525.png|400]]
![[Pasted image 20240623184427.png|400]]
If I'm in one state, and I take a n action from there, I get an immediate reward, and then I look at where I end up, and I ask "what's the action value function of the state I end up in, under the action I would pick from that point onwards?"

These are all basically saying "The value of the thing given an action is the value of taking that action plus {the recursive value of the thing we end up at, given the same definition}"

Earlier, we talked about flattening MDPs into Markov Reward Processes by defining average state transition dynamics and average reward functions.
- The bellman equation gives us a description of the equation that we can solve, and when we solve it, we've got the value function.
- ![[Pasted image 20240623185530.png|300]]



In an MDP, we care about determining the best way to behave!
![[Pasted image 20240623185601.png|400]]
- Let's talk about the essential problem we want to care about, which is finding the best behavior (policy) in an MDP. We want to find the best path through the system; the best way to solve our problem.

![[Pasted image 20240623190046.png]]
- Let's define $v_*(s) = max_{\pi}v_{\pi}(s)$ as the maximum possible reward that we can extract from the system, under any policy.
- q_* tells us the maximum amount of reward we can start, starting in state s, and taking action a. After taking that action, what's the most amount of reward we can get from that state onwards?
	- If you know q_\*, then you're basically done. You just greedily follow maximal q_\* moves, from any state.  We can think of solving an MDP as finding the optimal q star.

How do we compare policies?
![[Pasted image 20240623191427.png|300]]
- What does it mean for one policy to be better than another policy?
- We define a partial ordering over policies, where pi is >= pi' if the value function for that policy is >= the value function for the other policy in ALL states!
	- Meaning: *it can't be worse.* It's not possible to say that policy A is better than policy B if policy B is better in even *one* state.

There's a nice theorem:
- For any MDP, there EXISTS at least one optimal policy $\pi_*$ that IS BETTER THAN OR EQUAL TO ALL OTHER POLICIES (that extracts the maximal juice from the MDP)! This is pretty convenient.
- All optimal policies $\pi_*$ achieve the optimal value function $v_*(s)$
	- (Imagine there's two separate actions that take you from s to s'; it doesn't matter** which one you choose.)
- All optimal policies achieve the same optimal action-value function. 
	- (This isn't as obvious to me... especially if one of two policies has a "wrong" q value for some (s,a) that wouldn't be used anyways.)

![[Pasted image 20240623191830.png|400]]
If we know q_\*(s,a), we automatically have the optimal policy.

How do we get q star?
- We start at the final state, and work backwards while looking at rewards.
- The bellman optimality equation for v_* helps us solve

[[Bellman Equation]]
![[Pasted image 20240623192456.png]]
The value of the state is the maximum value of taking the best of {the available actions, receiving its reward, landing (probabilistically) in some next state, and optimally exploiting that state in the same manner(recursive)}.

![[Pasted image 20240623193423.png]]
Solving this requires iterative solution methods like [[Value Iteration]], [[Policy Iteration]], [[Q-Learning]], or other methods.

Q: Whats the intuition behind the bellman equation?
A: The bellman optimality equation... the q star or v star tell you the maximum amount of score you can get from a screen. And the intuition is to look at the principle of optimality, which tells you to behave optimally for one step, and then behave optimally for the remaining trajectory. So now you just need to figure out how to behave optimally for one step, which is to optimize over the value functions of the places you might end up. Just by breaking down your trajectory into these two parts (optimal decision at one step, and optimal decision from then-on), we can make some progress.

Extensions to MDPs
- Infinite and continuous MDPs
- Partially-observable MDPs
- Undiscounted, average-reward MDPs

We've mostly defined the RL problem using a markov decision process, and next time we'll start to talk more about how to actually solve them.

