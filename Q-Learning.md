References:
- Lecture: [[David Silver RL (5) -  Model-Free Control]] 
- Video: [Mutual Information's Temporal Difference Learning and Q-Learning](https://youtu.be/AJiG3ykOxmY?si=-YiCTdSHpv_e4jta)

The goal is to learn Q(s,a), and then take the action *a* at state *s* that maximizes Q(s,a). This means that at every state, we output a Q(s,a{i}) for each discrete {i} action.
	- Complexity: Can model scenarios where the action space is discrete and small; can't handle continuous action spaces.
	- Flexibility: The Policy is deterministically computed from the learned Q function by maximizing the reward; it doesn't learn stochastic policies. 

"Essentially just [[Temporal Difference Learning]] but on the Q function"
- Indeed, Claude says that ==*Q-Learning* IS indeed a TD learning algorithm==.

$Q_{new}(s_k,a_k) = Q_{old}(s_k,a_k) + \alpha(r_k + \gamma \underset{a}{max}Q(s_{k+1}, a) - Q_{old}(s_k,a_k))$ 

We have an estimate $Q_{old}(s_k,a_k)$ of what being in a state s and taking an action is, and so we measure the actual reward and future Q function of the place we find ourselves in... $r_k + \gamma \underset{a}{max}Q(s_{k_1}, a)$ and use that as the TD Target Estimate (observed cumulative reward), and we use that to generate an error signal (TD Error) to update our old Q function.

So if we got more reward than we thought we would, I should increase my Q function at that (s,a). If I experience more reward than I expected, I should have a positive instead in the Q function at my (s,a)

Note that in the TD target, we're maximizing over a; we're taking some action (a) to get our reward r_k, but it doesn't have to be our optimal on-policy action; I could do some random action to get this r_k; but for the next state s_k+1, we're going to maximize over our a.
- We cal this off-policy: I can actually take suboptimal actions $a$ to get this $r_k$ reward... but then I do need to $\underset{a}{max}$ to get my best Q function in the TD target.
- This has some benefits: I can learn from experience, replay old experiences (even when I know their aren't using an optimal policy $\pi$ ) -- I can watch a grandmaster play and learn from their actions! With off-policy learning, we can still learn from actions that are drawn from a policy other than our own target policy.

So while [[Q-Learning]] is [[Off-Policy]] [[Temporal Difference Learning|TD-Learning]] of the Q function, [[SARSA]] is [[On-Policy]] [[Temporal Difference Learning|TD-Learning]] of the Q function!
See: [[SARSA]]

------
Conversation with Claude about [[SARSA]] vs [[Q-Learning]]:

Both are examples of [[Temporal Difference Learning|TD-Learning]] algorithms, which are fundamentally about learning from differences between consecutive predictions. We update our current prediction based on immediate reward(s) and subsequent predictions (bootstrapping the remainder of a trajectory with an estimate).

- Both algorithms try to learn a $Q(s,a)$ function
	- They differ in how they update this Q-function
- Q-Learning is an "off-policy" algorithm:
	- When it updates its Q-values, it considers what the ==best== possible next action would be, regardless of what action its current policy might *actually* take.
	- Q Learning update:
		- $Q(s,a) = Q(s,a) + \alpha[r + \gamma \underset{a}{max}Q(s', a') - Q(s,a)]$
		- This $\underset{a}{max}Q(s', a')$ means that we're considering the *maximum Q-value* over all possible next actions, even if our policy would never choose that action. This makes Q-Learning more aggressive in finding the optimal policy.
- SARSA on the other hand is an "on-policy" algorithm,
	- SARSA Update:
		- $Q(s,a) = Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$ 
		- Notice that instead of taking the maximum over the next actions, SARSA just uses Q(s',a'), using the Q-value of the action that our current policy would actually choose.
		- This makes SARSA more conservative and potentially safer in some environments.
- Example: A robot learning to navigate along a cliff, where there's a safe path further from the cliff that takes longer, and a risky path near the cliff's edge that' shorter.
	- Q-Learning might initially learn to take the risky path because it only considers the best-possible outcome (reaching the goal quickly) without considering that its exploration policy might lead to falling off the cliff.
	- SARSA, because it considers the actual policy being followed (which includes exploration and potential mistakes) would tend to learn the safer path away form the cliff. It "knows" during learning that random exploratory actions might cause it to fall off the cliff, so it learns to stay from the edge entirely.

Relative Strenghts:
- Q-Learning can find optimal policies even while following an exploratory policy, which can be more efficient in safe environments. It separates the exploration policy from the learning process.
- SARSA tends to find safer policies that account for exploration, which can be better in environment where mistakes during learning are costly. It learns the best policy for the specific way it explores the environment.



------


![[Pasted image 20240625232932.png]]

![[Pasted image 20240627185925.png]]