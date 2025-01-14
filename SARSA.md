References:
- Lecture: [[David Silver RL (5) -  Model-Free Control]]
- Video: [Mutual Information's Temporal Difference Learning](https://youtu.be/AJiG3ykOxmY?si=-YiCTdSHpv_e4jta)

An algorithm for [[On-Policy]], [[Model-Free]] [[Temporal Difference Learning]], with [[Q-Learning]] being the [[Off-Policy]] comparison in the space.

[[SARSA]] is [[On-Policy]] TD Control, a form of model-free control, so we apply our algorithm to action-values Q(s,a), not state-values V(s).
"So while [[Q-Learning]] is [[Off-Policy]] [[Temporal Difference Learning|TD-Learning]] of the Q function, [[SARSA]] is [[On-Policy]] [[Temporal Difference Learning|TD-Learning]] of the Q function!"

----

$Q_{new}(s_k,a_k) = Q_{old}(s_k,a_k) + \alpha(r_k + \gamma Q(s_{k+1}, a_{k+1}) - Q_{old}(s_k,a_k))$

Notice 
This means we have to be taking our best policy actions (a_k, a_k+1) at every step, or else this will degrade our estimate. Because our quality function is always the estimate of the quality of being in that next state and taking that next action, assuming I do the best thing forever after. so if I don't plug in my best possible a_k+1 and take my best a_k to get the r_k reward, then this will be a suboptimal estimate of the quality function.
Very subtle difference!
- This is called on-policy because you have to enact the optimal policy for this to work and not degrade.
- SARSA can actually work for any of the TD-variants: TD-0, TD-1, TD-N.

![[Pasted image 20250113211854.png]]
Above: [[Q-Learning]] vs [[SARSA]]
- Q-Learning is what's called [[Off-Policy]], and we enable it to be off-policy by replacing our a_k+1 with a max over $a$. Because the quality function and value function are always assuming that you do whatever is optimal in the future. If you didn't maximize over a in the top, then if you took an off-policy a (some sub-optimal a), this would actually degrade your quality function over time. 
- But because we optimize over $a$ in our quality function at the next step, we can take suboptimal actions $a_k$ to explore and learn from off-policy experiences! This is a huge benefit. This is one of the things that allows Q-learning to learn from a lot of different types of experiences!
	- Q-Learning
		- Better, faster learning
			- Often faster because you can explore more
		- Higher (?) variance
		- Can learn from imitation and [[Experience Replay]] (Replaying your past experiences, even if they were suboptimal).
		- Lots of strategies to introduce randomness into what action you take -- [[Epsilon-Greedy]] is a simple example.`

- SARSA has benefits too!
	- Often safer in training because it's always going to do what you think is the best thing to do -- it's not going to randomly explore. Because it's [[On-Policy]], you're always going to do what you think is the best thing, even though it's more conservative -- so this will learn less quickly and less optimally, but probably safer.
		- If you're teaching a teenager to drive, you probably want to do more like SARSA than Q-Learning.
	- Because it's always on-policy, it will typically have more cumulative rewards during the learning process, if you care about that (it's not making occasional bad moves to gain experience).



----


![[Pasted image 20240627184515.png|300]]

With respect to our [[Temporal Difference Learning|TD-Learning]] algorithm, we change our updates a little bit.
![[Pasted image 20240627184409.png|400]]

What's the advantage of using n > 1, in n-step Sarsa?
- The best n is going to be problem-dependent.


![[Pasted image 20240625232913.png]]

------------
Now let's get into the full-fledged algorithms that do all of [[Generalized Policy Iteration]]: they do both [[Policy Iteration]] and [[Policy Improvement]].
We'll talk about on-policy, model-free TD-Control, [[SARSA]].

Model free-control means that we need to use Q(s,a), not V(s), since we don't know what the environment transition dynamics p(s'|s,a)

Redefine:
- Return is now the sum of discounted actual rewards for a few steps, followed by the bootstrap