---
aliases:
  - TD-Learning
---
References:
- [[David Silver RL (4) - Model-Free Prediction]]
- [[David Silver RL (5) -  Model-Free Control]]
- Video: [Mutual Information's Temporal Difference Learning and Q Learning](https://www.youtube.com/watch?v=AJiG3ykOxmY)

Almost all RL algorithms fall within the ==Generalized Policy Iteration== (GPI) paradigm:
- Policy Evaluation (learning $v_{\pi}$)
- Policy Improvement (using v_{\pi} to improve our current policy)

One of the requirements of [[Monte-Carlo Learning]] is that episodes must be completed before values can be updated. This means if episodes are long, learning can be really slow! In reality, the transitions and rewards *within an episode* have useful information that we could be learning from -- if we could learn from those, we can change our policy *within an episode*, which can be pretty efficient.

Let's compromise this idea that we need to observe until the end of the episode. 

![[Pasted image 20240627183059.png|500]]


![[Pasted image 20240627184008.png|500]]
MC minimizes the MSE over the data (known property of taking an average)
TD's criteria is to maximize the likelihood of the Markov Reward Process (MRP); the thing we're assuming generated our data.
- An MRP is defined by a distribution $p(s', r| s)$. We form our estimate for the probability as the count of transitions from (s -> s', r), divided by the count of visits to s. We use this to estimate the expected return from any state. This is equivalent to 1-step batch TD!
==When we use TD, it's as if we directly model the data-generating process; MC, on the other hand, is just an averaging of the returns from each state, which doesn't model the MRP -- In fact, this does have its advantages: it's more robust to violations of the Markov Property; but that robustness comes at the cost of doing less well when those assumptions *do* hold.==



