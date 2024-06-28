References:
- Lecture: [[David Silver RL (5) -  Model-Free Control]] 
- Video: [Mutual Information's Temporal Difference Learning and Q-Learning](https://youtu.be/AJiG3ykOxmY?si=-YiCTdSHpv_e4jta)

The goal is to learn Q(s,a), and then take the action *a* at state *s* that maximizes Q(s,a). This means that at every state, we output a Q(s,a{i}) for each discrete {i} action.
	- Complexity: Can model scenarios where the action space is discrete and small; can't handle continuous action spaces.
	- Flexibility: The Policy is deterministically computed from the learned Q function by maximizing the reward; it doesn't learn stochastic policies. 



![[Pasted image 20240625232932.png]]

![[Pasted image 20240627185925.png]]