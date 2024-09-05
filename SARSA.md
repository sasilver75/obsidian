References:
- Lecture: [[David Silver RL (5) -  Model-Free Control]]
- Video: [Mutual Information's Temporal Difference Learning](https://youtu.be/AJiG3ykOxmY?si=-YiCTdSHpv_e4jta)

An algorithm for [[On-Policy]], [[Model-Free]] [[Temporal Difference Learning]], with [[Q-Learning]] being the [[Off-Policy]] comparison in the space.

[[SARSA]] is [[On-Policy]] TD Control, a form of model-free control, so we apply our algorithm to action-values Q(s,a), not state-values V(s).

![[Pasted image 20240627184515.png|300]]

With respect to our [[Temporal Difference Learning|TD-Learning]] algorithm, we change our updates a little bit.
![[Pasted image 20240627184409.png|400]]

What's the advantage of using n > 1, in n-step Sarsa?
- The best n is going to be problem-dependent.


![[Pasted image 20240625232913.png]]