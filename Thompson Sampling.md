References:
- Video: [Luis Serrano's Thompson Sampling](https://youtu.be/nkyDGGQ5h60?si=OtuteLjHa6ROI_wx)

A probabilistic algorithm used for solving multi-argmed bandit problems and other exploration-exploitation dilemmas in machine learning.

In a [[Multi-Armed Bandit]] situation, we keep track of the [[Beta Distribution]] corresponding to our wins and losses for each slot machine.

![[Pasted image 20240706012121.png]]
The question is: How do we select which machine to play next?

Imagine we've played the machines many times, and these are our following beta distributions:
![[Pasted image 20240706012157.png]]
We're going to make the machines compete with a little bit of randomness added, with the idea that the strongest machines should have more probability of winning, but *also* that machines that we haven't explored well should be given a slightly higher probability of winning.

Process:
- We draw a random sample from each bandit's distribution, and choose the bandit with the highest probability value. Bandits with wider distributions have a chance to produce very high examples, encouraging exploration
	
![[Pasted image 20240706012408.png]]
In this case, the third machine wins; we play that machine next, update its beta distribution, and continue.

So why is this technique good?
Let's consider these three machines:
![[Pasted image 20240706012514.png]]
