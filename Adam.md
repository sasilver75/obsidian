---
aliases:
  - Adaptive Moment Estimation
---


See previous: [[Adagrad]], [[RMSProp]]
Improved upon by: [[AdamW]]
Aside: 
It might be tempting to think that every later algorithm is better than its predecessors, but that's not always true (eg [[Stochastic Gradient Descent|SGD]] might result in models that generalize better than Adam, because SGD pushes us out of sharp minima better than [[Adam]]))

![[Pasted image 20240705224447.png]]
The (effective) LR is again scaled by a v term that takes the same form as in [[RMSProp]], except this time, the gradient jump is parallel to some vector $m$, which we can think about as an *actual* velocity term.
- So this marries some of the benefits of [[RMSProp]] and of classical momentum.
	- The LR is adjusted according to the squared magnitude of recent gradients, and a velocity term is used to mimic the smoothing properties of momentum.

w is modified not only using m and v directly (in v), but also slightly modified terms scaled by $1-B^{t+1}$. 
![[Pasted image 20240705224552.png]]
Because Beta is in 0..1, this denominator edges closer to 1 with every timestep. Initially it magnified the m and v terms, because they'd be initially biased to 0 because thats where the initial m and v are initialized.


