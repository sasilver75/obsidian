---
aliases:
  - Adaptive Moment Estimation
---
References:
- Video: [Sourish Kundu's Who's Adam and What's He Optimizing?](https://youtu.be/MD2fYip6QsQ?si=l1hj8bWgbRM181Xk)

See previous: [[Adagrad]], [[RMSProp]]
Improved upon by: [[AdamW]]

Aside: It might be tempting to think that every later algorithm is better than its predecessors, but that's not always true (eg [[Stochastic Gradient Descent|SGD]] might result in models that generalize better than Adam, because SGD pushes us out of sharp minima better than [[Adam]]))

![[Pasted image 20240705224552.png]]
The (effective) LR is again scaled by a v term that takes the same form as in [[RMSProp]], except this time, the gradient jump is parallel to some vector $m$, which we can think about as an *actual* velocity term. The Mhat and Vhat terms arise to account for bias correction. Bias arises because M and V are initialized to zero at the start of training (so at the beginning, we're scaling down gradients significantly. Because B1 and B2 are typically set to ==.9== and ==.99== respectively (with eps=10^-8), it inherently delayed the impact of new gradients on the optimization process. The bias correction for Mhat and Vhat help to mitigate these issues by making the gradient updates not be overshadowed by M and V being initialized to zero; this helps tremendously in the initial part of optimization. As training goes on, the exponent t in the denominators continue to increase, making Mhat and Vhat's denominator approach 1, nullifying it's effect)
- So this marries some of the benefits of [[RMSProp]] and of classical momentum.
	- The LR is adjusted according to the squared magnitude of recent gradients, and a velocity term is used to mimic the smoothing properties of momentum.

w is modified not only using m and v directly (in v), but also slightly modified terms scaled by $1-B^{t+1}$. 
![[Pasted image 20240705224552.png]]
Because Beta is in 0..1, this denominator edges closer to 1 with every timestep. Initially it magnified the m and v terms, because they'd be initially biased to 0 because thats where the initial m and v are initialized.


