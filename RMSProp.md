---
aliases:
  - Root Mean Square Propagation
---
References:
- Video: [Sourish Kundu's Who's Adam and What's He Optimizing?](https://youtu.be/MD2fYip6QsQ?si=l1hj8bWgbRM181Xk)

![[Pasted image 20240705222226.png]]
$\eta$ is the learning rate; $\epsilon$ is to avoid divide-by-zero errors
- The fraction helps modulate the effective LR based on the parameter's gradient. Parameters with a history of larger gradients (higher v), the denominator will be larger, resulting in smaller updates; we dampen the updates to parameters with larger gradients -- a normalization balancing varying size gradients.

Like with [[Adagrad]], the algorithm keeps some memory of previous gradients, and the "v" term is updated from one step to the next according to some discount parameter Beta that determines how much of the previous v term is remembered.
- So when a large gradient is encountered, v is modified so that the effective LR is scaled down, and when a small gradient is encountered, it's scaled up.

![[Pasted image 20240708121129.png]]
(Still has the property of Adagrad where it optimizes parameters simultaneously, rather than the ones with the largest gradient first; so it's capable of accommodating for the difference in gradient size across parameters.)

This lets us retain some of the benefits of a decaying learning rate, without suffering some of the problems that I describe in [[Adagrad]] (where the LR peters out before we get to a good parameter setting).
- When the surface is relatively flat, w takes big jumps; when the surface is steep, w takes small jumps to avoid leaping over the target minima -- but it can still get stuck in bad minima.
![[Pasted image 20240705224220.png]]
To deal with this, [[Adam]] arrived to the scene!

RMSProp use in RL: As our agent explores new areas of the environment, the distribution of training data tends to shift; the best action at the beginning of the game might not remain the best action at the end of the game. This means that samples in RL are not independent and IID; in such cases, momentum might be a detriment to performance, because samples during training might vary wildly from eachother.