---
aliases:
  - Root Mean Square Propagation
---


![[Pasted image 20240705222226.png]]

Like with [[Adagrad]], the algorithm keeps some memory of previous gradients, and the "v" term is updated from one step to the next according to some discount parameter Beta that determines how much of the previous v term is remembered.
- So when a large gradient is encountered, v is modified so that the effective LR is scaled down, and when a small gradient is encountered, it's scaled up.

This lets us retain some of the benefits of a decaying learning rate, without suffering some of the problems that I describe in [[Adagrad]] (where the LR peters out before we get to a good parameter setting).
- When the surface is relatively flat, w takes big jumps; when the surface is steep, w takes small jumps to avoid leaping over the target minima -- but it can still get stuck in bad minima.
![[Pasted image 20240705224220.png]]

To deal with this, [[Adam]] arrived to the scene!