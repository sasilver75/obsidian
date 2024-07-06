---
aliases:
  - SGD
---
Other optimizers: [[Adagrad]], [[RMSProp]], [[Adam]], [[AdamW]], etc.

It might be tempting to think that every later algorithm is better than its predecessors, but that's not always true (eg [[Stochastic Gradient Descent|SGD]] might result in models that generalize better than Adam, because SGD pushes us out of sharp minima better than [[Adam]]))

![[Pasted image 20240706002633.png]]
Often finds better solutions than regular [[Gradient Descent]], interestingly. A common explanation is that SGD's extra noise increases the chance of escaping saddle points, escaping "bad", narrow local inima.