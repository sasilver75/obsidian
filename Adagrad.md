![[Pasted image 20240705221804.png]]
(Stabilizer is just a small number that we add to the denominator so as not to divide by zero)

For each individual weight, the gradient step is scaled according to the square root of the sum of prior encountered gradients.
The logic is similar to that of decaying learning rates, where the LR is reduced by some type of schedule as training progresses... with the idea that as training progresses, we're closer to the target minimum.
- The *trouble* with decaying learning rates is that we don't actually *know* if we're getting closer to the target minimum -- we may not be! The LR may fizzle out even though our target is far away.

Adagrad puts a spin on the decaying LR idea by scaling *each parameter* differently according to the amount that that *specific* parameter has changed during training (evaluated as the sum of squared prior gradients). Idea: If a parameter has changed significantly, then it must have made a lot of progress towards the target (and vice versa).

![[Pasted image 20240705221838.png]]
(Vector notation)
(Here, it's kind of the opposite of velocity), but it's updated at each timestep by incrementally adding the squared gradient.

![[Pasted image 20240705222009.png]]
See AdaGrad taking a more direct route than [[Stochastic Gradient Descent|SGD]] in this example because it senses that it's made a good amount of progress in the w2 axis, so it scales down the w2 gradient and scales up the w2 gradient, resulting in a curve that's more balanced.

Weakness: It can decrease the effective learning rate in response to the loss landscape, but it cannot later *increase it*.
- [[RMSProp]] attempts to address this by allowing for effective learning rates that can both decrease *and* increase.