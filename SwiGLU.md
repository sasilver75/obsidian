February 12, 2020
A [[Noam Shazeer]] Solo Classic 🗿
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

A combination of the [[Swish]] and [[GLU]] activation functions, featured as part of the 2022 [[PaLM]] architecture, and later [[LLaMA 2]] and [[LLaMA 3]].


---

$SwiGLU(x, W, V, b, c) = Swish(xW + b) \odot (xV + c)$ 
- $x$ is the input
- $W$ and $V$ are weight matrices
- $b$ and $c$ are bias vectors

It looks very similar to [[GLU]], except it uses [[Swish]] for its nonlinear/gating functionality, rather than a [[Sigmoid Activation Function|Sigmoid]].
- Combines the benefits of Swish (smoothness, non-monotonicity) with the gating mechanism of GLU, allowing for more complex and flexible representations.
Often performs better in practice than [[GLU]], which uses a Sigmoid for its gating.
- Sigmoid: Always positive, bounded between 0 and 1
- Swish: Can be negative, unbounded above

![[Pasted image 20240707174104.png]]
