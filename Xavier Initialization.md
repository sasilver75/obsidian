---
aliases:
  - Glorot Initialization
---
2010
Xavier Glorot, [[Yoshua Bengio]]
[Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
Takeaway:

See also: [[He Initialization]] (2015)

---

A weight initialization technique used to help prevent the [[Vanishing Gradients]]/[[Exploding Gradients]] problem in neural nets, aiming to initialize the weights in a way that maintains the same variance of activations and gradients across layers.
- For a layer with n inputs and m outputs, weights are initialized randomly from a uniform distribution in the range $[-limit, limit]$, where $limit=\sqrt{6/(n+m)}$ 
- The weights are set so that the variance of the inputs is equal to the variance of the outputs, helping with faster convergence and petter performance.
- Works well with linear and sigmoid activations, but might need some adjustment for ReLU (in contrast, [[He Initialization]] is specifically designed for ReLU activation functions.)


Abstract
> Whereas before 2006 it appears that deep multilayer neural networks were not successfully trained, since then several algorithms have been shown to successfully train them, with experimental results showing the superiority of deeper vs less deep architectures. All these experimental results were obtained with ==new initialization== or training mechanisms. ==Our objective here is to understand better why standard gradient descent from random initialization is doing so poorly== with deep neural networks, to better understand these recent relative successes and help design better algorithms in the future. We first observe the influence of the non-linear activations functions. We find that the logistic sigmoid activation is unsuited for deep networks with random initialization because of its mean value, which can drive especially the top hidden layer into saturation. Surprisingly, we find that saturated units can move out of saturation by themselves, albeit slowly, and explaining the plateaus sometimes seen when training neural networks. We find that a new non-linearity that saturates less can often be beneficial. Finally, ==we study how activations and gradients vary across layers and during training, with the idea that training may be more difficult when the singular values of the Jacobian associated with each layer are far from 1==. ==Based on these considerations, we propose a new initialization scheme== that brings substantially faster convergence.