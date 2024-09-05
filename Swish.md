October 16, 2017
[[Google Research]], including [[Quoc Le]]
[Searching for Activation Functions](https://arxiv.org/abs/1710.05941)

----

$Swish(x) = x * \sigma(\beta x)$
- $\sigma$ is the sigmoid function
- $\beta$ is a learnable parameter (can be fixed to 1)

Swish is smooth and monotonic, unbounded above and bounded below.
Often outperforms [[Rectified Linear Unit|ReLU]] in deep networks, helping to mitigate the vanishing gradient problem. 
![[Pasted image 20240707173251.png|300]]
(Above: Using Beta=1; higher Beta values will cause the left side to return to 0 more sharply)
Differences compared to [[Rectified Linear Unit|ReLU]]:
- Swish is smooth and differentiable everywhere, while ReLU has a sharp corner at x=0.
- Swish *allows small negative values to pass through*, whereas ReLU completely blocks all negative values.
- Swish has a gentle curve at x=0, providing a smoother transition between positive and negative inputs.

Abstract
> The choice of activation functions in deep networks has a significant effect on the training dynamics and task performance. Currently, the most successful and widely-used activation function is the Rectified Linear Unit (ReLU). Although various hand-designed alternatives to ReLU have been proposed, none have managed to replace it due to inconsistent gains. In this work, ==we propose to leverage automatic search techniques to discover new activation functions==. Using a combination of exhaustive and reinforcement learning-based search, ==we discover multiple novel activation functions==. We verify the effectiveness of the searches by conducting an empirical evaluation with the best discovered activation function. Our experiments show that the best discovered activation function,