Monte Carlo methods are a broad class of algorithms ==relying on *repeated random sampling* to obtain numerical results==.

The underlying concept is to ==use randomness to solve problems that might be deterministic in principle==.

Mainly used in:
- Optimization
- Numerical Integration
- Generating draws from a probability distribution

Are often implemented using computer simulations, and can be used to provide approximate solutions to problems that are otherwise intractable or too complex to analyze mathematically.

Generally a tradeoff is available between computational cost and accuracy. Be wary of the curse of dimensionality inflating computational costs.

Monte Carlo methods vary, but tend to follow a particular pattern:
1. Define a domain of possible inputs
2. Generate inputs randomly from a probability distribution over the domain
3. Perform a deterministic computation of the outputs
4. Aggregate the results


![[Pasted image 20240603185247.png|200]]
Above: Approximating $\pi$ via "throwing darts" at the image, and counting the number that appear within the radius of the circle (red). The more darts thrown, the more our estimate approaches reality (thanks to the [[Law of Large Numbers]]).
