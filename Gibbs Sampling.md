A [[Markov Chain Monte Carlo|MCMC]] method that is useful only really when sampling from multivariate distributions, where we have two or more dimensions for the distribution we're trying to sample from.

Let's say our goal is to sample from a two-dimensional gaussian distribution:
![[Pasted image 20240712110714.png|200]]
There are known ways to sample from this that aren't Gibbs sampling, but let's use Gibbs sampling to see how it works in practice.

You should use Gibbs sampling when *both*:
1. We're sampling from a multivariate distribution
2. Sampling from the joint distribution p(x, y) is difficult, but sampling from the conditional distributions p(x|y) or p(y|x) is easy (where in a high-dimensional distribution, it's just going to be "the density of one variable, given the others")

So what are the conditional distributions for this particular example?
![[Pasted image 20240712111352.png|300]]
(Derivation not shown)

Gibbs sample procedure:
1. Start by at some $x^{0},y^{0}$ , preferably close to the center of the distribution, but could be anywhere (just helps convergence speed)
2. We change x, keeping the y variable fixed. We sample $x^1 \sim p(x^1|y^0)$
3. We change y, keeping the x variable fixed. We sample $y^1 \sim p(y^1|x^1)$
4. (Go back to step 2)

![[Pasted image 20240712113218.png|400]]
If you take enough of these samples, it will be as if you'll be sampling from the multivariate distribution. You can extend this to any multivariate problem (sample first, given others, sample second, given others, sample third, given others, etc.)


![[Pasted image 20240712113642.png]]
One problem is spots where there is going to be parts of your space with "spikes" in probability (consider the 2d distribution with a lot of probability density on the dot, and everywhere else being low probability distribution). If we're currently in a low probability region, we're probably going to be in a low region again (by only being able to move in an x direction or a y direction at a single time). Conversely if we're in the high-density bubble, if we sample in the x or y direction, we're probably going to stay in the high density bubble. It will take unfeasible long to converge to the actual distribution, because you'll stay in lows and stay in highs.