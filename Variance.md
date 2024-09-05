The variance of a random variable $X$ is the expected value of the squared deviation from the mean of $X$, $\mu = E[X]$. Witten either as $Var(X)$ or as $\sigma^2$.

$Var(X) = \mathbb{E}[(X - \mu)^2]$

This can also be expanded to
$Var(X) = \mathbb{E}[(X - \mu)^2]$
$=\mathbb{E}[X^2 - 2XE[X] + E[X]^2]$  ... (expanded using the FOIL method)
$=\mathbb{E}[X^2] - 2E[X]E[X] + E[X]^2$ ... (applies linearity of expectation)
$=\mathbb{E}[X^2] - 2E[X]^2 + E[X]^2$ ... (The expected value of an expected value is just an expected value, which turns $2XE[X]$ into $2E[X]E[X]$, which we then turn into a square)
$=\mathbb{E}[X^2] - E[X]^2$ ... (subtraction to collapse the second and third terms)
In other words, the variance of X is equal to the mean/expectation of the square of X minus the square of the mean/expectation of X.

Aside: ==linearity of expectation== states that the expected value f the sum of random variables is equal to the sum of their individual expected values.

The variance can also be thought of as the [[Covariance]] with itself:
$Var(X) = Cov(X,X)$

----

For a discrete probability distribution:
![[Pasted image 20240708172505.png|200]]
Where $\mu$ is the expected value
![[Pasted image 20240708172525.png|150]]

The variance of a collection of $n$ equally likely values can equivalently be written as
![[Pasted image 20240708172559.png|200]]
