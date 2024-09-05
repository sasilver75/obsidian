A statistical measure that quantifies the degree to which two variables vary together from their respective means. Sometimes denoted as $\sigma_{XY}$.

$Cov(X,Y) = E[(X-\mu_X)(Y-\mu_Y)]$
or, written another way:
$Cov(X,Y) = E[(X-E[X])(Y-E[Y])]$

By using the ==linearity property of expectations==, this can be further simplified to the expected value of their product minus the product of their expected values.
![[Pasted image 20240709021349.png]]

Aside: The ==linearity of expectation== says that the expectation of a sum of random variables is equal to the sum of their individual expectations.
- In the above equation, we apply it to the outer expectation on the second line. Note that $E[E[X]] = E[X]$, but notice how it "expectifies" the XY, X, and Y? This then produces line 3, which is straightforwardly simplified.