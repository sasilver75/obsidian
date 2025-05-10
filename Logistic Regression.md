A ==Discriminative== model that learns p(y|x)
- Note that 


Assumes independence between all the records  that we're training on.

$P(y_1, y_2, y_3, .... | x_1, x_2, x_3) = \prod_{i=1}^NP(y_i|x_i)$

$\hat{w} = \underset{w}{min} L(w)$

$L(w) = -\sum_{i=1}^N log(p(y_i|x_i))$

Minimizing the negative log is the same as maximizing the function, as long as the function.