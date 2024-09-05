
Information theory concept that ==measures the amount of information that one random variable contains about another.==
In the context of [[Entropy]] (which quantifies the amount of uncertainty/randomness in a random variable), Mutual Information is used to quantify the *reduction in uncertainty/entropy* of one random variable, due to knowledge of another.

It essentially measures how much knowing *one variable* reduces uncertainty in *another*, and indicates some degree to dependence between the two variables.

Given two random variables $X$ and $Y$:
The *mutual information* $I(X;Y)$ is defined, in terms of entropy $H(...)$ , as:
$I(X;Y)=H(X)−H(X∣Y)=H(Y)−H(Y∣X)=H(X)+H(Y)−H(X,Y)$ 
This intuitively makes sense, right?
The mutual information between X and Y is the same as the amount by which the entropy of X is "reduced" when we learn about Y, or the inverse. This is a ==symmetric measure==.