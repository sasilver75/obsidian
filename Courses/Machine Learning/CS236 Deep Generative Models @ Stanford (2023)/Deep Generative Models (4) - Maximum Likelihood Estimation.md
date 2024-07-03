https://www.youtube.com/watch?v=bt3dqcbMLa0&t=7s

Recap:
![[Pasted image 20240702205405.png]]
There are problems with RNNs, including the information bottlneck and vanishing/exploding gradients.
In reality, we usually use attention-based models that are able to look at all of the vectors from previous timesteps to see what comes next.

![[Pasted image 20240702205500.png]]
There's usually some sort of *attention mechanism* that tells you how relevant some `query` vector is, with respect to some `key` vector.
- Allows you to take in the full context of history, and only pay attention to the important things, and don't pay attention to the irrelevant things... by comparing every pair of tokens.
![[Pasted image 20240702205840.png]]
For autoregressive modeling, you still need to use a masked self-attention mechanism during training to preserve the autoregressive structure.
- At training time, we can parallelize computation! This is a huge benefit over recursive neural networks.

![[Pasted image 20240702210706.png]]
Example of causal masking in convolutions, to make the model autoregressive.

Q: Are transformers more powerful than RNN?
A: Ehh hard to say; RNNs are already turing complete, so...

![[Pasted image 20240702211602.png]]
Can we use these sorts of models that learn p(x) to learn



![[Pasted image 20240702212525.png]]
![[Pasted image 20240702212414.png]]
![[Pasted image 20240702212745.png]]
![[Pasted image 20240702213013.png]]
Maybe we really care about being able to assign reasonable probabilities to any given input? If we can estimate the full joint probability distribution accurately, we can do a lot of things... but it's a pretty tall order/challenging problem.
Maybe we have a specific task in mind (spam or not): Should we actually build a discriminative model that predicts y, given x?
![[Pasted image 20240702213159.png]]
If we really care about the joint probability distributions p(x) being similar to eachother... we'll see that we can get really different models by changing the way we measure distance between distributions.
- Information theoretic (eg [[Maximum Likelihood|MLE]], compression)
- Other ways like "If I generate samples p_data and p_theta, you shouldn't be able to distinguish between the two" (eg [[Generative Adversarial Network|GAN]])

For autoregressive models, a natural way of determining similarity is using likelihood, because we have access to it (probability over next item).
![[Pasted image 20240702213453.png]]
[[Kullback-Leibler Divergence|KL-Divergence]]: Between p and q is the expectation w.r.t. all things that can happen, weighted with respect to probability... and then we look at the log ratio of the probabilities of p and q. It turns out that this is a  non-negative that's zero only if p == q.
- So it's a reasonable notion of similarity.
- If p = p_data and q=p_thetea, then if we drive this small as possible, we're trying to make the model closer to the data.
- NOTE that this quantity is ==asymmetric== (p||q and q||p are not the same!)
	- ==Forward KL and Backward KL==
![[Pasted image 20240702213703.png]]
Specifically, it tells us "If the data is truly coming from p, and we use an optimization/compression scheme optimized for q, how much worse would it be than a compression scheme actually based on the true distribution of the data." This also gives some intuition as to why it's an asymmetric measure.
![[Pasted image 20240702214121.png]]
More Code is a way of encoding letters to symbols; there's a reason why E and A get these short codes, while Q is assigned these very long codes. That's because these vowels are much more common than a rare consonant like Q. If we come up with a scheme based on a *wrong* assumption, if won't be as efficient as one based on the true data frequencies. Again, this gives some intuition as to why KL Divergence is not a symmetric measure.
How much more ineffective your code is... is exactly the KL divergence, which measures how much more inefficient your compression scheme will be. Optimizing for KL divergence optimizes for compression.

![[Pasted image 20240703141423.png]]
If we assign exactly the same probability to every x as our data-generating distribution, we have a perfect model.
Otherwise you'll suffer based on:
- How probable the data distribution thinks an x is
- How far away you are from the P_data estimate

((Note that if you did a KL divergence between P_theta || P_data (Reverse KL), you get much more of a mode-seeking objective -- there's an incentive to concentrate the probability mass, compared to the forward KL that forces you to spread out probability mass over all the things that could happen.))
Q: Is this forward/backward KL question kind of analogous to precision and recall? Where in one case we're considering the denominator as all *true* things, and the other as all positive predictions?
A: Yes! It has a very similar flavor.

Q: What about other measures of evaluating distance between distributions?
A: We'll see that in future lectures! We'll get different classes of generative models by changing the way we compare distributions. Here with KL, we're saying we just care about compression -- but maybe if you're generating pretty images, you don't care about compression, and care about something else! We'll learn more about that later.

![[Pasted image 20240703141551.png]]
If we decompose the log of the ratio as the difference of the logs, we get an expression that looks like this.
- Note that the first term doesn't depend on Theta; it's like a shift/bias that's constant on how we choose the parameters of our model.
- So for the purpose of optimizing our KL divergence, we can effectively ignore the first term, because it doesn't depend on the $P_{\theta}$ that we're optimizing.

If we try to find a theta that minimizes this KL Divergence expression... because the term we care about has a minus sign in from of it, we want to *maximize* the second term.
![[Pasted image 20240703141643.png]]
This says we should pick the distribution that assigns the highest probability to the x's that are sampled from the data distribution. We want our $P_{\theta}$ to *maximize the likelihood of the observed data* (this is equivalent to minimizing the KL divergence).
- This is [[Maximum Likelihood]] estimation! We're trying to create a model that puts high probability on the samples that we drew from the true data distribution.

Note that because we don't really know/have $P_{data}$, we only have an empirical distribution drawn from it, we'll never know exactly how good we're doing, relative to the optimum. In the limit as we grow the size of our $D$, we'll run up to the [[Entropy]] of our data distribution as the limit...

![[Pasted image 20240703142344.png]]
We can approximate the expected log likelihood using the empirical/average log likelihood on the training set.
- We'd really care about the average log likelihood with respect to $P_{data}$, which we don't have access to, but we can approximate it by going through our dataset and checking the log probabilities assigned to every $x \in D$ .
So maximum log likelihood learning tries to maximize the log probability of every datapoint in our dataset. Because the datapoints are independent, this is the same as maximizing the likelihood of the dataset (taking the log of the product expression, we get a sum of logs, which is the same thing as the former formula).


In [[Monte-Carlo]] estimation, when we have an expectation of some random variable, we can approximate it by just taking T samples from the distribution and look at the average value under these samples, and we consider this to be a reasonable estimation.
![[Pasted image 20240703143417.png]]
This $\hat{g}(x)$ is a random variable that depends on our sample.. but in expectation, it gives you back what you wanted ($g(x)$), when the set $T$ is large enough, by the [[Law of Large Numbers]]. It's an ==unbiased== estimation. 
![[Pasted image 20240703143539.png|300]]
Some more formal properties.

![[Pasted image 20240703143827.png]]
![[Pasted image 20240703144012.png]]
Bernoulli random variable (theta = probability heads). We evaluate the likelihood of the data produced as a function of theta. 
- As we vary theta, the probability that our model assigns to the observed data changes; maximum likelihood tells to pick the theta that maximizes the probability of the observed dataset, which here is .6.

We'll basically do the same thing, now, but for autoregressive models!
- Now, theta will be very high dimensional (eg all parameters of the neural net), but our y axis will still be the probability that your model assigns to the dataset...and we try to find a theta that maximizes the probability of observing the dataset we have access to.

![[Pasted image 20240703144220.png]]
The probability of x is just given by chain rule; it's the product of the conditional probabilities (eg of each pixel in the image). The same computation we did before. We go through the conditionals (however we choose these) and multiply them together.
- The second equation just gives this over a dataset.

To do this, we can do something like gradient descent.
![[Pasted image 20240703144606.png]]
- We go through all the datapoints, look at all the variables on each datapoint, and look at the log probability assigned to that variable by our neural model, given all the ones that comes before it, in that datapoint. p_neural is basically a classifier that tries to predict the next value given everything before it.
- We're basically evaluating the average loss of our classifiers among variables, among datapoints.
- Minimizing KL divergence is the same as maximizing log likelihood which is the same as making these classifiers perform as well as they can. ==So let's try to make these classifier as good as possible!==
	- We initialize random parameters, compute gradients, and do gradient ascent/descent. It's non-convex optimization, but we have tricks to make it work pretty well in practice.

![[Pasted image 20240703150121.png]]
[[Stochastic Gradient Descent]] (or Mini-Batch Gradient Descent) lets us estimate the gradient by just looking at a small samples from $D$. 

Now we can do Monte Carlo! We can approximate this expectation by taking a bunch of samples, and evaluate the gradient on those samples, and update our model accordingly.
![[Pasted image 20240703150208.png]]
Instead of evaluating the full gradient, we evaluate the gradient on a subset of datapoints to keep it scalable and get faster convergence.

![[Pasted image 20240703150254.png]]
But we don't just care about blindly minimizing the losses that we've been describing!
- We want to *generalize well*, not just do well on the training set!
We ned to some how *restrict* or *regularize* the hypothesis space so that the model doesn't just memorize the training set.

The problem is the usual [[Bias-Variance Tradeoff]], where if we limit the model too much (eg using logistic regression), we don't have enough capacity to fit the training data well. This is called ==bias== because we limit how well we can approximate the target distribution, even if we do the best with what we're given.
If we use a model that's too flexible, we encounter errors due to ==variance==, where even small changes to the training dataset can have huge changes to our parameters that we output. Our low error on $D$ doesn't generalize to out-of-sample data.

