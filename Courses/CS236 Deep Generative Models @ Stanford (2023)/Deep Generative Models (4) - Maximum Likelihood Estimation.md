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

![[Pasted image 20240702214121.png]]
More Code is a way of encoding letters to symbols; there's a reason why E and A get these short codes, while Q is assigned these very long codes. That's because these vowels are much more common than a rare consonant like Q.

![[Pasted image 20240702215453.png]]
If we assign exactly the same probability to every x as our data-generating distribution, we have a perfect model.
Otherwise you'll suffer based on:
- How probable the data distribution thinks an x is
- How far away you are from the P_data estimate


"Spend *days* working on your prompt! Make them clear, follow good prompting guides, etc."


![[Pasted image 20240702225029.png]]

