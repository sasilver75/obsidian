
Paper: https://arxiv.org/abs/2310.16834
Oxen: https://oxenai.notion.site/Text-Diffusion-with-SEDD-6b14123d70f245b2b23383c1ed5a78ac

# Text Diffusion Paper

SEDDs are the first published example of a diffusion-based model architecture showing **competitive performance with best-in-class autoregressive architectures of the same model size** class, achieving a better perplexity score than GPT-2.

But why are we so excited about matching performance to a 2019 model?

Many researchers believe that there are inherent limitations to autoregressive modeling:

- Since generating token i is dependent on generated tokens 0 .. i-1, generation **must happen sequentially, from left-to-right.** This prevents useful, novel generation styles like **infilling**.
- Sequential sampling is **slower**, as its dependence prevents it from being parallelized at inference time. Generating token n requires having first generated token n-1, etc.
- Sequentially sampled outputs tend to **[degenerate as the sequence length increases](https://arxiv.org/abs/1904.09751)**. This can be remedied through techniques like [nucleus sampling](https://arxiv.org/abs/1904.09751), but these introduce biases into the generation.
	- ((We use [[Top-P Sampling]] to deal with [[Exposure Bias]] fucking up our generations]))


### Modeling probability distributions for generative AI

When building generative AI, we are looking to create a model which will approximate a given distribution of data as closely as possible.

That distribution could be:

- Coherent english sentences
- Bug-free, best-practice python code
- Photorealistic natural images
- Cute, pixar-like cartoons

![[Pasted image 20240412101339.png]]

We don't have this magic black box though! What we do have is the Probability Mass Function, which maps input to output probabilities under the distribution.


![[Pasted image 20240412101452.png]]
Unfortuantely, the Probability Mass Function for the distribution of text is much harder to find!

So what do we do when we have a problem where we want to estimate some extremely complex function?
- We grab our NNs, throw a bunch of data at it, and estimate the function!
- We train a NN to approximate this underlying distribution!

# Solution #1: Train a Network to approximate the PMF

![[Pasted image 20240412101729.png]]
Downside: There are two primary constraints for PMFs
1. Any value in a PMF -- any output for any given x must be > 0
	- No problem, we just exponentiate them with $e^{f(x)}$ -- that makes everything positive
2. All values in the PMF must sum to 1
	- No problem... we just normalize it :O

((We end up with something like the softmax function!))


# Problem #2: The normalizing constant, z_theta, is intractable!
- GPT-2 for example has a vocabulary of 50,257 tokens, and the input sequence is of length 1024!
- Since tokens can be repeated, the number of combinations you can have is 50,257^1024, which is more than the # of atoms in the universe!

What's a solution to this?

# Solution #2: Autoregressive modeling

It lets us get out of this intractability of possible sequences -- it lets us escape this intractability!
We reframe the probability calculation such that each token to be generated is conditioned on all those that came before it!

A
A generated
A generated sentence
A genereated sentence here


# Solution 3: Model score, not probability mass!
- We don't really need to know that the probability of token x is .005 ; we just need to know that it's more probable than token y!
	- We're not interested in the actual value, we're just interested in stepping in the right direction, whichever that might be!
If we teach our modelto learn ==the direction in which to move== to increase the fidelity of our outputs...

![[Pasted image 20240412102453.png]]
This is automatically a lot more tractable, because that big normalizing Z number gets zeroed out!

So for an x in the training data,
- We don't even need to understand p(x) or (py)
- We just need to understand the relationship between these two!

![[Pasted image 20240412102656.png]]
We have an input observation  x, and a bunch of changes to it to make a new input y
- How much less likely is it that y came from our training data than this x?
- How different is this new y from the population we're trying to generate, relative to x (which is in that population?)


There are still two problems with this equation, though:
1. The set of possible y is still intractably large
	- Eek, the set of possible Y is still intractably large! Y for each X is every input that isn't X!
		- 50257^1024 -1
	- Authors: Let's restrict this to all possible Y which are "close to" X, defined by [[Hamming Distance]], the number of tokens that are distance == 1 away
		- (change one token in the sequence)
2. This loss function assumes we know the PMF, p(x), which is the whole point!
	- The function that gives us p(y) and p(x) is still a magical black box :(
	- This is why we end up turning to a Diffusion process!

# Entering the Diffusion Zone


Adding noise....
- Our way out of this conundrum is, rather than estimate p(y)/p(x) directly, estimate p_t(y) and p_x(t) ==perturbations== of the original data distribution, for various timesteps t
	- Think: different amounts of noise being added to the data

![[Pasted image 20240412103153.png]]

We can't check our work with the unknown quantity p(x), but we can check our work on the amount of noise we added! We know what was added, so we have the answer key.

If we can estimate the effects of the noise, we can REVERSE it, and our model can take us from noise to a coherent generation at inference time.


![[Pasted image 20240412103319.png]]
We can't do the same thing with text, quite -- if we jitter every single word in the sentence (Changing it), there no way to recover the original one!

We still need some signal for our network to be able to accurate learn what noise was added!
- We iteratively add this noise!

![[Pasted image 20240412103417.png]]

Once we have this calculated quantities, we can go in the opposite direction and ==denoise the data!==
We an eventually denoise random data, getting a coherent sentence!

To generate a new sample:
1. Start with a random (unconditional) new sample of your desired length (1024)
2. Us our learned concrete ratios and diffusion scheduler to step "backwards" i ntime, sampling what the model believes is a less on less noise version of the previous input
3. Arrive at t=0, with a denoise input zero will, under a trained network, closely resemble the training data distribution!


To genreate a sample with conditioning (==prompting==)
- Rather than starting with a pure noise sample, insert desired prompt tokens and fill the rest with diffused tokens.
- Apply the iterative schedule only to the diffused tokens:
![[Pasted image 20240412103749.png]]

Q: This doesn't let you change the length of the sequence, does it? 
A: Nope. You could still have the model predict early stops, though.

![[Pasted image 20240412104224.png]]

![[Pasted image 20240412104229.png]]

Competitive with autoregressive approaches, much better than other diffusion processes


# So what -- why is this impresive?
1. Unlike autoregressive models (GPT-2 et al) text diffusion models are not limited to Left to Right prompting/completing
	- In particular, the approach we discussed allows for INFILLING, a commonly used technique in imagery but infrequently applied to text.

2. Text diffusion models have better natural long-term coherence, as they're less prone to degeneration at longer sequence lengths!
	- Autoregressive models use additional "annealing" techniques to prevent sequence degradation over time.
	- A common one is [[Top-P Sampling|Nucleus Sampling]], which improves long-term coherence by excluding very low probability tokens from sampling

3. Diffusion models enable a direct tradeoff of compute for sample quality
	- As mentioned earlier, the diffusion process involves taking small steps backwards in the diffusion timescale, iteratively removing noise to generate higher and higher fidelity samples. This allows inference-time flexibility to determine how many denoising steps you want to take.






