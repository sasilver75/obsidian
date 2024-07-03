https://www.youtube.com/watch?v=2ojJUSMf-_g&list=PLwRJQ4m4UJjPIvv4kgBkvu_uygrV3ut_U&index=6

Not a very useful lecture, PhD student self half is schizophrenic

![[Pasted image 20240702144053.png]]
Autoregressive are *one type* of generative model (we'll cover ~5 in this course).
It has many advantages that make it quite popular as a formulation these days, but it's still not so clear which models are going to be the most important ones.
- It's doing quite well for text, and handles images and speech too...
As we learn more, we'll be able to contrast them better with eachother.

---

Motivation
![[Pasted image 20240702144518.png]]
Problems we'd like to solve:
- We want to train a model that we can ask to ==generate/sample data== similar to the data in some other dataset that we have (images, video, text, audio).
- We'd like to be able to ==*compress* data== -- if we build a good probabilistic model of our distribution, we know that we can use it to achieve optimal compression of our dataset.
- ==Detect anomalies==: If we just build a classifier, it will build us an answer of "dog" or "cat," but if the data is out-of-distribution, it will still make its best guess. Whereas a generative model can output the probability of that input, under the model's data distribution, and so when that probability is very low, we know that it's out-of-distribution, so we think our classifier will be less likely to give a good answer.

We'll look at likelihood-based models (other models can thought of in other ways):
- We want to estimate the probability distribution of the data p_data from samples drawn from that real p_data distribution.

Once we have our $\hat{p}_{data}$, we can:
- Compute the probability p(x) for some arbitrary x (eg for anomaly detection)
- Sampling new x datapoints from our model

Today, we'll only talk about discrete data.
- In continuous data, it's so easy to go out of distribution, because the space is so large.
	- So it's much harder to generate data in these situations..

![[Pasted image 20240702145233.png]]
Even something small like these images actually lie in a 50,000 dimensional space!

![[Pasted image 20240702145353.png]]
Our agenda: We'll do 1-dimensional, high-dimensional, and then a deeper dive into the most popular causal masked neural models...
- And then give a little treatment on some of the other less-popular techniques (for now)

---

# One Dimensional Distributions

![[Pasted image 20240702145445.png]]
Let's say we're estimating a one-dimensional distribution, where the samples are (eg) between 0 and 100.
- We see there are sort of two modes.
We've drawn a ==histogram==, which just counts how many datapoints there are across different buckets, normalized by the number of draws.

![[Pasted image 20240702150732.png]]
- To ==inference== (finding the probability for a random i), we just lookup in the array of p.
- To ==sample==, we build a cumulative distribution function from the model probabilities, draw some random number (representing a cumulative probability) from 0...1, and then find the smallest i that satisfies this number.

So now we know how to model a histogram, evaluate probabilities, and sample from it... but the issue is that this often has very poor generalization.
- Every bin is independent from every other bin
	- If you learn how often 40 appears, then that tells you nothing about other number bins.

![[Pasted image 20240702151049.png]]

![[Pasted image 20240702151140.png]]
We fit a distribution with a parametrization, rather than a histogram! The histogram fits our data the most precisely -- and it doesn't generalize well as a result. But maybe the yellow curve does!
Let's talk more formally...

![[Pasted image 20240702151305.png]]
The goal is to estimate our p_data from samples!
- We introduce a model parametrized by theta, where we learn theta so that p_theta approximates our sample from p_data.
	- We pose this learning as an optimization function with a loss function defining distance between distributions, with only access to our samples from p_data (the empirical distribution, not the true p_data).

![[Pasted image 20240702151807.png]]
- In [[Maximum Likelihood]], the loss function put forward essentially says that we want to minimize the negative log likelihood of each datapoint.
	- Where there's a lot of data, we want to allocate a lot of probability; where there's not a lot of data, we have to allocate low probability.
	- Basically maximizing the log probability of our datapoints.
	- We want to find a parametrization under theta that makes our data most likely.

It's true that minimizing the negative log loss above is the same as minimizing the KL divergence between our empirical data distribution and our model's distribution.

So now that we have a loss function, we need to optimize it by minimizing our expected negative log probability over our datapoints, by varying theta.
![[Pasted image 20240702152240.png]]
SGD looks at a subset of the data, evaluates it, takes a step, and repeats. Often leads to faster convergence.
- Whatever fits into your GPU memory basically defines your batch size. After every batch we determine loss/gradient and update our parameters theta.

---

# High Dimensional Distributions

![[Pasted image 20240702153353.png]]
The challenge with high dimensional data (even MNIST), means that we have an explosion in the number of possible images -- we can't easily compute probabilities for every specific image possibility.
- If we assume conditional independence of the pixels, then we don't have any generalization ability! Two identical images with one pixel difference are seemingly unrelated.

In this class we're going to introduce ==autoregressive models==, which essentially model high-dimensional distributions by modeling a bunch of low-dimensional (one-dimensional, in fact) distributions!
![[Pasted image 20240702153607.png]]
- We decompose our main probability function... into a series of probabilities (one pixel conditioned on the previous pixels).
- But these one-dimensional distributions are still *conditioned* on something very high-dimensional, so we still have an exponential blow-up in the things we need to count/represent.

==Different solutions people have proposed. We'll go through 4 families of solutions!==

## Bayesian Networks/Bayes Nets
- [[Bayesian Network]]s are essentially an autoregressive model.
- Instead of conditioning every conditional probability on ALL of the preceding variables, we condition only on the *parents* of variables.
- So instead of thinking of it as a sequence, we can think of it as a DAG; We call the parents in the graph the things we condition on, and everything else things we ignore.
- It *sparsifies* the relationships, making it much easier to represent, and lower-dimensional.
![[Pasted image 20240702154018.png]]
(We're only showing 4/5 of the tables for some reason. These tables are small so we could even use a histogram/table approach to these, or if we wanted to parametrize these distributions we could too).
- This sparsification introduces causal assumptions that, if they hold true, you're winning by doing this, by introducing real-world data structure into your model that it doesn't have to learn!
- If we miss out on some causal relationships, we'll handicap our model.

> It turns out that extra connections that aren't relevant don't hurt you -- it's just that you introduce additional tables that take up space you don't need. We'll even see at some point bayesian networks where every node is connected; zero sparsity! We'll have to do some work to not deal with tables.

This doesn't really work for images though
- Imagine saying "This pixel over here only depends on these other ones over there, and not these ones over here, etc." We don't know the structure, or there just *is no such structure* -- the best structure is instead of keep all the images, have a densely connected graph, and then... we don't really have any benefits of using a bayes net anymore.
We still wanted to introduce Bayesian Networks because maybe someday they'll make a comeback. 

## MADE
- Even if you're working on autoregressive models all day, you might not have heard of MADE, but it inspired/is the foundation for the idea that drives everything today -- parametrizing conditionals with neural nets.
![[Pasted image 20240702154920.png]]
- With a NN, we get a new representation out... by putting in some noise on x1/x2/x3, and have the neural net denoise it (huh, like a diffusion model, right?)
- This autoencoder doesn't output probabilities, though -- when you put in data, it turns it into a cleaned-up datapoint, but it doesn't output probabilities. We can't really correctly sample from it either, the way this is set up....
- So MADE turned this into a chain rule thing... when we generaet a cleaned up version of x2, it can't depend on anything. 
	- In this case, they generated x2, then x3, then x1 (see conditional)
	- Instead of using the full NN which would violate some assumptions, we need to cut out some images such that, when we get to the last layers, our distribution for x3|x2 doesn't depend on x1.
![[Pasted image 20240702155351.png|300]]
That's the only path leading to x3|x3, and it comes from x2 ((?))

So MADE feels pretty messy presented this way, but that's how it's presented in the paper.
![[Pasted image 20240702155539.png]]
Today, this is how people would draw it. We have the inputs at the bottom, predictions at the top, and we have a causal mask structure running left to right (see the top is shifted by one to make this easier to draw).
"Its a bayes' net where you can't have any more edges than what we have here, if you want to think about it that way."

This is sort of effectively what's being used today (if you squint), but it's missing a lot of the tricks that matter.

MADE is expressive, but there's not enough parameter sharing for efficient learning
## Causal Masked Neural Models
- Parametrize the conditionals with a NN like MADE, but adds parameter sharing across conditionals (think of it as a sliding window), and adds coordinate coding to still be able to individualize conditions (so while we have the same parameters for each location, we put that information back in by inputting the location coordinates, so that the parameter have to account for that).
![[Pasted image 20240702160714.png]]
The coordinate encodings can be one-hot encodings, relative encodings, etc.
- It's like a MADE model, but we feed in coordinate locations also, and there's parameter sharing.

![[Pasted image 20240702160854.png|300]]
Typically, at the very least, what's happening on all of these edges is the same; they use the same parameters.
- So we can have a lot of high-dimensional data, but a relatively low number of parameters, still.

![[Pasted image 20240702161028.png]]
[[WaveNet]] used this to generate speech better than anything else before!

![[Pasted image 20240702161048.png|400]]
They also used dilation in their convolutions, so they could learn to take information from further back too... 

Another trick used at the time (in WaveNet):
- "See this Tanh and Sigmoid? The MLP would both have a tanh output and in parallel a sigmoid output, and the sigmoid would gate the tanh; in today's world, attention models do the gating on the sigmoid, but it's the same kind of thing... KQ doing the gating on the sigmoid; similar idea"

So again:
- Causal Masked Neural Models
	- They're basically MADE but with more parameter sharing and a way to inject coordinate coding to make up for the parameter sharing.
	- Expressive and efficient, but the problem about sampling remains -- it's one at a time, and bit slow. Training is fast, because we can do a lot of samples in parallel.
	- Downside is that they might be limited on their context length -- but finite contexts can still be pretty large, and you can do retrieval too to help.


## Recurrent Neural Networks
- Parameter sharing with an "infinite lookback" in principle.
- ![[Pasted image 20240702162036.png]]
- We have an input layer, hidden layer, and output layer; in the hidden layer, we get this horizontal propagation, which effectively allows us to remember everything from the past that is relevant to the future.
	- In practice, does it really do this? Not really; in practice it doesn't work as well as you could hope for... but maybe someone will get it to work well!

Unfortunately, RNNs aren't as amenable to parallelization as Neural networks can be!
- Vanishing, Exploding gradients too
- It's hard to have the signal remain over a long history
- Expressive, but maybe not sufficiently so (and probably less so than the masked ones); might need different inductive biases.

![[Pasted image 20240702162304.png]]



![[Pasted image 20240702162757.png]]
![[Pasted image 20240702162753.png]]Clever convolution that preserves causal masking; nothing can propagate in a way that's disallowed; we learn a causal masked model that also captures some convolutional priors.
We can then sample from this...
![[Pasted image 20240702162928.png]]
And it kind of works
But there's a problem, which is the blind spot in the receptive field :( 
- ![[Pasted image 20240702163052.png|200]]
- Better models these days use attention.


Let's talk about another form of masking that's easier to impleemnt!

![[Pasted image 20240702163224.png]]
One of the main problems with convnets (eg wavenet, 1d convs) is that the receptive field grows linearly with the number of layers you have.
- That's not  great if you want to capture long-horizon dependencies across images, time in videos, or paragraphs in text.

Masked Attention could technically have unlimited receptive field (if you have unlimited compute)
![[Pasted image 20240702163416.png]]
This is how Scaled DP Attention works:
- Think of it as a soft lookup table
	- Query
	- Key
	- Values
- For each token in our sequence we compare its query projection to the key projection of other tokens via a dot product... and then we compute our output by reweighting value vectors based on these attention scores.
![[Pasted image 20240702164151.png]]


![[Pasted image 20240702164349.png]]Transformer consists of repeating blocks

"The MLP has no cross-token interactions"... 

![[Pasted image 20240702164538.png]]
All we really have to do is add a causal attention masking to the attention layers in the Transformer to make it into an autoregressive one. We can do inference by sampling one token at a time.

![[Pasted image 20240702164831.png]]
D = hidden dimension, L = sequence length
For MHA, quadratic in both D and L; the quadratic in L is what bites us here, because we'll get to very long sequence lengths.
in MLP, quadratic in D


![[Pasted image 20240702164923.png]]
![[Pasted image 20240702165044.png]]
![[Pasted image 20240702165049.png]]
Start with all the characters in our vocabulary, and then we perform some predefined number of merges by merging tokens that most frequently are adjacent. When we "merge" tokens, we really just add a *new* token to our codebook that's the merge of the tokens ("a" + "b" -> "ab", but we retain a and b too). So you'll often get words like "the" as whole-word tokens, but "xylophone" often will get "xylo"+"phoned";
- This reduces the # of unks as long as the tokens that you're doing inference on were actually present in the training data. So it will generalize well to new english words (likely), but you'll still have a lot of unks if you're doing chinese.

{Talk about GPT-2 being able to do zero-shot transfer}
{Talk about GPT-3 being able to do few-shot learning}
We'll have another lecture on language models later on in the semester.

Let's move on to tokenization for images:

![[Pasted image 20240702165604.png]]

![[Pasted image 20240702165839.png]]

![[Pasted image 20240702165909.png]]
Sparse Transformers can still do attention, but with a specific structure.
- It still follows the autoregressive property, but some of the tokens in the bottom left are masked out also.
Ideally we only want to be doing dot products where the mask is 1 (where things aren't being masked out); so on the right we wouldn't want to *train* our model using this approach, because a lot of the result of our dot products don't really matter.
- Under the hood, there's code that takes advantage of this sparsity and doesn't do computation that it doesn't have to do.

But this drives scaling from L^2 (sequence length) to something like L(log(l)) or L(root(l)) or something. This allows us to at least scale to longer sequences (10-12k).
At the time, this was SoTA...

![[Pasted image 20240702170807.png]]
The important part is that... for autoregressive transformers, we want the data to be discrete, so we want the encoding z to also be discerete
- There are many methods for this!
![[Pasted image 20240702170837.png]]
==Discrete Autoencoders==
GS: Used for Dall-E
VQ-VAE was popular option a few years back
FSQ gained more traction in the last year

These are all just ways to create a discrete z.
![[Pasted image 20240702170947.png]]

![[Pasted image 20240702171138.png]]
GANs using the VQ technique from above to discretely encode data.


![[Pasted image 20240702171357.png]]
We have ICL for langauge (eg a madeup task and some examples)... is there somethign analagous for images?
- can we do visual prompting and have it learn a visual task?
- If we had no segmentation data at all... and we curate and prompt with 3 example segmentations and a new image, that would be cool!
This paper curated a lot of data on different types of visual data...

![[Pasted image 20240702171541.png]]
Above: The task is something like keypoints on an object as it rotates
We prompt with a new novel object and ask it to create the rest of the rotation/keypoints?
![[Pasted image 20240702171618.png]]
Here's another example of pattersn -- a person getting more sad, or more happy, or increasing in number, or getting darker.


![[Pasted image 20240702171734.png]]
They trained a model on a bunch of different conditioning (on text, video, audio, etc.)... and the nice thing is that transformers are generic enough that many of the parameters are shared across modalities.

---

Now let's talk about Caching.
- Say we have two tokens already generated, x1 and x2, and we want to generate x3.
	- Compute keys and values for prior token
	- Compute q for current one
	- We do attention
![[Pasted image 20240702172917.png]]
Annoying thing: We have to recompute all of the KVs for all of the prior timestamps every time! Meaning we have to run all of x1...xn-1 all through the entire transformer network -- this makes it very expensive.
- Every sampling pass is a full forward pass of the transformer

Instead, we cache the K and Vs for all attention layers of each sampling step!
![[Pasted image 20240702173039.png]]
Before, we would just throw our K and V.... but in this case, we explicitly store these tensors in our memory and keep them persistent as we sample the next tokens.
![[Pasted image 20240702173108.png]]
Then we compute k3,v3, and get q3, and compute the attention...
We apeend (k3,v3) to our cache

We then do x4...
So we have our cache from k1 to k3, 
![[Pasted image 20240702173138.png]]
We compute k4,v4 and q, do attention, cache k4,v4, ...
![[Pasted image 20240702173158.png]]
We're essentially trading off GPU memory for speed; If we have 10 attention layer, we have to stash k,v's for every attention layer -- but it's ver worth it, because it brings our sampling down from O(L^2) to O(L).

----

# Other things to be aware of

## Decoder only vs Encoder-Decoder models
- The original attention is all you need paper used the [[Encoder-Decoder Architecture]], which is similar... but it has an additional encoder stack, and the decoder stack performs an additional cross-attention to the bi-directional encoder.
- ![[Pasted image 20240702173433.png|250]]
- In general, this is a pretty good architecture for cases when you have a clear conditional distribution to our model ( a clear input and output )
	- MT
	- Text to image generation
	- Image captioning
	- Video captioning
	- Summarization
(Clearly structured in some way)
It just doesn't work as well for Chat-style models where you don't know "What is input and what is output (?)" -- output will be input eventually as you sample the model, making it harder to structure a model like this (?).
![[Pasted image 20240702173556.png|400]]
- It's nice to condition things on the representations produced by the encoder stack of these models (though you could also just use an encoder-only model? 😄)

## New Incarnations of recurrent models
- ![[Pasted image 20240702174201.png]]
- Can we get around the sequential nature of recurrent models, to improve their performance during training?
![[Pasted image 20240702175135.png]]
- We treat x as the hidden state of our RNN, and u and the input, and y is the output.
- So it's basically multiplying a few matrices.... and outputting it (?)

(Terrible explanation)
