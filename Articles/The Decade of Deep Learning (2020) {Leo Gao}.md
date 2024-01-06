Author: Leo Gao (Cofounder EleutherAI, Research Engineer @ OpenAI)
https://bmk.sh/2019/12/31/The-Decade-of-Deep-Learning/

Let's look back at some of the most important papers of the monumental decade of 2010-2020!

## 2010: [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
- Explored some problems with deep networks, especially surrounding random initialization of weights!
- The lasting contribution is with initialization of weights -- when initialized with normally-distributed weights, it's easy for values in the network to explode or vanish, preventing training.
	- Assuming that the values from the previous layer are i.i.d. Gaussians, adding them adds their variances! Thus, thus the variance should be scaled down proportionally to the number of inputs in order to keep the output zero mean, within unit (1) variance.
	- The same logic holds in reverse (i.e. with the number of outputs) for the gradients.
	- This paper introduced something called Xavier initialization, where the variance of the Gaussian from which weights are initially set is determined by the number of neurons in the previous and next layers, respectively.

![ReLU and Softplus (<a href='http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf'>Source</a>)](https://bmk.sh/images/rectifier.png)
## 2011: [Deep Sparse Rectifier Neural Networks](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)
- Most NNs used sigmoids (eg logistic, tanh) for intermediate activations. These have the advantage of being differentiable everywhere and having a bounded output. 
- Because the derivative of sigmoid functions decays quickly away from zero, the gradient often diminished rapidly as more layers were added. 
	- This is known as the [[Vanishing Gradients]] problem, and is one of the reasons that it was difficult to scale networks depthwise.
- This paper found that using [[ReLU Activation Function]]s helped to solve the vanishing gradient problem, and paved the way for deeper networks!
- Still, ReLUs have flaws; they're non-differentiable at zero, and can grow unbounded, and neurons could "die" and become inactive due to the saturated half of the activation. Since 2011, many improved activations have been proposed to solve these problems, but vanilla ReLUs still remain competitive!


![AlexNet architecture (<a href='https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks'>Source</a>, Untruncated)](https://bmk.sh/images/alexnet_diagram.png)
## [2012: Imagenet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) (The "AlexNet" paper)
- [[AlexNet]] is an 8-layer Convolutional Neural Network using the ReLU activation function and 60 million parameters!
- It was generally recognized as the paper that sparked the field of Deep Learning! It was one of the first networks to leverage the processing power of GPUs to train deeper convolutional networks than before!
	- This paper lowered the state of the art error rate on the ImageNet dataset from 26.2% to 15.3% -- a blowout! This attracted a lot of attention to the field of deep learning.


## [2013: Distributed Representations of Words and Phrases and their Compositionally](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- This paper introduced [[Word2Vec]], which became the dominant way to *encode text* for use in Deep Learning NLP models.
- It's based on the idea that words which appear in similar contexts likely have similar meanings, and thus can be used to embed words into vectors, for later downstream use.
- Word2Vec in particular trains a network to predict the context *around a word* given the word itself, and then extracts the latent vector from the network.

Honorable Mention: The [[GloVe]] (Global Vectors for Word Representation) paper is an improvement based on the same core ideas of Word2Vec, but realized slightly differently.

![DeepMind Atari DQN (<a href='https://arxiv.org/abs/1312.5602'>Source</a>)](https://bmk.sh/images/dqn_atari2.png)
## [2013: Playing Atari with Deep Reinforcement Learning
](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- The results of Deepmind's Atari DQN kickstarted the field of Deep Reinforcement Learning. 
- Reinforcement Learning was previously used mostly on low-dimensional environments like gridworlds, and was harder to apply to more complex environments. Atari was the first successful application of RL to a higher-dimensional environment.
	- This brought Reinforcement Learning from an obscurity to an important subfield of AI.
- This paper uses Deep [[Q-Learning]] in particular, a form of value-based RL. 
	- Value-based means that the goal is to learn how much reward the agent can expect to obtain at each state (or in each state-action pair) by following the policy *implicitly* dictated by the Q-value function.
- This policy used in this paper is the *epsilon-greedy* policy!
	- This takes the greedy/highest-scored action with probability *1-e*, and takes a completely random action with probability *e*. This allows for exploration of the state space.
- The object for training the Q-value function is derived from the [[Bellman Equation]], which decomposes the Q-value function into the current reward plus the maximum (discounted) Q-value of the *next* state: 
![[Pasted image 20231221213922.png]]
- The technique used for updating the value function based on the current reward plus future value functions is known as [[Temporal Difference Learning]] (TD-Learning).


![GAN images (<a href='https://papers.nips.cc/paper/5423-generative-adversarial-nets'>Source</a>)](https://bmk.sh/images/gan_res.png)
## [2014: Generative Adversarial Networks](https://papers.nips.cc/paper/5423-generative-adversarial-nets
- [[Generative Adversarial Network]]s rely on a *minimax game* between a Generator network and a Discriminator network.
	- As as result, GANs are able to model complex, high-dimensional distributions (often, images). 
- The objective of the Generator is to minimize the log-probability *log(1-D(G(z)))* of the Discriminator being correct on fake samples, while the Discriminator's goal is to minimize its classification error (*logD(x) + log(1 - D(G(z))*) between real images and fake images produced by the Generator.
	- In other words, the Generator learns to create convincing images, and the Discriminator learns to tell which images are fake and which are real.
	- (In practice, the Generator is often trained to instead maximize the log-probability D(G(z)) of the Discriminator being incorrect... this minor change reduces gradient saturating and improves training stability.)

![A visualization of the attention (<a href='https://arxiv.org/abs/1409.0473'>Source</a>)](https://bmk.sh/images/attention.png)
## 2014: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- This paper introduced the idea of [[Attention]] -- instead of just compressing information down into a latent space in an RNN, one could instead keep the entire context in memory, and then allow every element of the output to "attend" to every element of the input, using *O(nm)* operations.
- Despite requiring quadratically-increasing compute, attention is *far more performant* than fixed-state RNNs, and have become an integral part of not only textual tasks like translation and language modeling, but also for models as distant as GANs!


## [2014: Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [[Adam]] has become a vey popular adaptive optimizer due to its ease of tuning!
- Adam is based on the idea of adapting *separate learning rates for each parameter!* 
	- Recent papers have cast doubt on the performance of Adam, but it remains one of the most popular optimization algorithms in Deep Learning.
Honorable Mention
- The [[RMSProp]] optimizer is another popular adaptive optimizer (it's not clear if it's better or worse than Adam). RMSProp is notorious for being perhaps the most cited lecture slide in deep learning.


![Residual Block Architecture (<a href='https://arxiv.org/abs/1512.03385'>Source</a>)](https://bmk.sh/images/residual.png)
*Residual block architecture*
## [2015: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Initially designed to deal with the problem of [[Vanishing Gradients]] and [[Exploding Gradients]] in deep [[Convolutional Neural Network]], the [[Residual Block]]  has become the elementary building block for almost all CNNs today.
	- The idea is simple: Add the *input* from *before* each block of convolutional layers to the *output*.
- The inspiration behind [[Residual Network]]s is that NNs should theoretically never degrade with more layers -- additional layers could/should *in the worst case* be simply set as identity mappings (but in practice, deeper networks often experienced difficulties training).
	- Residual networks made it easier for layers to learn an identity mapping, and also reduced the issue of gradient vanishing.
	- Despite the simplicity, residual networks vastly outperform regular CNNs, especially for deeper networks.

Honorable Mentions:
- Highway Networks 
- The Inception architecture is based on the idea of factoring the convolutions to reduce the # of parameters and make activations sparser.
- VGG networks, explored the use of only 3x3 convolutions, instead of larger convolutions as used in most other networks; this reduces the number of parameters significantly.
- Neural ODEs won the best paper award at NIPS 2018, drawing a parallel between residuals and Differential Equations; if you view residual networks as a discretization of a continuous transformation.

## [2015: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- [[Batch Normalization]]is another mainstay of nearly all NNs today -- it's a simple but powerful idea:
	- Keep mean and variance statistics during training, and use that to scale activations to mean zero and unit variance.
- The exact reasons for the effectiveness of Batch norm are *disputed*, but it's undeniably effective, empirically!
Honorable mentions: Other alternatives sprung up based on different ways of aggregating the statistics!
![A visualization of the different normalization techniques (<a href='https://arxiv.org/abs/1803.08494'>Source</a>)](https://bmk.sh/images/norms_comparison.png)



![Supervised Learning and Reinforcement Learning pipeline; Policy/Value Network Architectures (<a href='https://www.nature.com/articles/nature16961.pdf'>Source</a>)](https://bmk.sh/images/alphago_arch.png)## [2016: Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961)
- After the defeat of Kasparov by Deep Blue, Go became the next goal for AI community, thanks to it having a much larger state space than Chess and a greater reliance on *intuition* (among human players).
- AlphaGo combines many previous techniques (Monte-Carlo tree search, handcrafted heuristics) with big compute; AlphaGo consists of a policy network (that narrows the search tree) and a and a value network (that truncates the search tree).
	- These networks were first trained with standard Supervised Learning and then further tuned with Reinforcement Learning.
- AlphaGo had a huge impact on the public mind, with an estimated 100 million people globally tuning in to AlphaGo vs. Lee Sedol match. The influential and unconventional 37th move by AlphaGo baffled analysts and captured the imagination.
- The followup paper "[Mastering the Game of Go without Human Knowledge](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)" introduced AlphaGo Zero, which removed the supervised learning phase and trained the policy and value networks *purely* through **self-play**! Despite not being imbued with any human biases, AlphaGo Zero was able to rediscover many human strategies and invent superior strategies that challenged common Go wisdom!

![Transformer Architecture (<a href='https://arxiv.org/abs/1706.03762'>Source</a>)](https://bmk.sh/images/transformer_arch.png)
## 2017: [Attention is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
- The [[Transformer]] architecture makes use of the aforementioned attention mechanism, but at scale! It's become the backbone of nearly all state-of-the-art ML models today.
- Transformer models beat RNNs in large part due to the computational benefits in very large networks...
	- In RNNs, the gradients need to be propagated through the entire *"unrolled"* graph, which makes memory access a large bottleneck, while *also* exacerbating the exploding/vanishing gradients problem (which necessitated the more complex and expensive [[LSTM]] and [[GRU]] models).
- Instead, Transformer models are optimized for highly parallel processing -- the most computationally expensive components are 
	- the feed forward networks after the attention layers, which can be applied in parallel, 
	- and the attention itself, which is a large matrix multiplication and is also easily optimized.

![The architecture of NASNet, a network designed using NAS techniques (<a href='https://arxiv.org/abs/1706.03762'>Source</a>)](https://bmk.sh/images/nasnet.png)
*The architecture of NASNet, a network designed using NAS techniques*
## 2017: [Neural Architecture Search with Reinforcement Learning](https://openreview.net/forum?id=r1Ue8Hcxg)
- [[Neural Architecture Search]] has become common practice in the field for squeezing every drop of performance out of networks.
	- Instead of designing networks painstakingly by hand, NAS lets this process be automated! In this paper, a controller network is trained using RL to produce performant network architectures. This has created many SOTA networks.
	- Other approaches in other papers (eg AmoebaNet) use evolutionary algorithms instead.

![BERT compared to GPT and ELMo (<a href='https://arxiv.org/abs/1810.04805'>Source</a>)](https://bmk.sh/images/bert_compare.png)
## 2018: [BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [[BERT]] is a bidirectional, contextual text-embedding model. Like [[Word2Vec]], it's based on assigning each word (or rather, sub-word tokens) a vector. However these vectors in BERT are contextual, allowing *homographs* (eg *console*, the noun vs *console*, the verb) to be properly distinguished!
- BERT is deeply bidirectional, with each latent vector in each layer depending on all latent vectors from the previous layer.
	- In unidirectional LMs like GPT, the model is trained to predict the next token at each timestep, which works because the states at each timestep can only depend on previous states.

Honorable Mentions
- Since the publication of BERT, there's been an explosion of other transformer-baed language models -- they're all quite similar.
	- ELMo is arguably the first contextual embedding model, but BERT has become much more popular in practice.
	- OpenAI's Generative Pre-Training model (GPT) explores the idea of using the same pretrained LM downstream for many different types of models.
	- OpenAI's GPT-2 is in some senses simply a scaled-up version of GPT. It has more parameters (up to 1.5 billion!), more training data, and much better test perplexity across the board! It also exhibits an impressive level of generalization across datasets. Its claim to fame is its impressive text-generation abilities.
	- Transformer-XL attempts to improve on the limitation of Transformer-based models having a fixed attention window, which prevents attending to *longer-term* context. Transformer-XL attempts to fix this by attending to some context from the previous window.
	- Better tokenization techniques have also been a core part of the recent boom in language models. These eliminate the necessity of "out-of-vocab tokens" by ensuring that *all words* are tokenizable in pieces.

![Deep Double Descent (<a href='https://openai.com/blog/deep-double-descent/'>Source</a>)](https://bmk.sh/images/deepdoubledescent.png)
## [2019: Deep Double Descent: Where Bigger Models and More Data Hurt](https://arxiv.org/abs/1912.02292)
- The phenomenon of (Deep) [[Double Descent]] seems to run contrary to popular wisdom in both machine learning and in modern Deep Learning; In classical ML, model complexity follows the [[Bias-Variance Tradeoff]]:
	- Too weak a model is unable to fully capture the structure of the data, while too powerful of a model can overfit and capture spurious patterns that don't generalize to data outside the training set.
	- Because of this understanding, it's assumed that it's expected that test error will decrease a models get larger, but then start to increase again once the models begin to overfit.
- In Deep Learning practice, however, it seems that models are often massively over-parametrized and yet *still* seem to improve on test performance with larger models.
	- As the capacity of models approaches the "interpolation threshold," the demarcation between the classical ML and Deep Learning regimes, it becomes possible for gradient descent to find models and achieve near-zero error (on the training data), which are likely to be overfit...
	- But as the model capacity is increased even *further*, the number of different models that can achieve zero training error increases, and the likelihood that some of them actually fit the data smoothly (i.e. without overfitting) increases! 
	- [[Double Descent]] thus posits that gradient descent is more likely to find these smoother zero-training-error networks that *actually DO generalize well*, despite being overparametrized!

## 2019: [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
- "A randomly-initialized, dense neural network contains a subnetwork that is initialized such that -- when trained in isolation -- it can match the test accuracy of the original network after training for at *most* the same number of iterations!"
	- This seems to say to me that there's trimmable fat in NNs (though it's not obvious where the fat might be, perhaps?).
- Basically, this Lottery Ticket Hypothesis asserts that most of a network's performance comes from a certain *subnetwork*, due to a lucky initialization (hence the name, "lottery ticket", to refer to these subnetworks), and that larger networks are more performance because of a *higher chance of lottery tickets occurring!* 🎫 ✨
	- This allows us to prune the irrelevant weights, and also retrain from scratch using *only* the "lottery ticket" weights, which obtains close to the original loss!


## Conclusion
- The past decade has marked an incredibly fast-paced and innovative period in the history of AI, driven by the start of the Deep Learning Revolution.
- Neural networks have weaknesses too, though:
	- Require large amounts of data to train
	- Have inexplicable failure modes
	- Cannot generalize beyond individual tasks (?)



























