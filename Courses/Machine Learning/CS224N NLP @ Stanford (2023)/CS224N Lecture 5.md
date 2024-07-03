
#lecture 
Link:https://www.youtube.com/watch?v=PLryWeHPcBs&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=5

----

Agenda:
- Neural Dependency PArsing (20 mins)
- A bit more about NNs (15 mins)
- Language modeling + RNNs (45 minutes)
	- New NLP task: Language modeling
	- Motivates: a new family of NN:s Recurrent Neural Networks!



![[Pasted image 20240402141850.png]]

![[Pasted image 20240402141839.png]]
At the end of the day, these traditional ML classifiers aren't that powerful; they only give linear decision boundaries
- ((I think an SVM with Kernel Trick could be interpreted as a nonlinear boundary in the original representation space, no?))

![[Pasted image 20240402142112.png]]

Effectively, what happens is:
- the classification decisions are linear as far as the top-level softmax is concerened
- But nonlinear in the original representation space

The NN warps space around and moves the datapoints around such that they're linearly separable!
This is what the simple Feed-Forward Network linear classifier does!

![[Pasted image 20240402142902.png]]
Above:
- We start with a (dense) representation of an input to our classifier
- We put it through a hidden layer with a MatMul and a nonlinearity (this transforms the space and maps it around)
- The output of this can be put into a softmax layer, from which we can make our classification decisions.
- We then backpropagate the loss to the parameters of our model

He's introduced a [[Rectified Linear Unit]], which is the activation function that we'll be using in our classifiers.
- We'll come back to this in a minute


Let's talk about our NN dependency parser model architecture!
- It essentially the above but applied to the transition based dependency parser configuration


![[Pasted image 20240402143057.png]]
Construct an input layer embedding by looking up the various elements (?)
Feed it through the hidden layer to the softmax layer, get probabilities
From the probabilities, we choose what the next action is. No more complicated than that!


Dependency Parsing for Sentence Structure
- Chen and Manning (2014) showed that neural networks can accurately determine the structure of sentences, supporting meaning interpretation.
![[Pasted image 20240402143133.png]]



![[Pasted image 20240402143350.png]]

A few slides on Graph-based dependency parsers
((I'm skipping these, I don't really care about them very much.))

A bit more on Neural Networks
- Questions you should ask
	- How do we initialize the weights in our neural network?
	- What kind of optimizers do we use?
	- What sort of regularization techniques do we use?

The first thing we don't discuss at all is the concept of regularization:
- We're now building models with a huge number of parameters.
- Usually, our models' loss functions are going to include a regularization component! (eg for [[L2 Regularization]])

![[Pasted image 20240402144149.png]]
Above:
- We're sticking a regularization tool that sums the square of every parameter in the model. 
- What this says:
	- You only want to make parameters non-zero if they're *really* useful! Parameters that don't help much are pushed towards zero -- only the meaningful ones should be allowed to grow in size.
		- Note: This is only assessed once per batch, not for each example.
- The classic problem is ==overfitting==
	- As you train, your training error will continue to go down as you fit a line closer and closer to it
	- After a point, your test error rate will start to grow, as your model begins fitting to what is basically *noise* in the training set.

==We use regularization to make sure that our models generalize well to out-of-training-sample data.==
- Our test error will always be higher than our training error, but we want to minimize our test error.

Our big NNs today hugely overfit on the training data -- our NNs have so many params that you can continue to train on the training data until the training data is *zero!*
- In general providing that models are regularized well, those models will still generalize well and predict will on new data.

We want to figure: How much do we regularize?
- The lambda parameter in the pink box above determines the strength of our regularization.
- You don't want it to be too big (won't fit the training data well) or too small (won't generalize well)

L2 regularization essentially *doesn't cut it* for our biggest model!
We need more!

![[Pasted image 20240402145158.png]]
We use [[Dropout]] (Hinton et al, 2014)
- It's something you do while training to avoid feature co-adaptation!
- At training time, for each instance of evaluation (or for each batch), for each neuron in the model, randomly set 50% of the inputs to each neuron to 0!
- Test time: Halve the model weights (now there are twice as many!)
- (Except usually only drop first layer inputs a little (~15%) or not at all)
- This prevents feature co-adaptation: A feature cannot only be useful in the presence of particular other features.
	- ((This is weird to me, because we know that we build hierarchical representations))
- In a single layer: A kind of middle-ground between Naive Bayes (where all features are set indepedently) and logistic regression models (where weights are set in the context of all others)
- ==Can be thought of as a form of model bagging== (esembling) because you're using a different subset of features.
- Nowadays can usually be thought of as a strong, feature-dependent regularizer
	- (Different features can be regularized different amounts to maximize features)


![[Pasted image 20240402145001.png]]

"Vectorization"
- If you want to have your NNs go fast, it's important that you use vectors, matrices, tensors -- don't use for loops!


On Nonlinearities
- You *have* to have nonlinearities! No composition of linear functions will ever become nonlinear! Combinations of linear functions can themselves be shown as a linear function!

Logistic/"Sigmoid"
- -: It moves everything into the positive space, since the output is between (0, 1)
- As a result, people thought that it might be useful to have a variant, Hyperbolic Tan, that was more expressive?

Tanh (Hyperbolic Tan)
- Output in the (-1, 1) range
- recall: Hyperbolic Tan can be represented in terms of exponentials as well. It's actually the case that a tanh is just a rescaled and shifted sigmoid.
- -: For both this and logistic, calculating exponentials is quite slow for computers. People began to think: What else can we do?

Hard Tanh
- Surprisingly, these kinds of models appeared to be very successful!
- This led into what proved to be the most successful nonlinearity in a lot of recent deep learning work: The ReLU (Rectified Linear Unit)

ReLU: [[Rectified Linear Unit]]
- rect(z) = max(z, 0)
- ((There are some variants of this, e.g. [[Leaky ReLU]], [[Swish]], etc.))
- It seems weird -- how could this be useful? Think about how you can approximate things with piecewise linear functions very accurately -- you might start to think about how you can use this to approximate nonlinear things via combinations of piecewise linear functions.

![[Pasted image 20240402145810.png]]
We still use logistics for probability outputs, but they're no longer the default for making deep networks
You should think about using ReLU nonlinearities as a starting point. They train very quickly -- it's interesting that the simplest nonlinearity imaginable is enough to make a complicated NNs.



But how do we initialize our parameters?
![[Pasted image 20240402150114.png]]
- NNs just don't work if you start matrices off as 0 -- everything is symmetric, nothing can specialize in different ways, and you just don't have the ability for a NN to work.
- We set that *r* range so that the numbers in our NN don't get to big or small
	- We use [[Xavier Initialization]] to look at the fan-in and fan-out of connections in the neural network.


![[Pasted image 20240402151422.png]]
- ((There are some tips on how to choose a good learning rate -- see Jeremy Howard's tricks from fast.ai))
- There's been an explosion of optimizers
	- [[Adagrad]]
	- [[RMSProp]]
	- [[Adam]], [[AdamW]]  <--- Adam is a good place to start; In PyTorch, you can just say "Please use Adam"
	- SparseAdam
- They all basically keep track of some additional information (eg momentum) to help adjust the learning rate.

![[Pasted image 20240402151615.png]]
If you *are* using simple stochastic gradient descent, you have to select a *Learning Rate*
- You don't want it too big (diverge) or too small (takes too long, might land in a bad minimia?)
- ((Really, just use Jeremy Howard's learning rate finder techinque))

A common recipe:
- Decreasing LR as you train
	- Perhaps every k epochs you half the learning rate
	- By a formula: $lr = lr_0e^{-kt}$ 
- When you make a pass through the data, you don't like to go through the epoch the same order every time -- ==when possible, you should shuffle your data every epoch!==
- Fancier optimizers still use a learning rate, but use an initial learning rate that the optimizer shrinks over time.


# Language Modeling

![[Pasted image 20240402152003.png]]
- Language modeling is the task of predicting what word comes next
- A language is a probability of next words, given a preceding context.

You can think of a Language Model as a system that assigns a probability to a piece of text.
For example, if we have some text, then the probability of that text is: {decomposition using probability chain rule}

![[Pasted image 20240402152033.png]]


The traditional techniques that powered language models were decades were ==n-gram language models==
- We work with n-grams, which are just chunks of consecutive words
	- unigrams: "the", "students," "opened"
	- bigrams: "the students", "opened their"
	- trigrams: "the students opened", ...
	- 4-grams
	- 5-grams
	- ...

We make a ==Markov Assumption== for our Markov Models:
- The word $X^{t+1}$ ***ONLY*** depends on the preceding $n-1$ words

![[Pasted image 20240402152834.png]]
- Using the definition of conditional probability:
	- $P(B|A) = P(A\ and\ B)/P(A)$ 

Here's an example of that!
- Say we're learning a 4-gram language model!
- We throw out all words besides the last 3 words: those are our conditioning
- We use the counts from some large training corpus, and see how often "students open their books" occured, "students open their minds" occurred, etc...  and just do a simple division of our desired word over the sum of all the words
![[Pasted image 20240402152935.png]]
Above:
- ==We can see the disadvantage of the Markov assumption, where we got rid of all of the earlier context that would make this trigam easier to predict!==


![[Pasted image 20240402153547.png]]
Above:
- You have sparsity problems with the semantics being distributed over a sentence that make n-gram language models not work very well!
	- ((Think: You need larger and larger datasets and storage to really make good use of high-n n-gram models))
- What if your numerator is zero in the probability equation? That doesn't really make sense, right? Well we just add a small $\delta$ term to make our probabilities non-zero. This is called ==Smoothing==
- What if the denominator is zero/not-available? We have to shorten our context. This is called ==Backoff==



Our models also get huge! We have to store counts of word sequences, which is expensive!

You can see taht we could use a series of n-gram language models to generate text too! 

![[Pasted image 20240402154029.png]]
![[Pasted image 20240402154604.png]]

![[Pasted image 20240402155207.png]]