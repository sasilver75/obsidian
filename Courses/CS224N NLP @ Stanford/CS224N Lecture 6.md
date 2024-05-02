#lecture 
Link: https://www.youtube.com/watch?v=0LixFSa7yts&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=9

----

Lecture Plan:
1. RNN Language Models (25 mins)
2. Other use sof RNNs (8 minutes)
3. Exploding and vanishing gradients (15 mins)
4. LSTMs (20 mins)
5. Bidirectional and Multi-layer RNNs (12 mins)

------

Overview:
- Last lecture we learned:
	- Language models
	- [[N-Gram]] language models
	- Recurrent Neural Networks ([[Recurrent Neural Networks|RNN]]s)

Today:
- Training RNNs
- Uses of RNNs
- Problems with RNNs (exploding/vanishing gradients)
	- Motivating [[Long Short Term Memory|LSTM]]s
	- Other more complex RNN options:
		- Bidirectional RNNs
		- Multi-layer RNNs

Next lecture:
- How can we do neural machine translation (NMT) using an RNN-based architecture called "sequence to sequence with attention"?

------

Simple RNN Language Model

![[Pasted image 20240408153813.png]]
Above:
- Sequence of words, for which we've gotten word embeddings
	- We get each word embedding my doing a linear projection/matmul of the one-hot encoding of our word from our vocabulary with a learned embedding matrix.
- Then we have a neural language model where, for each point we have a hidden state, and we're feeding in to the next hidden state both the previous state *and* the transformed encoding of a word using our RNN equation (on the left). Based on this, we compute the new hidden representation at the *next* timestep.
	- The next hidden timestep is a combination of the previous hidden timestep and the incoming word (both multiplied by learned matrices), plus a bias, and then the entire thing is softmaxed.
	- We use some (randomly initialized?) hidden state for our RNN hidden state.
- We repeat this along successive timesteps.
- We also want our NN to produce outputs! At each timestep we also generate an output. To do that, we feed the hidden layer into a softmax layer; we do another MatMul+Bias+Softmax, which gives us a probability distribution over words. We use this to predict how likely it is that a given word will be the next word, given the previously generated words.
	- The output is the result of multiplying our hidden state with another learned matrix (plus a bias), and then softmaxxing it to produce a probability distribution over words in our vocabulary.


## Training an RNN Language Model
![[Pasted image 20240408155310.png]]
1. Get a big corpus of text, which is a collection of sequences of words (x1...xt)
2. Feed them into the RNN-LM; 
	- Take prefixes of the sequence
	- Based on each prefix, predict the probability distribution for the word that comes next
		- (We predict the probability of every given word, given words so far)
	- Train the model by assessing how good a job we do, compared to the *actual* word that came next
		- The loss function we use is [[Cross-Entropy]], which is a negative log likelihood loss.
3. Average this to get the overall loss for the entire training set

![[Pasted image 20240408155324.png]]
Above:
- Given our corpus of text, we run it through our RNN.
- At each position $x$ , we predict a probability distribution over words, $\hat{y}^{(x)}$  
- We know what the *actual* word (a distribution where all the mass is on one word) was, so we compare our prediction to the actual word, and we calculate a loss.
- After we've calculated individual losses for all predictions, we get an average loss over the (eg) training batch.

(At each step, we "reset" our "previous sequence" to what was actually in the corpus)

![[Pasted image 20240408160556.png]]
Above:
- Recall, we don't usually chug through a whole corpus before doing gradient updates -- we usually chop it up into pieces.

![[Pasted image 20240408161003.png]]


![[Pasted image 20240408162022.png]]
If you want to generate text with an RNN
- Use a "beginning of sequence token" that you feed in as the first token. It has its own embedding, etc.
- You use the RNN update, and generate+sample the next work using a softmax, etc.

How do you end?
- You also have an "end of sequence token", and at some point the RNN will generate this EOS token, and you say "I'm done!"

----
Let's now go on to some of the more difficult content in the lecture!

The standard evaluation metric for Langauge models is [[Perplexity]]!
- Given a series of T words, what probability do you give to the T+1'th word?
- Repeat this at each position
- Take the INVERSE of this probability and raise it to the (1/T) for the length of your text sample
- It's a geometric mean of the inverse probabilities.
- An easier way to think of it:
	- Perplexity is simply  the cross-entropy loss we introduced before, exponentiated (but the other way around, so low perplexity is better)

![[Pasted image 20240408162605.png]]
Perplexity was literally invented by some information theory people coming into NLP who wanted an easier to understand metric than Cross Entropy
- ==Perplexity of 53: How uncertain you are of the next word is equivalent to the uncertainy of tossing a 53-sided dice and it coming up as (eg) 1.==


Things stick, and to this day, everyone evaluates their language models by providing complexity measures.
As people started to build better neural networks, people started producing much better perplexities -- we're getting complexities down to (eg) 30 (using some LSTMs in the mid 2010s), or even lower today @ low single digits!

Why should we care about language modeling?
- If what we want to do is build ML models of language, and the ability to predict the next word in a context shows that we both understand the structure of language and the structure of the human world that language talks about.
- Language models are the secret tool of NLP
	- Predictive typing
	- Speech recognition
	- Author identification
	- Diarization
	- ...


Recap:
- A ==Language Model== is a model that predicts the next word in a sequence.
- A ==Recurrent NN== is a family of NNs that
	- Take sequential input of any length
	- Apply the ==same weights on each step==
	- Can ==optionally produce output== on each step
- Recurrent NNs are not necessarily language models, though!
	- EG maybe we want to generate an idea of whether a sentence is positive or negative
		- What if we run an RNN over the entire sentence; at the end, the hidden state could be said to "have the whole meaning of the sentence!"
		- We can use that end-of-sentence hidden state, in combination with a classifier, to predict the sentence sentiment!
	- EG language encoder module uses; Any time you have some text (eg "what nationality was beethoven?"), we'd like to construct some osrt of neural representation of this question.
		- We can run an RNN over it, and take the (eg) final hidden state, or some function of *all* the intermediate hidden states, and say "that's the sentence representation!"
		- We could then build some more NN structure on top of that 
	- Speech recognition
		- We want to generate some text conditioned on the speech signal! If we can break the audio into individual tokens and then have an RNN process them and output transcribed text, that'd be nifty!

## Problems with [[Vanishing Gradients]] and [[Exploding Gradients]]
- Vanishing gradients
![[Pasted image 20240408164320.png]]Above: 
- The gradient (small numbers) being multiplied by eachother to propagate to early layers of the network means that the signal peters out.

![[Pasted image 20240408164928.png]]
The vanishing gradient problems means that RNNs aren't super good at modeling long-term dependencies!

![[Pasted image 20240408165213.png]]


Exploding Gradients are also a problem!
- If gradients become too big, and we multiply many such of these together, the stochastic gradient descent update step becomes too big!
- You think you're coming up to a steep hill to climb, but the gradient is so steep that you came some enormous update, and suddenly you parameters are over in Iowa and you've lost your hill altogether!
![[Pasted image 20240408165356.png]]

![[Pasted image 20240408165504.png]]
An easy fix to the exploding gradient is [[Gradient Clipping]]
- We choose some reasonable number and say: "We just aren't going to use gradients that are bigger than this number."
	- If the norm of that gradient is greater than that threshold, we just scale down the gradient.

----


To combat both of these problems, the 1997 paper from Schmidhuber on [[Long Short Term Memory]] networks (LSTMs)
- Note this 1997 paper was actually missing the most important part of the LSTM, and the Gers/Schmidhuber paper in 2000 introduced the Forget Gate

For those that think that mastering NNs is the path to fame at fortune: At this point in time, very few people were interested in NNs.


![[Pasted image 20240408170335.png]]
Crucial innovation of the LSTM:
- Rather than having one vector as the hidden state in the model, we'll build a model with TWO hidden vectors at each timestep!
	- The ==hidden state== $h$ , of length `n`
	- The ==cell state== $c$, also of length `n`, which stores long-term information.
- In retrospect, these were named incorrectly. In many states, the "cell" state is most similar to the hidden state in the previous one.
- The LSTM can ==READ, ERASE, and WRITE== information from the cell.
	- The cell becomes conceptually rather like RAM in a computer.

The selection of which information is erased/written/read is controlled by three corresponding probabilistic `GATES`
- These gates are also vectors of length `n`

At each step:
- We work out a state for the gate vectors; each element of the gates can be open (1), closed (0) or somehwere in-between. 
- These values say: "How much do you erase, how much do you write, how much do you read"
- These values are dynamic, and computed based on context.


LSTM walkthrough
- We have a sequence of inputs $x^{(t)}$  and we will compute a series of both hidden states $h^{(t)}$ and cell states $c^{(t)}$ 

![[Pasted image 20240408171452.png]]

- At a given timestep $t$ , we start by computing the state of each of our gates, given the token $x^{(t)}$ 
	- We compute the gate values using equations that are identical to the equation that we used for the single RNN's update of the hidden state... (but we have different learned matrixes and biases for each gate)
	- Recall, there's a:
		- ==Forget Gate==, controlling what is kept in the cell at the next time step, versus what is forgotten
		- An ==Input Gate==, determining which parts of the calculated new cell content are written to the cell memory.
		- An ==Output Gate==, determining what parts of the cell memory are moved over into the hidden staet.

then we have other equations that are the mechanics of the LSTM:
- Our *candidate update*, which will calculate our new cell content! This is the new content to be written to the cell.
	- This is the usual RNN equation, but we use the tanh activation function for (-1,1)
- For our *new cell content*, we want to remember *some* but probably not *all* from the previous timesteps.
	- We want to store *some*, but probably not all of the value that we've calculated as our new cell update
	- We take the *previous* cell content, and take its ==Hadamard product (?)== with the current forget gate value... and *add* the *candidate update* hadamard producted with the current input gate's value!
- For workign out the *ne hidden state*
	- We work out which parts of the cell to expose in the hidden state, using our output gate with the cell state, using a Hadamard product.

Note:
1. Note that all of these things are being done as vectors of the same length $n$
2. The candidate update and the forget, input, output gates all have a very similar form -- the only difference is three logistics and one tanh -- and all depend on eachother, so all four of these can be calculated in parallel. If you want an efficient LSTM implementation, that's what you do!

![[Pasted image 20240408171738.png]]
Above: [[Chris Olah]]'s wonderful LSTM diagram
- We've got, from the previous timestamp, both your cell and hidden recurrent vectors. 

![[Pasted image 20240408171914.png]]You'll probably notice that while we've introduced this "long term" memory in the form of the hidden state (unfortunately named), we still have the problem along long sequences of boiling the whole world into the hidden state, so we don't get to incredibly long sequence lengths -- just longer than a vanilla RNN!

Re: the name: The idea is that there's the concept of "Short Term Memory" from psychology; it was suggested that the hidden state in an RNN is a model of human memory short term memory, and there was something else that would deal with long term memory.
Schidhuber et al were interested in how we could construct models with *long* short term memory (so there's not long term memory, just a *long* version of short-term memory!)

![[Pasted image 20240408172357.png]]

![[Pasted image 20240408172530.png]]

---

[[Residual Network]]s
- Note that the vanishing/exploding gradient problem rears its head in almost all deep neural networks (feedforward, convolutional neural networks) -- any time you have long sequences of chain rules, the gradient can become vanishingly small or explodingly large. Generally, lower/earlier layer are harder to train, as a result.
- There has been effort to come up with different architectures that learn more efficiently in deep NNs!
- The most common way is to ==add more direct connections that allow gradients to flow==
![[Pasted image 20240408172917.png]]
You have two paths that are summed together:
1. An identity path
2. Some that goes through some neural network layers
So the default behavior is to just preserve the input, which might seem a little like what we just saw for LSTMs.

There are other methods:
- DenseNets, where you add skip connections of each layer to all future layers
- HighwayNets from Schmidhuber, where, rather than just having an identity connection, it introduces an extra gate that looks more like an LSTMM, which says how much to send the input through the highway versus how much to put it through a neural net layer; these two are then combined into the output.


![[Pasted image 20240408173053.png]]
Above:
- This problem first arose with recurrent NNs; they're particularly unstable because you have this one weight matrix being repeatedly used throughout the time sequence.


-----

[[Bi-Directional Recurrent Neural Network]] and multi-layer RNN motivation

Say we wanted to do sentiment classification using an RNN
- WE had the thing where we said that we'd run an RNN over it, and take some combination of the hidden states across timesteps as the representation of the sentence, and then feed it into a softmax classifier to classify for sentiment.

![[Pasted image 20240408174209.png]]

Above:
- So the hidden state at each word can be thought of (sort of) as the contextual representation of that word in the sentence (kinda).
- But we calculate this hidden state left to right.
- So we'd look at the one for "terribly" and the RNN will think, ah, this is a bad word! Add more "bad" to the hidden state!
- But we know that terribly in this context actually means "very!", once you see the next word "exciting"!

![[Pasted image 20240408174441.png]]
This motivates the introduction of a bidirectional RNN!
- An easy way to deal with this would be to just have another RNN that runs "backwards" through the sentence! 
- This second RNN has its own completely separate, learned parameters.
- We could run it backwards through the sentence
- We could get an "overall" representation of a word in a sentence by simply ==concatenating the representations from the forward and backward RNNs!==

![[Pasted image 20240408174641.png]]