#lecture 
Link:https://www.youtube.com/watch?v=X0Jw4kgaFlg&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=3

-----

If this material seems difficult and scary, take heart -- this is really the only lecture that dives deep down into neural networks! In the future we'll use software to do a lot of the complicated math for us!
But it's encouraged that you work through it and get help when you need it!


# Named Entity Recognition (NER)
- [[Named Entity Recognition]] is a common NLP task where the goal is ot look through pieces of text and you want to label words as to what entity category they belong to (people, dates, times, cars, etc.)
![[Pasted image 20240401174928.png]]
Above:
- Some words are being labeled as named entities, illustrating the point
- You can't just do this task by using a dictionary -- yes a dictionary can be *helpful*, but you have to use context (see: Paris) to get NER correct.

How might we do this with a Neural Network?
![[Pasted image 20240401175011.png]]
 
- One way of doing NER with a simple NN is to use the word vectors that we've previously learned about
- We consider a context window of word vectors, and put them through a *logistic classifier* and feed through a softmax classifier to say whether it's a high probability that something is (eg) a *location* or not.
	- For each word, we form a window around it (say +/- 2 word window); for those 5 words, we get word vectors for them (eg from [[Word2Vec]] or [[GloVe]]), and we make a single *long* vector by concatenating these word vectors, and then feed the vector to a classifier that will output a probability that a word is (eg) a location
	- We could have another classifier saying the probability that something is (eg) a *person's name*.

![[Pasted image 20240401175543.png]]
Above:
- The layer in the NN multiplies the vector by a matrix, adds a bias vector, and adds a nonlinearity (eg the softmax transformation). 
- This resulting hidden victor (which might be a smaller dimensionality) is then dot-producted with some extra vector $u^T$ to get us a single, real-valued number.
- We then put that number through a logistic transform of some kind, giving us a probability distribution

Recall: Our Stochastic Gradient Descent equation:
![[Pasted image 20240401180747.png]]
Above:
- Determine loss
- Move our parameters in the direction opposite the gradient, modulated by some learning rate

But how do we compute the gradient of the loss function with respect to our parameters?
1. By Hand
2. Algorithmically, using the backpropagation algorithm!

Lecture Plan:
1. Intro
2. Matrix Calculus (40 mins)
3. Backpropagation (35 mins)
	- Efficient application of calculus on a large scale

![[Pasted image 20240401181039.png]]
You should be able to do this stuff if you remember some of the basics of single-variable calculus!
You can also look at the textbook for Math51, which has a lot of material on this.


![[Pasted image 20240401181210.png]]





