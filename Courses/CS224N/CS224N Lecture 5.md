
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
- We start with an input representation of an input
- We put it through a hidden layer with a MatMul and a nonlinearity (this transforms the space and maps it around)
- The output of this can be put into a softmax layer, from which we can make our classification decisions.
- We then backpropagate the loss to the parameters of our model



