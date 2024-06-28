## In NLP/Sequence Context
- We might have input sequences of various lengths in our batch, but our model generally requires that our tensors be rectangular, meaning that we have to pad and/or truncate our inputs to be both:
	- All the same length
	- All <= the maximum sequence length of the model.

It makes sense most to applying the padding to your sequences when you're batching them, so that the longest sequence in your *datasets* doesn't dominate the padding requirements of other batches.
Applying batching and padding at the same time is called "Dynamic Batching." This is in contrast to just picking some sequence length and padding all sequences in your dataset that number.

![[Pasted image 20240620172703.png|350]]

## In Computer Vision Context
- c.f. [[Same Padding]], [[Zero Padding]]
![[Pasted image 20240628021406.png]]
