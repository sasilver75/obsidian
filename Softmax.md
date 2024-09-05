Given raw values in the \[-infty, infty\] range, softmax scales these values to \[0,1\].
Mentally, you can think of this as a function that "create probability distributions," since the outputs are positive and sum to one.
This is used to do things like create a probability distribution over the entire vocabulary, and can be modified by adding a [[Temperature]] parameter to make the resulting distribution flatter or peakier.

![[Pasted image 20240521220209.png]]

Convert a vector representation into a vector that represents a probability distribution (over the entire vocabulary (in the sense that every element is positive and sums to one))