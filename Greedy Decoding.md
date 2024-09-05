Language models generate a probability distribution over the next token.
The "decoding" process refers to the manner in which we choose to sample from this distribution.
"Greedy decoding" means that we always choose the token with the most probability mass.

Note: Decoding strategies are orthogonal to that of [[Temperature]], which influences the *shape of the upstream probability distribution*.

![[Pasted image 20240627234906.png|100]]
Here's an example of a generation.
Although each token in the sequence is the likeliest token at the time of prediction, the selected tokens of "being" and "doctor" were both assigned relatively low probabilities -- this might suggest that "of", our first predicted token, may not have been the most suitable choice, since it leads to "being," which is quite unlikely.
- But we didn't have the ability to "look ahead" at the time, which is a reason why techniques like [[Beam Search]] exist.