---
aliases:
  - Ancestral Decoding
---
A decoding method for NNs where we just sample from the distribution produced by the model.

![[Pasted image 20240614195610.png]]

Issues with Ancestral Sampling:
- Human language has a long-tailed distribution in practice! There are 32k vocabulary tokens in LLaMA, and while many of the ones on the long tail are low likelihood, the cumulative probability of the bottom (eg) 30k vocabulary terms is very high, so you're likely to sample from somewhere on the tail!
![[Pasted image 20240614195720.png]]

So you might think: "Well... can we just avoid/ignore the long tail?"
- Yeah! This is where techniques like [[Top-K Sampling]], [[Top-P Sampling]], etc. come in.