---
aliases:
  - MBR
  - MBR Decoding
---
![[Pasted image 20240614195937.png|300]]
Even though the green ones weren't the most likely generations, they're nearly-synonymous and their cumulative probability is high!
"The cat sat down" is high probability, but is very different from the other generations. Is it dissimilar because it's better, or dissimilar because it's worse?
"The cat an away" is high probability and similar to other generations, so it's lower risk!
What we ideally want is an output that's *relatively high probability and low risk*! Here, that might be "the cat ran away."
- How do we get these outputs? 
	- How do we estimate probability?
		- We usually can't sample everything from the model, so we take a sample Y_e (eg 100 samples) and use this instead.
	- How do we estimate risk?
		- Idea: Treat the rest of the candidate generations as "pseudo-references," and evaluate the agreement with each of them, using some kind of similarity metric (n-gram overlap like ROUGE or BLEU, or something Neural like BERTScore or BARTScore.)
- This is the introduction to "[[Minimum Bayes Risk Decoding]]" (MBR).

![[Pasted image 20240614214736.png]]
- This equation captures the intuition we were talking about above, where we choose something low risk (similar to a lot of other things in our set of outputs we sampled) and relatively high probability.
We choose some metric `G` (eg [[ROUGE]], [[BERTScore]]) 

![[Pasted image 20240614215116.png]]
This method seems to output [[Greedy Decoding]] and [[Beam Search]] (BS)
- Note that BS with k=10 seems to perform worse than BS with k=5, which is often called the ==Curse of BeamSearch==; the generations will be higher-probability (since that's what we're searching for), but worse for downstream tasks. A great paper on this concept is called *On the Inadequacy of the Mode* (since BS is mode-searching).


![[Pasted image 20240614215816.png]]


![[Pasted image 20240614215935.png]]
Another thing in this category is something called [[Self-Consistency]]
1. We prompt the model for an answer using [[Chain of Thought|CoT]]
2. We sample multiple outputs using this
3. We completely throw away the CoTs and just take the majority vote of the answer from each generation.
This is actually an MBR where the metric we care about "exact match of this answer," ignoring the rest of the generation. It's high probability and low risk (because other generations voted on it too)