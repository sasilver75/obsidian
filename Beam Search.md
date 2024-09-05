References
- [Video: How is Beam Search Really Implemented?](https://youtu.be/tOhWpF5-_z4?si=mu2u62mObdpZekTu)
- Video: CMU Advanced NLP 2024 L6 @ 19:36

Beam search is a heuristic search algorithm widely used in fields like [[Natural Language Processing]] and [[Automatic Speech Recognition]] to find the *most likely sequence of states or elements*, given a model. 

Idea: We don't want to miss a good high-probability generation that's "hidden" behind a low-probability prefix.
- We use a type of BFS to explore many options for each decoding step, before generating candidates for the next step.

![[Pasted image 20240614175123.png]]

We pick a number of candidates (beam width) we want to explore at a given timestep.
- At each timestep, from each of our generations, we choose k most-likely generations from our current positions.

In general, this works much better than [[Greedy Decoding]], but there are some issues with beam search:
- When we do maximum likelihood sampling, we might be sacrificing a lot of diversity in our outputs; we might end up with 3 generations that are all pretty much the same.
If we we want more diversity, we might want to retain some of the benefits from sampling *during* our beam search for high probability
- "==Diverse Beam Search==": Modify the scoring step when pruning beams to avoid choosing overly-similar beams. (You can use any similarity function you'd like, here. In the original paper, she thinks they use some sort of lexical matching metric)
- "==Stochastic Beam Search==": We keep the scoring the same, but instead of choosing the top (eg) 3 most likely tokens to expand each beam, we sample from some distribution (using whatever sampling you'd like). The idea is to get wider exploration of our model's output space.

Note that Beam Search has ==mode-seeking behavior;== it's whole goal is to generate sequences with high probability -- and it's likely that the most probable generations aren't the ones that are best for your task!
- In fact, in many cases, like "What's your favorite dance move," something like "I don't know" is likely to be more probable than any specific output that we'd actually want. This is because we're maximizing likelihood of the observed data.

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

There's such a thing as the "==Curse of Beam Search==", where, as we increase the number of beams, our generations become more and more probable, but actually *worse* for our downstream tasks!
- A great paper on this concept is called *On the Inadequacy of the Mode* (since BS is mode-searching).



![[Pasted image 20240528222827.png]]
From CS685 Advanced NLP