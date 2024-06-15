References
- [Video: How is Beam Search Really Implemented?](https://youtu.be/tOhWpF5-_z4?si=mu2u62mObdpZekTu)
- Video: CMU Advanced NLP 2024 L6 @ 19:36

Beam search is a heuristic search algorithm widely used in fields like [[Natural Language Processing]] and [[Automatic Speech Recognition]] to find the *most likely sequence of states or elements*, given a model. 

Idea: We don't want to miss a good high-probability generation that's "hidden" behind a low-probability prefix.
- We use a type of BFS to explore many options for each decoding step, before generating candidates for the next step.

We pick a number of candidates (beam width) we want to explore at a given timestep.
- At each timestep, from each of our generations, we choose k most-likely generations from our current positions.

In general, this works much better than [[Greedy Decoding]], but there are some issues with beam search:
- When we do maximum likelihood sampling, we might be sacrificing a lot of diversity in our outputs; we might end up with 3 generations that are all pretty much the same.
If we we want more diversity, we might want to retain some of the benefits from sampling *during* our beam search for high probability
- "==Diverse Beam Search==": Modify the scoring step when pruning beams to avoid choosing overly-similar beams. (You can use any similarity function you'd like, here. In the original paper, she thinks they use some sort of lexical matching metric)
- "==Stochastic Beam Search==": We keep the scoring the same, but instead of choosing the top (eg) 3 most likely tokens to expand each beam, we sample from some distribution (using whatever sampling you'd like). The idea is to get wider exploration of our model's output space.

![[Pasted image 20240614175123.png]]

![[Pasted image 20240528222827.png]]
From CS685 Advanced NLP