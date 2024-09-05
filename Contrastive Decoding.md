Resources:
- [VIDEO: CMU Advanced NLP 2024 L6 @ 15:00](https://youtu.be/96MMXDA7F74?si=DjLiIokqK3JSl4ID)

Compare with [[Top-K Sampling]], [[Top-P Sampling]], [[Epsilon Sampling]], [[Beam Search]], etc.

The idea is to incorporate some extra information at decoding time using another model; if you've ever fooled around with small language model (eg GPT-2-small), you give it some inputs and it eventually degenerates into repeating the same sequence over and over, or hallucinating answers to questions; you don't see this as often with larger models. Can we use what the smaller model is getting wrong to make our larger model better?

If the smaller model doesn't have a lot of probability on some answer, but the larger model does, it's likely because the larger model has learned something the smaller model doesn't know.

We take the output coming from our expert model, and *subtract* the probabilities coming from our weaker model, and end up with a probability distribution measuring things the stronger model knows that the weaker model does not.

![[Pasted image 20240614173701.png]]
