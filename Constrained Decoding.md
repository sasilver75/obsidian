---
aliases:
  - Constrained Generation
---
References:
- [VIDEO: CMU Advanced NLP 2024 (6): Generation Algorithms](https://youtu.be/96MMXDA7F74?si=hjzP2vmai5keSfNv&t=3749)


What if we *don't* want our model to generate some certain phrase?
![[Pasted image 20240614221829.png]]
![[Pasted image 20240614221823.png]]

What if we did something more simple?
![[Pasted image 20240614221910.png]]
Well, this is still pretty inefficient.

Maybe we can guess at some point during our generation if the sequence is going to end up being about climbing, and taking some action (restarting, changing direction) to correct this.

A paper called ==FUDGE== (Yang and Klein, 2021) tackles this (they were trying to predict "formal" sentences)
- At each step of prediction, we get the output of what the model predicts is the next token.
	- We also have some second distribution that says; "Given what we have so far, how likely are we to have a model that's formal, at the end?"
![[Pasted image 20240614222058.png]]
- We combine these distributions by multiplying them together, and then sample from *that* distribution!
- This is another method of modifying our sampling distribution with some external information, and results in sentences that are more likely to be formal, without requiring (eg) regeneration of sequences.

But how do we get this red box?
- In FUDGE, FUD stands for "Future Discriminator"; They train a model on prefixes to guess whether that sequence will be formal or not.
- This idea of trying to guess at a given decoding step if we're going to end up with our constraint satisfied is a key way of doing constrained decoding.

But maybe one of the constraints we care about is a little more nebulous, like wanting to satisfy human preferences?
- We'd usually do this via [[Reinforcement Learning from Human Feedback|RLHF]], but what if we wanted to skip the whole fine-tuning step, or add some additional protection?

In *==Reward-Augmented Decoding==* (2023, Deng & Raffel), we again have another sidecar model that predicts the reward of the final sequence, at each step of the generation.
![[Pasted image 20240614222457.png]]
This gives some of the benefits of RLHF without having to do Reinforcement Learning, and it works in a very similar way to FUDGE.
- ((A problem here though, relative to the lightweight classifier in FUDGE, is that reward models generally have to be pretty large; to do a forward pass at every step of generation is expensive!))