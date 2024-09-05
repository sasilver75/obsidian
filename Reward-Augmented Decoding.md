References:
- [VIDEO: CMU Advanced NLP 2024 (6): Generation Algorithms](https://youtu.be/96MMXDA7F74?si=hjzP2vmai5keSfNv&t=3749)

A type of [[Constrained Decoding]]

In *==Reward-Augmented Decoding==* (2023, Deng & Raffel), we again have another sidecar model that predicts the reward of the final sequence, at each step of the generation.
![[Pasted image 20240614222457.png]]
This gives some of the benefits of RLHF without having to do Reinforcement Learning, and it works in a very similar way to FUDGE.
- ((A problem here though, relative to the lightweight classifier in FUDGE, is that reward models generally have to be pretty large; to do a forward pass at every step of generation is expensive!))