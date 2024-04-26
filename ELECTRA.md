March 23, 2020
[ELECTRA: Pre-training Text Encoders as Discriminators Rather than Generators](https://arxiv.org/abs/2003.10555)

References:
- [XCS224U: ELECTRA](https://youtu.be/QFMBRk26AjU?si=xbGgu7NRtuP0NCCA)
	- See Lecture Notes: [[XCS224U Lecture 3]]

Aims to improve on some limitations of BERT, which included (among others):
- A mismatch between pre-training and fine-tuning, since the MASK token isn't seen during fine-tuning.
- The second downside of using an MLM is that only 15% of tokens are predicted in each batch. Only 15% even contribute to the MLM objective, despite processing every item in the sequence -- this isn't very data efficient!

![[Pasted image 20240425151728.png]]
- Given sequence $x$ : the chef cooked the meal
 - Create $x_{mask}$ , which is a masked version of the input sequence; can use the same protocol as BERT, say, by masking 15% of the tokens at random
- The generator, a small BERT-like model that processes the input and produces $x_{corrupt}$ , where we replace some of the original tokens not with their original inputs, but with tokens having probability proportional to the generator probabilities.
	- Sometimes we'll replace with the actual token, and other times we'll replace with some other token
- The discriminator, the heart of the Electra model, has a job that is supposed to figure out which tokens in the sequence are original, and which are replaced.
- we train the model jointly with the generator and the discriminator, and allow the generator to drop away, and focus on the discriminator as the primary pre-trained artifact produced by the process.

![[Pasted image 20240425180150.png]]