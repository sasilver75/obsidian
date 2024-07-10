---
aliases:
  - NLI
  - Textual Entailment
---
A task in NLP typically posed as a closed classification problem where, given a Premise (P) and a Hypothesis (H),  produce give one of three labels (entailment, contradiction, neutral).
- Entailment: H is true, based on P
- Contradiction: H cannot be true, if P is true
- Neutral: H may or may not be true, based on P

Example: 
- Premise: "The cat is sleeping on the couch."
	- H1: "There is a cat on a piece of furniture." (Entailment)
	- H2: "The dog is barking loudly." (Neutral)
	- H3: "The cat is playing with a toy."  (Contradiction)

Requires an understanding of context, common-sense reasoning, sometimes "world knowledge", dealing with ambiguity and nuanced language, and handling negation and complex sentence structures.


Datasets: [[Stanford Natural Language Inference|SNLI]] (Stanford NLI), [[MNLI|MultiNLI]] (Multi-Genre NLI), XNLI (Cross-lingual NLI)