---
aliases:
  - SHP
---
??? -- Stanford
(I can't find the paper that this dataset was released with/under)

SHP is a dataset of ==**385K collective human preferences** over responses to questions/instructions in 18 different subject areas, from cooking to legal advice==. The preferences are meant to reflect the helpfulness of one response over another, and are intended to be used for training RLHF reward models and NLG evaluation models (e.g., [SteamSHP](https://huggingface.co/stanfordnlp/SteamSHP-flan-t5-xl)).

Each example is a Reddit post with a question/instruction and a pair of top-level comments for that post, where one comment is more preferred by Reddit users (collectively). SHP exploits the fact that if comment A was written _after_ comment B but has a higher score nonetheless, then A is ostensibly more preferred to B. If A had been written before B, then we could not conclude this, since its higher score could have been the result of more visibility. We chose data where the preference label is intended to reflect which response is more _helpful_ rather than which is less _harmful_, the latter being the focus of much past work.

How is SHP different from Antrophic's [[Helpful and Harmless]] dataset? Most notably, all the data in SHP is ==naturally occurring and human-written==, whereas ==the responses in HH-RLHF are machine-written==, giving us two very different distributions that can *complement* each other.

![[Pasted image 20240420014126.png]]
