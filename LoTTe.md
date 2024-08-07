---
aliases:
  - Long-Tailed Topic-Stratified Evaluation
---

A dataset introduced in the [[ColBERTv2]] paper. Its name (pronounced like the latte coffee beverage) is a play on words on [[BEIR]] (pronounced "Beer"), another benchmark in IR.
- In ColbertV2, we evaluate in-domain (in the training distribution, on MS MARCO), and out-of-domain (how well can these models work in settings that you didn't train them for? For this we use LoTTe, where the intuition was about long-tail question-answering -- not things like "Who is Tom Cruise," but on niche things like StackExchange for Bicycles)
- Key thing: "We made a specific, explicit distinction between having a dev set that's out of distribution, and a test set that's out of distribution, giving you a way to track your progress on OOD domains without having a risk of overfitting on the test data."