---
aliases:
  - FiD
---
July 2020
Link: [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282)
See also: [[KG-FiD]]

Uses retrieval with generative models for [[Open-Domain]] [[Question Answering]]. It supports both BM25 and DPR for retrieval. Named FiD for how it performs fusion on the retrieved documents.
![[Pasted image 20240414161436.png]]
For each retrieved passage, the title and passage are concatenated with the question (we also add special `question:`, `title:`, and `context:` tokens before corresponding sections). These pairs are processed independently in the encoder. The decoder attends over the *concatenation* of these retrieved passages.
