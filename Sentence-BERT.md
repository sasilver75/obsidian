---
aliases:
  - sBERT
---
August 27, 2019
Paper: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

Refer to: [[Bidirectional Encoder Representations from Transformers|BERT]]

Abstract:

> BERT (Devlin et al., 2018) and RoBERTa (Liu et al., 2019) has set a new state-of-the-art performance on sentence-pair regression tasks like semantic textual similarity (STS). However, it requires that both sentences are fed into the network, which causes a massive computational overhead: Finding the most similar pair in a collection of 10,000 sentences requires about 50 million inference computations (~65 hours) with BERT. The construction of BERT makes it unsuitable for semantic similarity search as well as for unsupervised tasks like clustering.  
> In this publication, we present Sentence-BERT (SBERT), a ==modification of the pretrained BERT network that use siamese and triplet network structures== to derive ==semantically meaningful sentence embeddings== that can be compared using cosine-similarity. This reduces the effort for finding the most similar pair from 65 hours with BERT / RoBERTa to about 5 seconds with SBERT, while maintaining the accuracy from BERT.  
> We evaluate SBERT and SRoBERTa on common STS tasks and transfer learning tasks, where it outperforms other state-of-the-art sentence embeddings methods.

((There's something about the embeddings being produced by (eg) BERT not being semantically meaningful, whatever that means? I feel like I have some misunderstanding here))