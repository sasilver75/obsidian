January 16, 2013
Paper: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

Abstract
> We propose ==two novel model architectures== for computing ==continuous vector representations of words== from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks. We observe large improvements in accuracy at much lower computational cost, i.e. it takes less than a day to learn high quality word vectors from a 1.6 billion words data set. Furthermore, we show that ==these vectors provide state-of-the-art performance== on our test set for measuring syntactic and semantic word similarities.

Introduces: [[Continuous Bag of Words]], [[Skip-Gram]] algorithms to generate dense word representations (dense vectors) from large datasets.
- [[Continuous Bag of Words]] (CBOW): "Given the surrounding words, what's the probability of the center word?"
- [[Skip-Gram]]: "Given the center word, what's the probability of the outside words?"