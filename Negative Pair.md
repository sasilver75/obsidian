See also: [[Hard Negative Mining]]

An important part of training retrievers is to provide (if possible) ($q, d_+, d_-$) triplets.
The strategy to sample negative examples (assuming you don't have a lot of human expert time to generate these for your dataset, a strategy that doesn't scale to meaningful dataset sizes) is an important one, and there are many options, including:
1. Explicit Negatives
2. Random Negatives
3. BM25 Negatives
4. Gold Negatives
5. In-batch Negatives
6. Cross-batch Negatives
7. Approximate Nearest Neighbors
8. Hybrid

These are described more in [[Positive and Negative Sampling Strategies For Representation Learning in Semantic Search (March 22, 2023) {Sumit Kumar}]]

