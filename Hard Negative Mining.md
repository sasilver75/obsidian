Resources:
- Eugene Yan's [More Design Patterns for Machine Learning](https://eugeneyan.com/writing/more-patterns/?utm_source=convertkit&utm_medium=email&utm_campaign=2023+Year+in+Review%20-%2012699108)


Hard mining can be considered an extension of data augmentation where we find or generate challenging data points to train the model on.

The intent is to improve model performance on difficult cases by exposing it to more of them during training.

Pros:
- Often leads to improve model performance. At the very least, we can measure how our models perform on these difficult cases.

Cons:
- Finding these hard examples can be, well, hard. It's also not as simple as training the model solely on these hard examples too.

One approach to hard mining is to analyze model predictions for misclassified or low-confidence examples, find similar examples (e.g. nearest neighbors), and emphasize them in subsequent training. This forces the model to learn from its mistakes and improve.

When Meta built embedding-based retrieval for search, they used hard mining to address the problem of easy negatives in the training data (negative examples that are not challenging for the model to distinguish from positive examples). Counterintuitively, we find that models trained on hard negatives didn't perform better than models trained on random negatives -- this is because the hardest negatives didn't reflect the actual search behavior on Facebook where most documents (eg people, places) were relatively easy. Authors found that sampling hard negatives from rank 101-500 (in the search results) led to the best performance. Blending random and hard negatives improved recall, saturating at an easy:hard ratio of 100:1.
- This is sort of sad, I suppose. But I'm sure it helped in the small tail of examples where people were searching using obscure queries, right?

Another approach is to blend hard negatives in [[Curriculum Learning]], where training data is sorted to gradually increase sample difficulty. This model starts learning from easy samples, before progressively tackling harder samples.
- Cloudflare used this approach by training their malicious payload classifier on easy synthetic data, followed by increasingly difficult synthetic data, before finally fine-tuning on real data. To make the task harder, they appended noise of varying complexity to both malicious and benign samples, with the goal of making the model more robust to padding attacks. This improved the true positive rate for fuzzed content from 91% to 97.5%.

