References:
- [Positive and Negative Sampling Blog post](https://blog.reachsumit.com/posts/2023/03/pairing-for-representation/)

Options for determining Positive examples when positive labels haven't been provided include (among others):
- Explicit Positives
- [[Inverse Cloze|Inverse Cloze Task]]
- Recurring Span Retrieval
- Independent Cropping
- Simple Text Augmentations
- Others (masked salient spans, random cropping, neighboring text pieces, top-k BM25 passages)

Note that while [[Hard Negative Mining]] gets a lot of attention, Hard Positive Mining is important too! In Facebook Search, authors mined potential target results for failed search sessions from searchers' activity logs; they found these hard positives improved model's effectiveness.