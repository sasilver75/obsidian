---
aliases:
  - MRR
---
Simply the average of [[Reciprocal Rank]] scores over multiple Queries.

 [[Mean Reciprocal Rank]], in contrast to recall, is an *order-aware metric*, meaning that returning an actual relevant result at rank *one* scores better than returning one at rank *four.*
	- Another difference is that MRR is calculated based on *multiple queries* (think: mean).
![[Pasted image 20240614104621.png|300]]
Q is the number of queries, q is a specific query, and $rank_q$ is the rank of the first *actual relevant result* for query q.

Pros
- It's an *order-aware metric*, which is a massive advantage for use cases where the rank of the first relevant result is important, like chatbots or question answering.
Cons
- We retrieve the rank of the *first* relevant item, but not others. So if we retrieved 10 products for a user, and *only* the first item were always relevant, but no others, we would score a perfect MRR of 1.0, but do a pretty bad overall retrieval.
- MRR is slightly less readily interpretable compared to a simpler metric like recall@k, but it's still more interpretable than many other evaluation metrics.

![[Pasted image 20240426144735.png]]