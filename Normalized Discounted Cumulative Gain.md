---
aliases:
  - NDCG
---
 [[Normalized Discounted Cumulative Gain]] (NDCG@K) is another *order-aware* metric that we can derive from a few simpler metrics.
- We start with Cumulative Gain (CG@K), which is calculate like so:
![[Pasted image 20240614110814.png|300]]
Annoyingly, here, rel_k is different (from its use in [[Mean Average Precision|MAP]]). It's a range of relevance ranks where 0 is the least relevant, and some higher value is the most relevant. The number of ranks doesn't matter; in our example we'll use a range of 0 -> 4.
![[Pasted image 20240614110929.png|300]]
![[Pasted image 20240614111014.png|300]]
Above: The circled numbers represent the *IR system's predict ranking*, and the diamond shapes represent the rel_k *actual ranking*.
To calculate the cumulative gain at position K (CG@K), we sum the relevance scores up to the predicted rank K. So when K=2:
![[Pasted image 20240614111116.png|300]]
It's important that CG@K is *not order aware!* If we swap images 1 and 2, we'll return the same score when K >= 2 despite having the more relevant item placed first.
Because of this lack of order awareness, we modify our [[Cumulative Gain]] metric to [[Discounted Cumulative Gain]] (DCG), adding a penalty in the form of $log_2(1+k)$ to the formula:
![[Pasted image 20240614111810.png|300]]
Now, when we calculate DCG@2 and if we were to swap the position/ordering of the first two images, we return different scores!
![[Pasted image 20240614111856.png|300]]

Unfortunately, while DCG@K scores are order-aware, they're very hard to interpret, as their range depends on teh variable rel_k range we choose for our data.
We use the [[Normalized Discounted Cumulative Gain]] (NDCG@K) metric to fix this!
- NDCG@K is a special modification of the standard NDCG that cuts off any results whose rank is greater than K. This modification is prevalent in use-cases measuring search performance.
	- NDCG@K normalizes DCG@K using the *ideal* DCG@K (IDCG@K) rankings.
	- For IDCG@K we assume the most relevant items are ranked highest, and in order of relevance; calculating IDCG takes nothing more than reordering the assigned ranks and performing the same DCG@K calculation.
![[Pasted image 20240614112148.png|300]]
Using NDCG, we get a more interpretable result, where we know that 1.0 is the *best* score that we can get with all items ranked perfectly (eg the IDCG@K).

Pros
- One of the most popular offline metrics for evaluating IR systems, in particular web search engines, because it optimizes for highly relevant documents, is order-aware, and is easily interpretable.
Cons
- There's a signifiant disadvantage for NDCG@K; not only do we need to know which items are relevant for a particular query, we need to know whether each item is more/less relevant than other items -- the data requirements are more complex.