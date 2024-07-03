https://www.pinecone.io/learn/offline-evaluation/

----

Evaluation measures for IR systems can be split into two categories?
1. ==Online Metrics==
	- Captured during actual usage of the IR system when it is online; consider user interactions like whether a user clicked on a recommended show from Netflix, or if a particular link was clicked from an email advertisement (CTR).
2. ==Offline Metrics==
	- Measured in an isolated environment before deploying a new IR system. These often look at whether a particular set of relevant results are returned when retrieving items with the system.

![[Pasted image 20240613194737.png|450]]

Organizations often use both online and offline metrics to measure performance of the system; usually offline metrics to predict system performance *before deployment*, and online metrics to measure the performance of live IR systems.

We'll focus on the most useful and popular offline metrics:
- [[Recall (Information Retrieval)]]@K
- [[Mean Reciprocal Rank]] (MRR)
- [[Mean Average Precision]]@K (MAP@K)
- [[Normalized Discounted Cumulative Gain]]@K (NDCG@K)

These metrics are deceptively simple yet provide invaluable insight into performance!

There are two more subdivisions for these metrics:
1. Order-Aware
2. Order-Unaware
If the order of results impacts the metric score, the metric is order-aware; else it is order-unaware

![[Pasted image 20240613195231.png|350]]
Above: Let's use this cat dataset for the rest of the article. See that the images show some combinations of cats, dogs, and boxes.

### Actual vs Predicted 
- **Actual condition**: Refers to the true label of every item in the dataset (these are positive if an item is relevant to query, or negative if the item is irrelevant)
- **Predicted condition:** The predicted label returned by our IR system; if an item is returned, it's predicted as being *positive* $\hat{p}$ , and if it's not returned, it's predicted as being *negative* $\hat{n}$ .

From these actual and predicted conditions, we can calculate all of our offline metrics!
- Actual relevant results being returned are True Positives ($p\hat{p}$)
- Actual irrelevant results being returned are False Positives ($n\hat{p}$)
- Actual irrelevant results not being returned are True Negatives ($p\hat{n}$)
- Actual irrelevant returns being returned are False Negatives ($n\hat{n}$)

## Recall@K
- [[Recall (Information Retrieval)]]@k is one of the most interpretable and popular offline evaluation metrics.
	- Measures how many relevant items were returned ($p\hat{p}$) against how many relevant items exist in the entire dataset ($p\hat{p} + p\hat{n}$)
![[Pasted image 20240614104135.png]]
The K in this and all other offline metrics refers to the number of items returned by the IR system.
![[Pasted image 20240614104220.png|300]]
Above: 1/(1+3) = 0.25
- With recall@k, score improves as K increases, and the scope of returned items increases.

Pros
- Easily on the most interpretable evaluation metrics; we know that a perfect 1.0 indicates that all relevant items are being returned, and we know that a smaller *k* value makes it harder to get higher recall@k values.
Cons
- By increasing K to N, or near N, we can get a perfect score every time, so relying solely on recall@k can be deceptive.
- It's an *order-unaware* metric; meaning that if we used recall@4 and returned one relevant result at rank *one*, we'd score that the same as if we returned one relevant result at rank *four* (which would be a worse retrieval).

## Mean Reciprocal Rank (MRR)
- [[Mean Reciprocal Rank]], in contrast to recall, is an *order-aware metric*, meaning that returning an actual relevant result at rank *one* scores better than returning one at rank *four.*
	- Another difference is that MRR is calculated based on *multiple queries* (think: mean).
![[Pasted image 20240614104621.png|300]]
Q is the number of queries, q is a specific query, and $rank_q$ is the rank of the first *actual relevant result* for query q.

Pros
- It's an *order-aware metric*, which is a massive advantage for use cases where the rank of the first relevant result is important, like chatbots or question answering.
Cons
- We retrieve the rank of the *first* relevant item, but not others. So if we retrieved 10 products for a user, and *only* the first item were always relevant, but no others, we would score a perfect MRR of 1.0, but do a pretty bad overall retrieval.
- MRR is slightly less readily interpretable compared to a simpler metric like recall@k, but it's still more interpretable than many other evaluation metrics.


## Mean Average Precision (MAP)
- [[Mean Average Precision]] (MAP@K) is another popular *order-aware* metric. It seems to have an odd name -- the *mean* of an *average*?
- To calculate MAP@K, we start with a metric called [[Precision (Information Retrieval)]]@k:
![[Pasted image 20240614105340.png|300]]
You might think this looks pretty similar to [[Recall (Information Retrieval)]]@k, and it is! The only difference is we've swapped $p\hat{n}$ to $n\hat{p}$ here. This means we now consider both actual relevant and non-relevant results *only* from the returned items. We don't consider non-returned items.
Now we can calculate *average precision@k* (AP@K) as:
![[Pasted image 20240614105614.png]]
Above: Note that $rel_k$ is a *relevance* parameter that (for AP@K) is equal to 1 when the kth item is relevant, or 0 when it is not. We calculate precision@k and rel_k iteratively using k=\[1, ..., K\].
![[Pasted image 20240614105811.png|300]]
Because we multiple precision@k and rel_k, we only consider precision@k where the kth item is relevant:
![[Pasted image 20240614110240.png|300]]
Given these values, we can calculate the AP@K_q score where K=8 as:
![[Pasted image 20240614110341.png|300]]
Each of these AP@K calculations produces a single average precision @ K score for each query. To get the *mean* average precision (MAP@K), we simply divide the sum by the number of queries Q:
![[Pasted image 20240614110424.png|300]]
Pros
- A simple offline metric that is *order-aware.* This is ideal for the cases where we expect to return multiple relevant items.
Con
- The rel_k relevance parameter is binary; we must either view items as *relevant* or *irrelevant.* It doesn't allow for items to be slightly more/less relevant than others.

## Normalized Discounted Cumulative Gain (NDCG)
- [[Normalized Discounted Cumulative Gain]] (NDCG@K) is another *order-aware* metric that we can derive from a few simpler metrics.
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













