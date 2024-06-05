
Link: https://weaviate.io/blog/retrieval-evaluation-metrics

---

How do you evaluate the quality of your search results in a [[Retrieval-Augmented Generation|RAG]] pipeline, or in a recommendation system?

### Metrics to evaluate information retrieval systems

We'll explore [[Precision (Information Retrieval)]]@k, [[Recall (Information Retrieval)]]@k, [[Mean Average Precision]]@k, [[Mean Reciprocal Rank]]@K, and [[Normalized Discounted Cumulative Gain]]@k.

Note that all of these metrics end in @K, meaning that we've only evaluating these metrics over the top K results.

All of the following metrics take values between 0 and 1, where higher values mean better performance.

Additionally, they can be categorized into:
- ==Not Rank-Aware== (Only reflect the NUMBER of relevant items in the top K results)
	- Precision
	- Recall
- ==Rank-Aware== (Consider the NUMBER of relevant items and the POSITION in the list of results)
	- [[Mean Average Precision|MAP]]
	- [[Mean Reciprocal Rank|MRR]]
	- [[Normalized Discounted Cumulative Gain|NDCG]]

---

## Precision
- The [[Precision (Information Retrieval)]]@K metric measures how many of the retrieved items are relevant (as a percentage of all retrieved items). This metric is not rank-aware, thus only considering the NUMBER of relevant results, but NOT their order.

![[Pasted image 20240604221629.png]]
Above: "What percentage of the top-k items are actually relevant?"
- It seems that this will be capped at 1, assuming that there are more relevant items in the dataset than K. (ie that you can fully stock the top K with relevant items)

## Recall
The [[Recall (Information Retrieval)]]@K metric measures how many relevant items were successfully retrieved from the entire dataset (as a percentage of all relevant items in the *dataset*). 

![[Pasted image 20240604221737.png]]
Above: "What percentage of all relevant items in the dataset made it into the top K?"
- It seems to me that this is going to be "capped" at K/#Relevant items; i.e. if you're only retrieving K=3, but there are 10 relevant items in the dataset, then your recall is going to be capped at .3.

## Mean Reciprocal Rank (MRR)
- The [[Mean Reciprocal Rank]]@K metric measures how well the system finds a relevant result as the *top result*. This metric only considers the ORDER of the first relevant result, but not the NUMBER or ORDER of the OTHER relevant results!
- You can calculate by averaging the [[Reciprocal Rank]] (RR) across multiple queries (in IR) or users (in RecSys)
![[Pasted image 20240604222410.png]]
![[Pasted image 20240604222430.png]]


## Mean Average Precision (MAP)
- [[Mean Average Precision]]@K measures the system's ability to return relevant items in the top K results while placing more relevant items at the top.
- You can calculate MAP@K by averaging the average position at K across multiple queries (in IR) or users (RecSys):
![[Pasted image 20240604222534.png]]
![[Pasted image 20240604222546.png]]
MAP is also the default metric used in [[MTEB]], the Massive Text Embedding Benchmark leaderboard for the ==Reranking== category.

## Normalized Discounted Cumulative Gain (NDCG)
- [[Normalized Discounted Cumulative Gain|NDCG]]@K measures the system's ability to sort items based on relevance. In contrast to the above metrics, this metric requires not only IF a document is relevant, but also HOW it is relevant (relevant, somewhat relevant, not relevant).
- You can calculate the NDCG@K by calculating the Discounted Cumulative Gain, then normalizing it over the *ideal* discounted cumulative gain (IDCG).
![[Pasted image 20240604222756.png]]
![[Pasted image 20240604222803.png]]
NDCG@K is the default metric used in the [[MTEB]] leaderboard for the ==Retrieval== category.
(Note that I'm missing the formula of IDCG here)

