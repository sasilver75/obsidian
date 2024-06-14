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


## Mean Reciprocal Rank (MRR)


## Mean Average Precision (MAP)


## Normalized Discounted Cumulative Gain (NDCG)










