---
aliases:
  - MAP
---
 [[Mean Average Precision]] (MAP@K) is another popular *order-aware* metric. It seems to have an odd name -- the *mean* of an *average*?
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