The ==[[Return Set]]== Ret(D, K)  of a ranking at value K is the set of documents at or above K in D.

The ==[[Relevance Set]]== Rel(D, q) for a query, given a document ranking, is the set of all documents that are relevant to he query (anywhere in the ranking).

[[Precision]] in an IR context is thus:
![[Pasted image 20240426145216.png]]
If we think about the values >= K as the "guesses" we made, Precision says how many of those were "good" ones.

And [[Recall]] is the dual of that:
![[Pasted image 20240426145220.png]]
Out of all of the relevant documents, how many of these relevant document bubbled up to be > K, in the ranking.

![[Pasted image 20240426145607.png]]

 [[Recall (Information Retrieval)]]@k is one of the most interpretable and popular offline evaluation metrics.
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