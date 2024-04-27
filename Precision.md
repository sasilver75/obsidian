---
aliases:
  - Positive Predictive Value
---
Precision can be seen as a measure of *quality*, and [[Recall]] as a measure of *quantity*.

Higher precision means that an algorithm returns more relevant results than irrelevant ones, and high recall means that an algorithm returns most of the relevant results (whether or not irrelevant ones are also returned).

==Precision = TruePositives / (TruePositives + False Positives)==
Of all the times you say 'yes,' how many times were *actually* 'yes'?
  
![{\displaystyle {\text{Precision}}={\frac {\text{Relevant retrieved instances}}{\text{All retrieved instances}}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/f2fe5aa3d0e91f91abc0ead472c59737af6c47c0)

 Remember: [[Precision]] is the number of true positive results divided by all samples predicted to be positive, including those not predicted correctly. "Of all the times you say 'yes', what percentage are you correct?"

![[Pasted image 20240420220007.png]]


---

# In information retrieval
The ==[[Return Set]]== Ret(D, K)  of a ranking at value K is the set of documents at or above K in D.

The ==[[Relevance Set]]== Rel(D, q) for a query, given a document ranking, is the set of all documents that are relevant to he query (anywhere in the ranking).

[[Precision]] in an IR context is thus:
![[Pasted image 20240426145216.png]]
If we think about the values >= K as the "guesses" we made, Precision says how many of those were "good" ones.

And [[Recall]] is the dual of that:
![[Pasted image 20240426145220.png]]
Out of all of the relevant documents, how many of these relevant document bubbled up to be > K, in the ranking.

![[Pasted image 20240426145606.png]]
