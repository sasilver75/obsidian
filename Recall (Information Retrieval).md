The ==[[Return Set]]== Ret(D, K)  of a ranking at value K is the set of documents at or above K in D.

The ==[[Relevance Set]]== Rel(D, q) for a query, given a document ranking, is the set of all documents that are relevant to he query (anywhere in the ranking).

[[Precision]] in an IR context is thus:
![[Pasted image 20240426145216.png]]
If we think about the values >= K as the "guesses" we made, Precision says how many of those were "good" ones.

And [[Recall]] is the dual of that:
![[Pasted image 20240426145220.png]]
Out of all of the relevant documents, how many of these relevant document bubbled up to be > K, in the ranking.

![[Pasted image 20240426145607.png]]
