Average Precision is a less-sensitive metric to values of K than [[Precision]] is.

Average precision for a query, relative to a document ranking
![[Pasted image 20240426145734.png]]
For the numerator, we get precision values for every step where there *is* a relevant document (every place where there is a star; we sum those up). The denominator is the set of relevant documents.
It's more clear with a picture:
![[Pasted image 20240426145858.png]]
See: D1 is the clear winner 