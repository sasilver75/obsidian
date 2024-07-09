---
aliases:
  - Structured Sparsity
---
![[Pasted image 20240628125324.png]]
The benefit of this is that it can still be considered as a dense matrix, so we can use our usual hardware acceleration!
- But this is certainly a less flexible method of pruning -- we might be pruning "good" weights in our model too. 

![[Pasted image 20240628125524.png]]
So there's a very broad spectrum of pruning regularity for convolutional operations.