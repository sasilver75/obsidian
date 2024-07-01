[[K-Means Quantization]]
![[Pasted image 20240630191241.png]]
Say we have this matrix
![[Pasted image 20240630191251.png]]
We cluster the matrix using [[K-Means Clustering]], so that similar weights are near eachother.
![[Pasted image 20240630191313.png]]
Then we just use the centroid to represent each of them, and we just store the index in the former weight matrix, now just a cluster index. The centroids can be stored in whatever precision you'd like.

In terms of storage:
![[Pasted image 20240630191713.png]]
This is a toy example where the matrix is only 4x4; if the matrix got much larger, then the savings in the cluster index become much larger, and the cost of the codebook doesn't grow very much.
- Obviously there's a big question about how to choose your k value/size of the codebook.

How do we recover the accuracy loss when we train the model?
- We ge t the gradients
- We cluster the gradients in the same pattern as the weights, and we accumulate the same color gradients all together... sum them up or average them, and then we get the reduced gradient. We subtract that, times the learning rate, from the original centroid, to get our fine-tuned centroid. 

![[Pasted image 20240630191924.png]]
![[Pasted image 20240630200531.png]]

![[Pasted image 20240630192003.png]]
Pruning and quantization together can be combined together in a friendly manner to push the model to be much smaller, while maintaining accuracy.
