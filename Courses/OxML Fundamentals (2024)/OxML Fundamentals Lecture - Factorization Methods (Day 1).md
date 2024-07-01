Professor Cho-Jui Hsieh, UCLA & Google
His work primarily focuses on enhancing robustness of ML systems.
This dude's accent is fucking terrible


![[Pasted image 20240519170728.png]]

2009 Netflix Prize: Netflix provided a dataset, asking people to solve the problem, which was how to predict whether a user likes a move, based on the rating history -- not only for one user, but for all users to all movies!

In this RecSyss problem, we're given a rating matrix
![[Pasted image 20240519171035.png]]
This shows some of the ratings that users have of various movies. Note that a given user has only watched some sparse subset of movies. Given this rating matrix which is partially observed, how can we predict the rest of the entries in this matrix (like in that red ❓ box)

![[Pasted image 20240519171311.png]]
We assume that this actual rating matrix comes from some low-rank W and H matrixes. We do our best to approximate our A matrix by using our lower-rank W and H matrices.
- We hope that if we find the best W and H, to approximate all of the observed entries in the current data!

![[Pasted image 20240519171421.png]]
At prediction time, when we want to do inference on that ❓ mark box, we take the inner product of these two red vectors to get the prediction.
- So for every entry in the ratings matrix, we estimate it by using the inner product of the appropriate vectors from W and H.

![[Pasted image 20240519171747.png]]`
This is the formalized way of writing it down. 
- A is our m x n matrix; imagine that we have m=billions of users and n=millions of movies.
- We observe some subset of observed entries in this matrix A; we use Omega to denote this set.
- We try to find the matrix W and H to minimize the prediction loss (w^Th is the prediction, A_i,j is the answer). See that we also have a L2 regularization to make sure that W and H are small; we want to regularize to avoid overfitting.
- So this is basically saying minmize the sum of squared errors of (y - yhat) + regularization.

So why does this work?
- We can imagine that each user has some underlying preference vector, and each movie has some underlying latent factors (say, one of the elements tries to capture whether a movie is action or romance, and another whether it's a comedy or horror movie.)
![[Pasted image 20240519171932.png]]


If A is fully observed, we can use [[Singular Value Decomposition]] (SVD):
![[Pasted image 20240519173307.png]]
In Linear algebra, SVD decomposes a matrix A into U times Sigma times V transpose.
- U is the left singular vectors
- Sigma is a diagonal matrix of singular values (first is largest, second is second largest, last is smallest singular value)
- V is the right singular vectors

In this case, we just need to take the first k singular vectors and the first k singular values to get the best rank-k approximation (we do this for both U and V), where k was that inner dimensionality of our approximation matrices W and H from the previous image.

But this isn't that useful, because this only works well for the fully-observed case, and in reality we really only have the partially-observed case (plus, why even do prediction if we have the full matrix? We already every user's rating of every movie!)

It turns out that there's no closed-form solution for the real partially-observed case, so we need to solve a non-convex optimization problem.
- If it were a convex problem, we could use gradient descent to find the optimal solution, but if it's non-convex, there's no guarantee that we find the global minimum!

----

The first algorithm people proposed for this problem is ALS: Alternating Least Squares

## Alternating Least Squares (ALS)

![[Pasted image 20240519173933.png]]

Here, we have two variables we want to solve for, W and H -- we don't know how to solve them jointly, so how about we fix one, update the other, swap (lock the other, optimize the other), fix one, update the other, ...

So in every iteration, we fix one, and update the other.
- When we fix W and update H, this subproblem is actually a convex problem!

![[Pasted image 20240519174137.png]]
This is easier to solve (believe me), and we can parallelize this operation too...

Using this idea we can iteratively update W and H (alternating) until we've done (eg) 100-200 iterations, and we get a reasonably good solution


## Stochastic Gradient Descent (SGD)

![[Pasted image 20240519174443.png]]
We try to decompose the objective function into a sum of individual objective functions, and then every time, we just pick one of the functions, compute a gradient of this function, and do an update.
- We try to decompose our objective into many small components. A natural way is to decompose based on the number of observations (related to Omega).


... Boring ...

![[Pasted image 20240519180000.png]]
Above: "I think everything in recommendations can be formulated this way; the question is what model you use for F and G."

![[Pasted image 20240519175704.png]]
- How to bootstrap recommendations when a new item is added to the system. This is the cold start problem in recommendation systems.
	- Encoders help; used to map text description of products (perhaps image) to latent features. This generalizes well to new products.

![[Pasted image 20240519175950.png]]
In our examples so far, we only have positive examples of times when someone chose to use a product and rate it, we don't have negative examples.


----
![[Pasted image 20240519180156.png]]
Now let's move onto the next part of the lecture and talk about how to use low-rank estimations generally in machine learning, even outside of a RecSys cost
- For compression
- For training

![[Pasted image 20240519180231.png]]
Model size is growing exponentially

We need model compression for:
- Running on smaller devices
	- Mobile devices
- More performant inference latency
	- Serving LLMs in real-time is difficult, when forward passes are extremely expensive


![[Pasted image 20240519180426.png]]
Above: Matrices are getting bigger! These need to be loaded into GPU memory to do inference/training.

![[Pasted image 20240519180608.png]]
- Sparse: Sparse networks; every matrix operation needs to then be a sparse matrix operation, which is less efficient than dense operation :( 
- Low rank: Decomposing weight matrices into low-rank approximations
- Quantization: Using a less granular representation (float32 -> 2 bits)

![[Pasted image 20240519180809.png]]
Above:
- We reduce our model size from m x n to mk + kn, where we can choose the inner dimension of our factor matrices.

![[Pasted image 20240519181213.png]]

![[Pasted image 20240519182005.png]]

![[Pasted image 20240519182228.png]]
((Note: It's now known that LoRAs learn less and forget less in finetuning, compared to full finetuning.))


![[Pasted image 20240519182259.png]]
Some people are even starting to try to use LoRA-based methods for memory-efficient pretraining?


![[Pasted image 20240519182403.png]]


