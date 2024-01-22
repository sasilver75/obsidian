
Takes a collection of input vectors and applies three different linear transformations, to create query, key, and value vectors for each of the input vectors. These three matrices are initialized randomly, and transform the input in different ways.

![[Pasted image 20240122093755.png]]

Once we have the query, key, and value vectors, we calculate the attention from a given input vectors to the other input vectors.
We compute the scalar product (dot product) between the query vector of the token of interest, and the key vectors of every token in the input sequence. Then we divide by the square root of the dimension of the key vectors (above, sqrt(3)). Then we apply the softmax over all these values; we interpret these softmax scores as an "importance score" of how important each token in the collection of input vectors is for the vector in question. Now, to get the final representation of the input token that we're considering, we take weighted sum over all of the *value* vectors, weighted/multiplied by teh softmax result. So ==attention combines the representation of input vector's value vectors, weighted by the importance score (computed by the query and key vectors).==

![[Pasted image 20240122094148.png]]
One single "round" of attention isn't enough to capture the relational semantics of our data! This is why we have [[Multi-Head Attention]], where we have N sets of Q,K,V matrices. Each is ideally going to detect a different pattern -- this is a hyperparameter, and is an expensive one to increase, because ==attention scales quadratically in time and memory==. This is an active area of research (to either improve it or replace it with other operations that do the job of mixing information between tokens ("token mixing procedures").

![[Pasted image 20240122100003.png]]

Recap: We have our input embeddings. They go through the self-attention layer that gives us represeneations that are informed on teh fellow embeddings in the sequence. Then they go through the MLP layer. If we were to reorder our input tokens, the output would be the same, because all of the operations in the self-attention layer are commutative. But order matters, in langauge! So we need

This is what [[Positional Embeddings]] do; We add them to our input vectors. (More [here\(https://www.youtube.com/playlist?list=PLpZBeKTZRGPOQtbCIES_0hAvwukcs-y-x))
We can either learn these vectors or use simple rules.

![[Pasted image 20240122100639.png]]

Good explanation here: https://www.youtube.com/watch?v=ec9IQMiJBhs




Question:
- Why does attention necessarily require Q,K,V to get the contextual semantic meaning of 

Types:
[[Self-Attention]]
[[Cross-Attention]]
[[Masked Attention]]
[[Multi-Head Attention]]

Variants:
[[Flash Attention]]