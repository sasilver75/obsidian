#article 
Link: https://huggingface.co/blog/matryoshka
Writers:
- Tom Aarsen
- Joshua
- Omar Sanseviero

----

[[Matryoshka Representation Learning]]

# Understanding Embeddings

- ==An [[Embedding]] is a *numerical representation of a more complex object*== (text, images, audio, etc.)

![[Pasted image 20240413151656.png]]
Above: You can represent many things as dense vectors, as long as you have a useful objective function when you learn the embedding!

Embedding models always produce embeddings of the same fixed size -- ==you can then compute the similarity of complex objects by computing the similarity of respective embeddings!==

![[Pasted image 20240413151823.png]]

There are huge number of uses for embedding models -- they're the backbone for so many things:
1. [[Recommendation Systems]]
2. [[Information Retrieval]]
3. One-Shot or Few-Shot learning
4. Outlier detection
5. Similarity search
6. Paraphrase detection
7. Clustering
8. Classification
9. ...


# Matryoshka Embeddings 🪆

New SoTA embedding models started producing embeddings with *increasingly higher output dimensions!*
- ==Adding more dimensionality to vector representations improves performance at the cost of downstream tasks such as search or classification.==

In 2022, the authors (Kusupati et al) were ==inspired to create embedding models whose embeddings could be reasonably shrunk without suffering too much on performance!==


![[Pasted image 20240413152222.png]]

These Matryoshka embedding models are trained such that these smaller *truncated* embeddings would still be useful!
- In short, Matryoshka embedding models can produce useful embeddings of various dimensions.


# Matryoshka Dolls
- For those unfamiliar, Matryoshka dolls, also called "==Russian Nesting Dolls==" are a set of wooden dolls of decreasing size that are placed inside one another.
- In a similar way, ==Matryoshka embedding models aim to store more important information in earlier dimensions, and less important information in later dimensions.==
	- ==This lets us truncate the original (large) embedding produced by the model, while retaining enough of the information to perform well on downstream tasks!==
	- ((I wonder how they determine which are the most important dimensions?))

![[matryoshka-small.gif]]

1. Compute Matryoshka Embedding of an input sentence: "The weather is nice" (eg dim=1024)
2. Truncate your embedding (creating many different vector representations?)
3. Select Dimensionality (eg dim=256)



# Why would you use Matryoshka Embedding models?
- Having variable-size embedding models can be quite valuable to practitioners!

1. Shortlisting and Reranking
	- Rather than performing your downstream task (e.g. ==nearest neighbor search==) on the *full* embeddings, you can shrink ==the embeddings to a smaller size==.
		- ==Afterwards==, you can then ==process the remaining (retrieved) embeddings using their full dimensionality==!
2. Trade-offs
	- Matryoshka models will ==allow you to scale your embedding solutions to your desired storage cost, processing speed, and performance.==


# How are Matryoshka Embedding models trained?

### Theoretically
- The Matryoshka Representation Learning ([[Matryoshka Representation Learning|MRL]]) ==approach can be adopted for almost all embedding model training frameworks==.

Normally, a training step for an embedding model involves producing embeddings for you training batch (of texts, for example), and then using some loss function to create a loss value representing the quality of the produced embeddings. The optimizer will adjust the model weights throughout training to reduce the loss value.

==For Matryoshka Embedding models during training==, a training step *also* involves producing embeddings for your training batch, but then ==you use some loss function to determine *not just the quality of your full-size embeddings*, but *also the quality of your embeddings at various different dimensionalities!*==
- For example, your dimensionalities are 768, 512, 256, 128, 64 -- ==the loss values for each dimensionality are added together, resulting in a final loss value.==
- In practice, this ==incentivizes the model to frontload the most important information at the start of an embedding==, such that it will be retained if the embedding is truncated.

# Use in Sentence Transformers 🤖

- `SentenceTransformers` is a commonly-used framework to train embedding models, and it recently implemented support for Matryoshka models.
	- The only difference: rather than applying some loss function on only the full-size embeddings, we also apply that same loss function on truncated portions of the embedding.

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CoSENTLoss, MatryoshkaLoss

model = SentenceTransformer("microsoft/mpnet-base")

base_loss = CoSENTLoss(model=model)
loss = MatryoshkaLoss(  # Note the use of a special MRL loss function
    model=model,
    loss=base_loss,  # Which sort of wraps the base loss function, it seems?
    matryoshka_dims=[768, 512, 256, 128, 64], 
    matryoshka_weight=[1, 1, 1, 1, 1],  # Cool that you can weight them differently!
)

# Note above that even though our loss is only being evaluated on these 5 dimensions, I'm guessing that you can later prune down to (eg) 200 dimensions, though that wasn't one of the ones that hte loss was calculated on? If that's true, how should you choose these numbers?

model.fit(
    train_objectives=[(train_dataset, loss)],
    ...,
)
```


# How do I *use* Matryoshka Embedding models?

In practice, getting embeddings from a Matryoshka embedding model works the same way as with a normal embedding model.
- The only difference is that, after retrieving the embeddings, we can optionally truncate them to a smaller dimensionality.

After truncating, we can either directly apply them for our use case, or store them such that they can be used later! After all -- smaller embeddings in your vector database should result in considerable speedups! 😄

## In SentenceTransformers

In the `SentenceTransformer` library, you can load a Matryoshka Embedding model like normal, and run inference with it using the `SentenceTransformers.encode` function. 

After you get embeddings, we can then truncate them to our desired size, and normalize them if we want
- ((I don't know yet why we would want to normalize our embeddings 🤔))

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka") # Get our embedding model

matryoshka_dim = 64
embeddings = model.encode(
    [
        "The weather is so nice!",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
)  # Generate 3 Matryoshka Embeddings using our sequences and embedding model

# Shrink each of the embeddings down to the matryoshka_dim dimension
# Interesting use of the elispsis (...) operator here....
embeddings = embeddings[..., :matryoshka_dim]  # Shrink the embedding dimensions
print(embeddings.shape)
# => (3, 64)

# Now we can use our reduced-dimesionality embeddings to do downstream tasks, like determining cosine similarity.
# Similarity of the first sentence to the other two:
similarities = cos_sim(embeddings[0], embeddings[1:])
print(similarities)
# => tensor([[0.8910, 0.1337]])
```

# Results
- Now that Matryoshka models have been introduced, let's look at the actual performance that we may be able to expect from a Matryoshka embedding model versus a regular embedding model.

![[Pasted image 20240413160922.png]]
Above: We can see that the Matryoshka model reaches a higher Spearman similarity than the standard model at all dimensionalities, indicative that the Matryoshka model is superior in this task.

The performance of the Matryoshka model falls off much less quickly than the standard model -- this is shown clearly in the second figure, which shows the performance at the embedding dimension relative to the maximum performance.
- ==Even at only 8.3% of the embedding size, the Matryoshka model preserves 98.37% of the performance, much higher than the 96.46% by the standard model==.

These findings are indicative that truncating embeddings by a Matryoshka model could:
1. Significantly speed up downstream tasks such as retrieval
2. Significantly save on storage space, all without a notable hit.







