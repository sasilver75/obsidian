Link: https://blog.reachsumit.com/posts/2023/03/pairing-for-representation/

---

## Encoders for Learning Representations
- IR retrieval problems usually take a query as input and return the most relevant documents from a large corpus. 
- These systems are often implemented in a ==cascade== fashion, where less-powerful but more-efficient =="retriever"== algorithms (Elastic, [[BM25]], [[Bi-Encoder]]) first *reduces the search space* for candidates, and then a more complex but powerful =="reranker"== algorithm(s) (like a [[Cross-Encoder]]) reranks the retrieved documents.
	- ==The retriever is focused on optimizing for high recall, and the reranker focuses more on precision.==

![[Pasted image 20240527220213.png]]

- BM25 is a widely-used approach that is based on token matching and [[TF-IDF]] weights. But it lacks semantic context and can't be optimized for a specific task.
- In contrast, Dual Encoders ([[Bi-Encoder]]s) are embedding-based methods that embed queries and documents in the same space, and use a similarity measure like inner product to measure the similarity between learned representations for queries and documents.
- Apart from just using the textual data for query and document towers, we can also add extra contextual information as input: Facebook uses representations of text inputs, but also location signals like city, region, country, and language for queries, and tagged locations for documents.
![[Pasted image 20240527220629.png]]
Above: Bi-Encoder

## Need for Effective Sampling
- Sampling instances for training such IR systems is not straightforward.
- It's infeasible to annotate all candidates for a given query; Annotators are usually given top-k candidates from a simplistic approach like BM25, so it's very likely to have a lot of unlabeled positive data.
- An approach to improve effectiveness of dual encoders is to also use [[Negative Pair]]s to further emphasize the notion of similarity. It helps in separating the irrelevant pairs of queries and documents, while keeping distance smaller for [[Positive Pair]]s.

## Calculating Contrastive Loss
- Such models fall under the self-supervised learning category and are optimized by a ==contrastive objective in which  a model is trained to maximize the scores of positive pairs and minimize the scores of negative pairs.==
	- Contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning approaches.

Given a query $q$ with positive document $d_+$ and a *pool of* negative documents $k_i$, with $\tau$  as a temperature parameter. The contrastive [[InfoNCE]] loss is defined as 
![[Pasted image 20240527221526.png]]
- This encourages positive pairs to have high scores, and negative pairs to have low scores.

For a given triplet ($q^{(i)}, d^{(i)}_+, d^{(i)}_-$) , we can define the contrastive triplet loss as:
![[Pasted image 20240527221802.png]]
Where $D(u,v)$ is a distance metric, and $m$ is the ***margin*** enforced between positive and negative pairs. More on Contrastive Loss Functions [here](https://lilianweng.github.io/posts/2021-05-31-contrastive/#contrastive-training-objectives)


## How to Sample Negative Examples
- For retrieval problems, positive examples are often explicitly available, while negative examples are to be sampled from a large corpus based on some strategy. How to sample negative documents is a key challenge during learning.
	- The retriever has to look through a large corpus to find appropriate negative samples.
	- The reranker at a later stage can simply select irrelevant examples from the previous retrieval stage to be the negative samples.

### Explicit Negatives
- Some tasks naturally allow for a way to pick negative samples. For Facebook Search, non-clicked impressions work.

### Random Negatives
- A simple approach to selecting negative examples is to randomly sample documents from the corpus for each query. 
- It's found that the model trained on non-click impressions (above) as negatives have significantly *worse* model recall, compared to using negative samples! 🤔

### BM25 Negatives
- ==Another way to sample negative documents is to take the top documents returned by BM25 which *do not* contain the answer, but *still match* the input query tokens!==
	- ((Intuitively, these seem like they should be ~hard negatives, right?))
- BM25 biases the dense retriever to *not* retrieve documents with much query term overlapping, which is a distinct characteristic of these negative examples. Such behavior leads to optimization bias, which *harms* retrieval performance! 🥲

### Gold Negatives
- For a given query $q$, all documents that are paired with *other queries* in the labeled data can be considered as negative documents for the query $q$.
- Karupukhin et al. used this approach, but didn't find significant gains over choosing random negatives.
	- ((Wait, why does this not work well, but in-batch seems to?))

### In-batch Negatives
- A more effective approach to picking gold negatives is to select gold documents of other queries in the *same batch*.
	- So for a batch size B, each query can have up to B-1 negative documents.
	- ==This is one of the most common approaches used to sample negatives for training dual encoders.==
		- ((Wait, how is this at all different from the previous method?))
- ==Karpukhin et al. found that the choice of in-batch negatives (random, BM25, or gold) doesn't impact the top-k accuracy much when k >= 20.==
	- ==They also showed that while in-batch negatives are effective at learning representation, they are not always better than sparse methods like BM25 applied over the whole corpus==
	- ((Wait, now I'm confused; which one works best?))

### Cross-Batch Negatives
- Qu et al. proposed a cross-batch negative sampling method in a multi-GPU environment.
- By using their approach, for a given query $q$, we sample in-batch negatives on the same GPU along with sampling negatives from all other GPUs. ![[Pasted image 20240527223725.png]]

### Approximate Nearest Neighbors
- Another costly but well-performing sampling strategy was proposed by Xiong et al. who used asynchronous ANN as negatives.
- After every few thousand steps of training, they used the current model to re-encode and re-index the documents in their corpus. Then they retrieved the top-k documents for the training queries and used them as negatives for the following training iterations until the next refresh.
	- In other words, they picked negatives based on outdated model parameters.
- The near-SoTA performance of their models comes at significantly higher training time. To tackle this, authors implement async index refreshes that update the aNN index once every few batches.

### Hybrid
- A lot of research work has chosen to combine some of the methods described above. For example, a combination of gold passages from the same mini-batch and one BM25 negative passage, or a combination of BM25 and random sampling.


## How to Sample Positive Examples

==Supervised==
### Explicit Positives (Supervised)
- Under supervised settings, positive instances are explicitly available. These labels are usually task-specific and can be sampled intuitively from user activity logs.

==Self-Supervised/Unsupervised Setting==
### Inverse Cloze Task
- In the standard Cloze task, the goal is to predict masked text given the context. The [[Inverse Cloze|Inverse Cloze Task]] (ICT) requires predicting the *context*, given a sentence.
- It generates two mutually exclusive views of a document, first by randomly sampling a span of tokens from a segment of text while using the complement of the span as the second view. The two views are then used as a positive pair.

### Recurring Span Retrieval
- In Spider (Span-based unsupervised dense retriever), authors proposed a self-supervised method called recurring span retrieval for training unsupervised dense retrievers.
- Leverages recurring spans in different passages of the same document to create positive pairs.

### Independent Cropping
- Under this strategy, two contiguous spans of tokens from the same text are considered positive pairs.

### Simple Text Augmentations
- Simple data augmentations like random word deletion, replacement, or masking can be used to make a positive pair.

### Others
- Other approaches for creating positive pairs include using masked salient spans as in [[REALM]], random cropping as in [[Contriever]], neighboring text pieces in CPT, and query and top-k BM25 passages.

---

## Hard Example Mining
- Modern information retrieval tasks work with large and diverse datasets that usually contain an overwhelming number of easy examples and a small number of hard examples.
- Identifying and utilizing these hard examples can make the model training process more efficient and effective.
- Hard example mining/[[Hard Negative Mining]] is the process of selecting hard examples to train machine learning models.
	- The key idea is to select instances on which our model triggers a false alarm.
- We adopt an iterative process that alternates between training a model using selected instances (that may include a random set of negative examples) and then selecting new instances by removing the "Easy" ones. We then iterate 1+ times.

### Hard Negative Mining
- The training and inference processes for dual-encoder-based retrievers are usually inconsistent: During training, the retriever is given only a small candidate set for each question (usually 1 positive pair per question) to calculate the question-document similarity, while during inference it's required to identify a positive document for a given question out of millions of candidates.
- There might also be a large number of unlabeled positives that might be picked as negative examples, leading to an increase in false positives by the model; Using reliable, hard negatives helps in alleviating such issues.

### Top-ranked Negatives
- A straightforward approach to picking hard negatives is to use trained retrievers to make inferences on ***negative samples***, and select the top-k ranked documents as negatives.
- Can still suffer from false positives, as a lot of negative samples could actually be unlabeled positive samples.

### In-batch Hard Negatives
- In FB Search, the authors used in-batch negatives to first get a set of candidate documents. Next, then used an ANN approach on the query and this set of negatives to find the negatives closest to the input query. This negative was then used as the hard negative.
- Authors found that the optimal setting is at most two hard negatives per positive. Using more than two hard negatives had an adverse impact on model quality.
### Denoised Hard Negatives
- To solve false negatives issue, RocketQA proposed to utilize a well-trained cross-encoder to remove false negatives from top-k ranked documents.
- Cross-encoders are inefficient to be used for inference in real time, but are also highly-effective and robust, so it can be used on the top-k ranked documents from a retriever, only keeping the documents that are predicted to be negatives with high confidence.
	- The final denoised set of negatives is more reliable and can be used as hard negatives.

### ANCE: ANN Negative Contrastive Learning
- We saw earlier how ANN negative contrastive learning ([[Approximate Nearest Neighbor Contrastive Learning|ANCE]]) can be used to sample negatives.
- The main drawback of this approach was that the inference is too expensive to compute per-batch, as it requires a forward pass on the entire corpus, which is much bigger than the training batch.

### Hard Positive Mining
- While the majority of the literature focuses on bootstrapping, some works also suggest mining hard positives!

## Other Approaches

### Data Augmentation
- Is another approach commonly used to create additional positive and negative pairs for training.
- RocketQA trains a cross-encoder and uses it on unlabeled data to make predictions, and then uses positive/negative documents with high confidence scores to augment to training. They select the top-retrieved documents with scores < 0.1 as negative, and those with score > 0.9 as positive.
	- ((I don't see how this is augmented data?))
- In AugTriever, authors propose multiple unsupervised ways to extract salient spans from text documents to create pseudo-queries that can be used to create positive and negative pairs.
	- For structured documents, they recommend utilizing structural heuristics (like title and anchor texts).
	- For unstructured ones, they use approaches like BM25, dual encoder, and pre-trained LMs to measure the salience between a document and various spans of text extracted from it.

## Misc
- In Learning to Retrieve (LTRe), authors use a novel approach that doesn't require ANY negative sampling
	- A pretrained document encoder represents documents as embeddings, which are fixed throughout the process.
	- At each training step, a dense retrieval model output a batch of query representations. LTRe then uses them and performs full retrieval.
	- Based on retrieval results, it updates the model parameters such that queries are mapped close to the relevant documents and far from the irrelevant ones.
```
To solve this problem, we propose a Learning To Retrieve (LTRe) training technique. LTRe constructs the document index beforehand. At each training iteration, it performs full retrieval without negative sampling and then updates the query representation model parameters. Through this process, it teaches the DR model how to retrieve relevant documents from the entire corpus instead of how to rerank a potentially biased sample of documents.
```


# Summary
- We looked at a wide variety of methods to create positive and negative samples for representation learning. We learned that both easy and hard examples are important for training a retriever model.
- ==Models optimized for recall can benefit from random negative sampling, while models focused more on precision benefit from hard negatives.==