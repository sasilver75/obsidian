#article 
Link: https://cameronrwolfe.substack.com/p/the-basics-of-ai-powered-vector-search

![[Pasted image 20240222211606.png]]

The recent generative AI boom has led many to wonder about the evolution of search engines
- Will dialogue-based LLMs replace traditional search engines, or will their tendency to hallucinate make them to untrustworthy?
- How can search technology be used in cooperation with "AI"?

The quick adoption of AI-centric search systems like you.com and perplexity.ai indicates a widespread interest in *augmenting* search engines with modern advancements in language models.


# Basic Components of a Search Engine

![[Pasted image 20240222212412.png]]
((Above: What I'm curious to learn about is why Retrieval and Ranking need to be different steps. I know that we do things like "fuzzy kNN" during the actual retrieval... so we don't get them returned in any specific order, and we need to rank/rerank them in-memory?))

Most search engines have two basic components:
- ==Retrieval==
	- From the set of all possible documents, identify a much smaller set of candidate documents that *might* be relevant to the user's query.
- ==Ranking==
	- Use more fine-grained analysis to order the set of candidate documents such that the most relevant documents are shown first.

Depending on our use case, the total number of documents that we're searching over might be very large (eg all products on Amazon, all web pages on Google). 
- As such, the retrieval component of search must be efficient -- it quickly identifies a small subset of documents that are relevant to a user's query.
- Once we've identified a smaller set of candidate documents, we can use more complex techniques (like a larger neural network) to optimally order the candidate set in a manner that's personalized and relevant to the user's query.

More details
- The intuitive idea behind search is straightforward, but many different approaches exist for retrieval and ranking, ranging from very traditional approaches that only rely on basic ML, while others leverage modern LLMs.

![[Pasted image 20240222212839.png]]


# Lexical Search

- The ==traditional approach to building a search engine is based on *matching words in a user's query to words in a document!*==
- This approach is called ==[[Sparse Retrieval]], or lexical retrieval==, and relies on a data structured called an [[Inverted Index]] to perform efficient keyword matching.
	- An inverted index is just a lookup that contains a mapping of words to documents (or locations in documents) it appears in.
	- Using this data structure, we can efficiently match terms to documents in which they appear, and even count the frequency of terms in each document.

Sparse Retrieval
- To understand why this process is called sparse retrieval, we first need to understand how we represent our data.
- Lexical search algorithms are based on *word frequencies*. If a word appears frequently in both a user's query and particular document, then this document might be a good match!

![[Pasted image 20240222214844.png]]

We first define a fixed-size vocabulary of relevant words. This doesn't necessarily need to be every word that's present in all documents.

Then, we represent a sequence of text (Either a query or a document) with a ***vector*** that contains an entry for each word in our vocabulary.
- Then, each of these entries can be filled with a number (possibly zero) corresponding to the frequency of that word within the text.

==Because our vocabulary is large, and only a (relatively) small number of words occur in any given document, this vector representation is relatively *sparse.*==
- This method of producing vector representations is called the ==Bag of Words Model==

![[Pasted image 20240222215130.png]]

Given a bag-of-words representation of a query, and a set of documents (also represented by vectors), the primary algorithm used for performing *sparse retrieval* is the [[BM25]] ranking algorithm.

==BM25== scores are entirely count based. We score documents by:
1. Counting words that co-occur between the query and the document
2. Normalizing these counts by metrics like *inverse document frequency* (which assigns higher scores to words that occur less frequently... meaning words common words like "and", that will have high commonality counts, don't dominate...and words like *Abracadabra* are important matches, even when it only happens a few times).

Although other sparse retrieval algorithms exist (eg [[TF-IDF]]), but BM25 achieves impressive performance.
In fact, ==BM25 is a hard baseline to beat== for even more complex sparse retrieval techniques and even modern approaches that use deep learning! BM25 remains the core of many search engines even today.

Practical Implementation
- BM25 is supported by major search tools like Elastic and RediSearch!
- To use these tools, the implementation of lexical search is both efficient and simple! We just:
	1. Store out documents in a database
	2. Define a schema for building the inverted index.
- Then, the inverted index is built asynchronously over the documents in our database, and we can perform searches with BM25 via an abstracted query language.


# Adding "AI" into a Search Engine
- BM25 *is* a machine learning algorithm, with tunable hyperparameters. We can improve its performance by leveraging common data science techniques like stemming, lemmatization, stop word removal, and more.
	- ==Stemming==: Eliminating prefixes and suffixes from words, and transforming them into their fundamental or root form. The objective is to streamline and standardize words. "Retrieval" - > "retrieve"
	- ==Lemmatization==: Grouping together the inflected forms of a word so that they can be analyzed as a single item, identified by the world's *lemma*, or dictionary form. "better" -> "good", "walking" -> "walk". Lemmas can be contextual ("meet" can be a noun or verb).
	- ==Stop Word Removal==: Removal of words like "the", "and", "is", etc.

How can we create a smarter search algorithm?
- The short answer is to use deep learning to improve both the retrieval and ranking process!
- There are two types of models in particular that we can use for this purpose:
	- [[Bi-Encoder]]s and [[Cross-Encoder]]s, which are both typically implemented using [[Encoder-Only Architecture]] [[Bidirectional Encoder Representations from Transformers|BERT]] models!

==Bi-Encoders==
- Form the basis of ==*dense retrieval*== algorithms.
- Take a sequence of text as input, and produce a dense vector as output.
- However the vectors produced by bi-encoders are semantically meaningful -- similar sequences of text produce vectors which are nearby in the vector space when processed by the bi-encoder!
	- As a result, we can match queries to documents by embedding both with a bi-encoder, and using vector search to find documents with high [[Cosine Similarity]] relative to the query.

![[Pasted image 20240222221043.png]]

Using algorithms like [[Hierarchical Navigable Small Worlds]] (HNSW), we can perform *approximate nearest neighbor* vector searches efficiently!
- Similar to performing lexical search, we can store document vectors within a database like Elastic or Redisearch and build an HNSW index on top of it. 
	- Then, we can perform vector search by:
		1. Producing an embedding vector of our query string
		2. Performing a vector search to find the most similar documents

![[Pasted image 20240222221313.png]]

==Cross-Encoders==
- Similar to bi-encoders in that they allow us to score the similarity between two sequences of text.
- Instead of separately creating a vector for each textual sequence, cross-encoders ingest *both* textual sequences using the same model... the model is then trained to predict an accurate similarity score for those textual sequences.
	- ==These models can more accurately predict textual similarity relative to bi-encoders, but searching for similar documents with a cross-encoder is *much* more computationally expensive!==

Namely, bi-encoders can be combined with vector search to efficiently discover similar documents, but cross-encoders require each textual pair to be passed through the model to receive a score.d
- ==Given this cost, cross-encoders are typically only applied at the *ranking* stage of search!== After we've retrieved a set of relevant candidate documents (using eg sparse retrieval BM25 or a dense-retrieval Bi-Encoder+HNSW strategy), we can pass these documents through a *cross-encoder* for a more accurate re-ranking.


# A simple framework for AI-powered search

As mentioned, search systems have two basic components: *==retrieval==* and *==ranking==*.

To create a basic *AI-powered search system*, the *retrieval* process could use *both*:
1. Lexical Retrieval / Sparse Retrieval using BM25
2. Dense Retrieval with a Bi-Encoder

We can then *combine* the results of these two retrieval algorithms by taking a *weighted sum* of each document's score from BM25 and vector search -- this combination of lexical and vector retrieval is referred to as [[Hybrid Search]]!

![[Pasted image 20240223103841.png]]

From the retrieval process, we receive a collection of query-related candidate documents. We choose to then more finely *rank* these documents using a cross-encoder that more accurately ranks the search results based on textual relevance to the query.

# Using BERT for Search
- ==Most of the commonly-used bi-encoder *and* cross-encoders are based on [[Bidirectional Encoder Representations from Transformers|BERT]]==
- As such, understanding the encoder-only architecture and self-supervised training strategy of BERT is important!

#### Encoder-Only Architecture
- Although the original transformer architecture contains both an encoder and decoder, BERT leverages an encoder-only architecture!
- The encoder-only architecture contains several repeated layers of *bidirectional* self-attention and feed-forward transformations, followed by a residual connection and layer normalization.

![[Pasted image 20240223104150.png]]

#### Crafting the Input
- The first step in understanding the encoder-only architecture is to understand how we construct the input.
- The input to our encoder-only transformer is just a sequence of text.
	- We use a *==tokenizer==* -- usually [[Byte-Pair Encoding]] (BPE) or a [[SentencePiece]] tokenizer to break this text into a sequence of tokens (i.e. words and subwords).
	- There are a few special tokens that are added to BERT's input, including:
		- ==[CLS]==: A token placed at the beginning of the sequence, and serves as a representation of the entire sequence.
		- ==[SEP]==: A token placed between multiple sentences that are fed as input to BERT, and serves as the separator between the sentences.
		- ==[EOS]==: An "end of *sequence*" token that is placed at the end of BERT's input to indicate that the sequence is over.

Once we've turned our sequence of words into a sequence of tokens...we can then ==embed== each of these tokens by performing a lookup within a large embedding layer, which forms a sequence of token vectors that are fed into the model as input!

![[Pasted image 20240223104535.png]]

Prior to being ingested into the model, however, each of these tokens must have a ==positional embedding== added to it, which allows the model to understand the position of each token in the underlying sequence.

#### Bidirectional Self-Attention
- At a high-level, ==the purpose of self-attention is to transform each token's representation by looking at other tokens within the sequence!==
	- ((Attention combines the representation of input vector's *value vectors*, weighted by the importance score computed by the query and key vectors))
- ==In the case of *bidirectional* self-attention, each token's representation is transformed by considering *all other tokens* within the sequence, including those that come after the current token!==
	- ((This is strictly more information, but this isn't how you would train a model that generates text, because that model should be autoregressive and used causal attention, since it will be generating text token by token in the wild.))

#### Feed-forward transformation
- In contrast, the feed-forward transformation within the encoder-only architecture plays somewhat of a different role (compared to self-attention).
- Namely, this operation is *point-wise*, meaning that the same transformation is applied to every token vector within the sequence.
- ==The transformation considers only a single token vector== and applies a sequence of linear transformations -- usually two linear layers separated by a [[Rectified Linear Unit|ReLU]] activation function -- to the vector, followed by normalization:

![[Pasted image 20240223105316.png]]

#### Putting everything together
- Bidirectional self-attention and feed-forward transformations each play a distinct and important role in the encoder-only transformer!
- ==The self-attention component learns useful patterns by considering the context of other tokens in the sequence, while the feed-forward transformation only considers individual tokens.==

#### Creating Vector Embeddings
- But how can BERT be used to create a vector embedding for a sequence of text?
- This is an ability that our bi-encoders will leverage, in order to craft vectors from sequences of text that will be used for vector search.
- ==Each layer of a BERT model takes a sequence of token vectors as input and produces an equal-size sequence of token vectors as output.==
	- As such, the output of BERT is a sequence of token vectors, and every intermediate layer within the model produces a sequence of vectors with the same size:
![[Pasted image 20240223105555.png]]
- But we want to convert these lists of token vectors into a *SINGLE EMBEDDING* that represents the full input sequence! ==To turn our collection of vectors into a single representative vector, we perform some type of pooling operation.== There are three styles of pooling that are commonly used:
	1. Use the final outputted [CLS] token representation.
	2. Take an *average* over the output token vectors.
	3. Take an average (or max) of token vectors across layers.

Each of the above approaches, below:
![[Pasted image 20240223110154.png]]

Above: In general, the style of pooling does not make a massive performance difference, but ==taking an average over output token vectors (approach #2) is by far the most common approach when creating text embeddings with BERT.==


# BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding
- Now that we understand the [[Encoder-Only Architecture]], BERT is pretty easy to grasp!
- ==BERT is just an encoder-only transformer that we *pre-train* using a few [[Self-Supervised Learning]] objectives, and then we *fine-tune* (apply additional training, usually supervised) to solve some downstream task!==
![[Pasted image 20240223110351.png]]

The self-supervised pretraining objectives used by BERT (above) include:
1. ==Cloze==: Randomly mask tokens within the input sequence and train the model to predict masked tokens.
2. ==Next sentence prediction==: Given a pair of sequences as inputs, predict whether these sentences naturally follow eachother in a textual corpus or not.

The beauty of self-supervised learning is that these "labels" are implicit within the raw textual data -- they don't require human annotation.
- In the case of BERT, this corpus is all of English Wikipedia and BookCorpus!

![[Pasted image 20240223115949.png]]
((Above: SElf-Supervised pretraining objectives for BERT))

Why is BERT so great?
- BERT was one of the first transformer-based language models to leverage self-supervised pretraining over large amounts of raw textual data.
- Such an approach proved to be highly effective, as BERT set a new state-of-the-art performance on all tasks that were considered in evaluation ([[GLUE]], SQuAD, [[SWAG]])

Put simply, we can finetune a pretrained BERT model to solve a variety of different sentence and token-level classification tasks with incredibly accuracy!

==BERT revolutionized research in NLP, replacing many domain-specific techniques with a single model that can solve nearly all tasks.==

==BERT Variants==
- [[RoBERTa]] is a popular BERT variant that carefully studied BERT's pretraining process, and discovered that a better model could be obtained by pretraining over *more* data and carefully training hyperparameters.
	- Training the model longer, with more data, using larger batches.
	- Using longer textual sequences during pretraining.
	- ==Removing the next-sentence prediction objective== ((Lol, seemed sort of strange, vaguely low-signal))
	- Dynamically changing the token masking pattern used by Cloze.
- ALBERT is a variant of BERT that proposes a parameter reduction technique to make BERT pretraining faster and less memory intensive -- the result is a smaller model that still outperforms BERT.
- mBERT was a multilingual version of the original BERT model, using a shared vocabulary and embedding space across languages.
- XLM-R improves on mBERT by creating a multilingual version of the RoBERTa model, pretraining over multilingual data from Common Crawl. 

## Vector Search with BERT
- BERT is incredibly useful for solving sentence and token-level *classification problems*, as well as *semantic textual similarity tasks*.
	- Given two text inputs, we can accurately finetune a BERT model to predict the level of similarity between these two texts. (This is a [[Cross-Encoder]]). But we know that using BERT as a cross-encoder in this manner is inefficient and can only be applied at the ranking stage of search.

To more effectively retrieve relevant documents with BERT, we must use BERT as a [[Bi-Encoder]] by producing textual embeddings for all documents and indexing them in a vector database. 
- ==But we learn that textual embeddings produced by BERT are *not semantically meaningful* -- BERT struggles to function as a Bi-Encoder, and even performs poorly when used for clustering!==

At this point, the reader must be wondering: "*How could BERT embeddings possibly not work well for semantic search?*"

We aim to provide an answer for this by introducing ==sBERT==, a BERT variant that's optimized to produce more semantically meaningful text embeddings that can be used to build performant AI-powered search applications.


# Better Bi-Encoders for Vector Search
- Going beyond the basic BERT model, we will now explore several BERT variants that are *optimized for use as bi-encoders.*
	- At a high level, ==these are just BERT models that are finetuned to produce more semantically meaningful text embeddings.==

### [[Sentence-BERT]]
- Adapting models to produce more useful semantic embeddings isn't difficult -- we just need to finetune these models using a *siamese or triplet network structure to derive semantically meaningful text/document representations!*

What are Siamese and Triplet Networks?
- It's actually a simple concept:
	- With sBERT, we use the same BERT (or RoBERTa) encoder-only architecture that we're familiar with. 
	- To make it into a ==siamese network==, we just pass *two different inputs* through the model *in parallel!*
	- Then, we can apply a loss to the two outputs that are generated -- see below!
![[Pasted image 20240223124329.png]]
To use this approach, we can train a BERT model to classify whether two sentences are similar or not. To do this, we obtain a dataset of similar (and dissimilar) sentence pairs, and train the siamese network to take a pair of sentences as input and classify whether they're semantically similar or not!


The idea behind ==triplet networks== is nearly identical to siamese networks, but we pass *three* different inputs (instead of two) through the same network in parallel.
![[Pasted image 20240223124443.png]]
- This way, we have the ability to train the model using a triplet loss that considers three different outputs.
- We often have an ==anchor sentence==, a sentence that is similar to the anchor sentence (==positive==), and a sentence that is different from the anchor sentence (==negative==). ==We train the model to simultaneously make the output of the similar sentences similar and the output of the dissimilar sentences different.== (In other words, pull the embeddings of the anchor/positive sentences closer, and push the embeddings of the anchor/negative sentences further away from eachother!)

Model architectures for sBERT:
![[Pasted image 20240223124629.png]]
- sBERT finetunes pretrained BERT and RoBERTa models using three types of siamese or triplet network structures. 
- The first finetuning step for sBERT uses a *classification objective* (picture above)
	- Two sentences are taken as input and passed through a BERT model to yield embeddings `u` and `v`. We then concatenate these embeddings together (*along with their element-wise difference*) and pass this concatenated representation through a linear transformation, producing a vector of size `k`. Finally, we apply a softmax to the output of the linear layer and train the model to perform `k`-way classification.
		- So if we wanted to finetune sBERT using pairs of sentences that are either similar or dissimilar, we would have `k=2`. 
		- ==Because we're only applying a *simple, linear classification layer* on top of the BERT embeddings, we force the model to craft semantically meaningful sentence embeddings to solve this task.==

![[Pasted image 20240223125758.png]]
The second finetuning setup for sBERT (above) is a *cosine-similarity-maximizing objective*, which is almost identical to the classification objective...
- This setup look almost identical to the classification objective.
- *Instead of* concatenating vectors `u` and `v` together and apply a linear transformation, we just:
	1. Compute the cosine similarity of `u` and `v` (i.e. via a dot product, if u and v are unit vectors)
	2. Perform regression to match this cosine similarity to some target similarity value.
In this way, we can finetune sBERT over pairs of sentences with similarity scores between -1 (not similar) and 1 (similar).


The third finetuning setup explored uses a *triplet objective*.
- Each training example contains three sentences:
	- Anchor sentence
	- Similar sentence (positive)
	- Dissimilar sentence (negative)

We pass each of these sentences through BERT to obtain an embedding.
We then apply a *triplet loss* (picture below) over these embeddings in order to move the embeddings of the anchor/positive closer, and more the anchor/negative further away.
- Similarity is measured by the euclidean distance of the embeddings, and we add a margin ($\epsilon$ =1) to the loss to ensure that the similar and dissimilar sentences have a sufficiently large difference in similarity to the anchor sentence.
![[Pasted image 20240223132538.png]]


#### Finetuning sBERT
- Three different network structures overviewed above gives us a brief glimpse at the different strategies that can be used to produce more semantic meaningful embeddings with BERT -- but there are many possible finetuning setups for sBERT beyond these three!
	- As a practitioner, it's often a siamese/triplet BERT network with an added classification/regression head -- but will be adapted to the exact style of sentence or document pairs that are available
		- (eg have humans annotate a 1-5 similarity score, collect descriptions of products that are commonly purchased together, etc.)

Interesting datasets:
- SNLI: 570k sentence pairs with label of contradiction/entailment/neutral
- MultiNLI: 430k sentence pairs (same labels as above) covering a range of genres of spoken and written text

We outperform all baseline techniques using either BERT/RoBERTa via the siamese network structure and finetuning approach proposed by sBERT.

When sBERT models are further finetuned on downstream tasks used for evaluation, the results observed follow a similar trend -- we get leading results.

Mina takeaway:
- BERT models are great for a variety of tasks, but BERT's textual embeddings are not semantically meaningful by default. The finetuning procedure proposed above for sBERT models improves the meaningfulness of BERT embeddings.
	- ==As a result, sBERT is an appropriate model for a broad range of tasks, including semantic similarity tasks like clustering and semantic (vector) search.==



## Useful Extensions of sBERT
- We learn how to make BERT models more usable for dense retrieval by extending it to sBERT, so it has more meaningful embedding representations.

Making sBERT multilingual
- ==Such models can be created by leveraging the idea that translated sentence should have the same embedding as the original sentence==. In particular, the original model is used to generate embeddings of sentences in the source language.
	- Embed sentences in English
	- Translate the sentences using another model.
	- Finetune our model over the translated sentences to mimic the original model's embeddings.

sBERT is used as the teacher, while the student model is based on XLM-R -- it's a form of knowledge distillation.


Augmenting sBERT's data
- Cross-Encoders and Bi-Encoders have pros and cons
	- Cross-Encoders tend to perform well, but are too costly for practical applications (unless we only apply them in the ranking of search).
	- In contrast, Bi-Encoders are more practical, but require more training data and finetuning to perform competitively.
![[Pasted image 20240223141314.png]]

==A solution to this problem is to generate a much larger training dataset for bi-encoders by using a BERT cross-encoder to label pairs of sentences that can then be used as extra data for the bi-encoder.==
- The exact approach used to *sample* data for labeling with the cross-encoder is crucial. Namely, we cannot just randomly sample sentence pairs -- rather, we should use a sampling approach (eg BM25 sampling and/or semantic search or simply removing negative examples) ==to ensure that the ratio of similar and dissimilar sentences matches the original training dataset (which should match your inference-time).==


## Pseudo-labeling for domain adaption
- On the subject of using cross-encoders to label data for bi-encoders...
- We can use an approach called Generative Pseudo Labeling (GPL) to combine the query generator with a cross-encoder that is used to generate labels.
![[Pasted image 20240223144149.png]]
Above:
- We use T5 to generate queries for a target domain, given a passage of text (the positive example) as an input.
- From here, an existing retrieval system (eg sparse BM25 or dense vector search) is used to retrieve a fixed number of *negative* examples for each query.
- Then we form triplets of (query, positive passage, negative passage)
- For each triplet, the cross-encoder is used to predict a margin that can be used as a training signal.
- Using this signal, we can adapt a bi-encoder to a new domain without the need for labeled data.
	- And all we needed was a pretrained T5 and cross-encoder model

## Large-scale semantic search
- Authors study the performance of vector search with bi-encoders like sBERT with respect to the size of the underlying search index.... and found that ==the quality of dense retrieval deteriorates as the size of the index increases==.
- We learn that
	1. Vector search works best with a smaller, clean search index
	2. The overall quality of vector search (even in larger search indices) can be improved by introducing added "hard" negative examples to a bi-encoder's training process!

## Benchmarking Search Systems
- A new, comprehensive information retrieval benchmark called ==Benchmarking-IR ([[BEIR]])== is introduced, comprised of the following 9 tasks and 18 datasets:
	1. Fact Checking
	2. Citation Prediction
	3. Duplication Question Retrieval
	4. Argument Retrieval
	5. News Retrieval
	6. Question-Answering
	7. Tweet Retrieval
	8. Bio-Medical IR
	9. Entity Retrieval

![[Pasted image 20240223145451.png]]
It turns out on BEIR:
- ==[[BM25]] is a robust baseline that is quite difficult to beat.==
- ==Re-ranking models (ie [[Cross-Encoder]]s) yield the best performance, but their computational cost is high.==
- ==[[Bi-Encoder]]s are more efficient, but they can perform poorly in certain domains (sometimes even worse than BM25!)==
- ==Search systems that perform well in *one domain* might generalize poorly to other domains!==

Put simply, we learn that there is a tradeoff between the efficiency and performance of search systems.
- ==Typically, the best approach will be achieved via a combination of lexical search with BM25 and vector search with a bi-encoder, as well as re-ranking the final few search results with a cross-encoder.==

The BEIR paper is a great practical resource for anyone that wants to understand the different components of search systems and their impact on top-level performance


SentenceTransformers: Semantic Search in Practice!
- The Python SentenceTransformers library is built on PyTorch and HuggingFace, and provides tons of state-of-the-art models including both bi-encoders and cross-encoders -- and makes it easy to use these models efficiently and finetune it on your own data.
- This package is based on sBERT, so embeddings are semantically meaningful -- similar sentences yield embeddings that are close in vector space.

# Final Remarks and takeaways
- Search algorithms proceed in two primary phases - ==retrieval== and ==ranking==
- ==Lexical retrieval== with [[BM25]] uses keyword-matching to produce high-quality search results, and can be efficiently implemented with an inverted index.
- Going beyond BM25, we can improve retrieval and ranking quality with ==dense-retrieval== [[Bi-Encoder]] and [[Cross-Encoder]] models, respectively.
- Both bi-encoder and cross-encoders use a [[Bidirectional Encoder Representations from Transformers|BERT]]-style architecture.
- ==Cross-encoders are more computationally expensive than Bi-encoder for retrieval, so should only be used for the final ranking.==
	- ((Note: It seems that "ranking" and "reranking" in this context are the same thing. It's only ever called "reranking" because some people refer to our "retrieval" step as "ranking" -- annoying!))
- ==To use BERT as a bi-encoder, we must finetune the model== (as is done by [[Sentence-BERT|sBERT]]) to yield more semantically meaningful embeddings!


This post has focused a lot on bi-encoders instead of diving deeper into cross-encoders, but a lot of useful information on cross-encoders has been published!
Researchers, for instance, have studied the impact of using multiple stages of ranking models, using LLMs for ranking, and even considering *more data* when ranking via a cross encoder!
![[Pasted image 20240223151543.png]]
Above: 
- Cross-encoders can be expensive to use in practice, so techniques like [[ColBERT]] have ben proposed that *augment* a normal BERT bi-encoder with an added, ==late interaction stage== that can compute more fine-grained textual similarities.
- This late interaction step is more computationally efficient than a cross-encoder, allowing ColBERT to find a middle ground between bi-encoder and cross-encoders.
	- Namely, we can use ColBERT as a bi-encoder for vector search, and then perform ranking via a simplified module.

