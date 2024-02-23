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
- ==Given this cost, cross-encoders are typically only applied at the *ranking* stage of search!==



