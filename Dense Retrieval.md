Unlike [[Sparse Retrieval]], Dense Retrieval uses dense vector text embeddings to search for documents that are similar to a query.

It consists of the following:
1. Finding the embedding vector corresponding to the query.
2. Finding the embedding vectors corresponding to each of the responses (in this case, Wikipedia article).
3. Retrieving the response vectors that are closest to the query vector in the embedding space.

Because Dense Retrieval uses semantic embeddings of terms, it might do better at finding things that are semantically similar to the query than a lexical search might do, like handling synonyms and acronyms.