A text embedding is a compressed, abstract representation of text data where text of arbitrary length can be represented as a fixed-size vector of numbers. Usually learned from a corpus of text like Wikipedia.

Note: While we usually discuss text embeddings, embeddings can take many modalities; [[CLIP]] is multimodal and embeds images and text in the same space, allowing us to find images most similar to input text, and vice-versa.

Note that while embedding-based search unlocks a bunch of interesting semantic search options, there are still situations in which it falls short, such as:
- Search for a person/object's name (Sam Silver, Eclipse 2.7)
- Searching for an acronym or phrase (RAG, RLHF)
- Searching for an ID (gpt-3.5-turbo, titan-xlarge-v1.01)
But keyword search has its own limitations too, since it only models simple word frequencies and doesn't capture semantic or correlation information, so it doesn't deal well with synonyms or hypernyms (words representing a generalization; *color* is a hypernym of *red*).
However, with conventional search indices, we can use metadata to refine results (eg date filters to prioritize newer documents, filters on average rating or categories, etc.)


Sparse Embeddings vs Dense Embeddings
[[Positional Embeddings]]



----
![[Pasted image 20240130162514.png]]
