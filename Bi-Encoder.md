Also known as:
- Dual Encoder
- Two Tower Network
- Siamese Network
- DSSM (Deep Semantic Structure Modeling)

Challenges
- Bi-Encoder models suffer from issues like the embedding space bottleneck, information loss, and limited expressiveness due to fixed-size embeddings, as well as lack of fine-grained interactions between embeddings.
- At the inference stage, it requires a large document index to search over the corpus, leading to significant memory consumption that increases linearly with corpus size.
- In a dual-encoder setup, the representation for queries and documents are obtained independently, allowing for only shallow interactions between them.
- By limiting query and document representations to a single fixed-size dense vector, dual encoders also potentially miss fine-grained information when capturing the similarity between two vector representations. This is even more critical in the case of multi-hop retrieval!