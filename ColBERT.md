April 27, 2020 -- [[Omar Khattab]] and Matei Zaharia
Paper: [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
See also: [[ColBERTv2]], ColBERT-QA
...
Significance: ...

References:
- [Video: Zeta Alpha's "ColBERT and ColBERTv2: Late interaction at a reasonable inference cost"](https://youtu.be/1hDK7gZbJqQ?si=iWtkZxCugM9WA05U)
- [Video: Vertex Venture's Neural Notes: ColBERT and ColBERTv2 with Omar Khattab](https://youtu.be/8e3x5D_F-7c?si=l-JaHCC63j4vgusp)

----

Takeaways:
...

It's said that ColBERT generalizes better out-of-domain than dense single-vector alternatives, like Bi-Encoders. This is probably because of the more granular token-level representation of data, as opposed to a document-level representation.

Abstract
> Recent progress in Natural Language Understanding (NLU) is driving fast-paced advances in Information Retrieval (IR), largely owed to fine-tuning deep language models (LMs) for document ranking. While remarkably effective, the ranking models based on these LMs increase computational cost by orders of magnitude over prior approaches, particularly as they must feed each query-document pair through a massive neural network to compute a single relevance score. To tackle this, we present ColBERT, a novel ranking model that adapts deep LMs (in particular, BERT) for efficient retrieval. ColBERT introduces a late interaction architecture that independently encodes the query and the document using BERT and then employs a cheap yet powerful interaction step that models their fine-grained similarity. By delaying and yet retaining this fine-granular interaction, ColBERT can leverage the expressiveness of deep LMs while simultaneously gaining the ability to pre-compute document representations offline, considerably speeding up query processing. Beyond reducing the cost of re-ranking the documents retrieved by a traditional model, ColBERT's pruning-friendly interaction mechanism enables leveraging vector-similarity indexes for end-to-end retrieval directly from a large document collection. We extensively evaluate ColBERT using two recent passage search datasets. Results show that ColBERT's effectiveness is competitive with existing BERT-based models (and outperforms every non-BERT baseline), while executing two orders-of-magnitude faster and requiring four orders-of-magnitude fewer FLOPs per query.


> Q: Why is ColBERT superior to traditional embedding models?
> A: The idea that you can accurately boil down the nuances of ~256 tokens (2/3s of a page, e.g.) into a single vector is a pretty wild proposition. No matter how good the model, semantic nuance and details will inevitably be lost. Instead, ColBERT's approach is to allocate a small, efficient representation to EACH TOKEN within the passage; this way, you're not crossing your fingers that your compression strategy isn't crushing a lot of semantic value. 
> At a high level, you embed the query and the passage, getting vector representations for every token in both. Then, for each query token, you find the token in the passage with the largest dot product similarity, which is called the "MaxSim" for each token. Finally, the similarity scores between the query and passage is just the summation of all the MaxSims you found. So while you might have to compute many such dot products, each vector is much smaller than usual (eg dimensionality of 4), so it scales much better than you'd think. The "RAGatouille" library is much easier to grok than the original library.
> Source: https://x.com/marktenenholtz/status/1751406680535883869

> The MaxSim operator's introduction in the original ColBERT paper is perhaps the deepest insight in the paper; inspired by the things that work well in traditional IR, where, event though you're working with a bag of words representation, you don't want to score e very document that has one or more of the terms. Instead, you do some pruning. In Search, if you can prove that some document's cant possibly have a high enough score, you save a lot of work. For our ColBERT scoring function, we have two bags of vectors, one from each the query and document. A document is relevant to a query (a bunch of words), IFF, for most terms in the query, there's a contextual match on the document end. So for each term in the query, we find the closest vector on the document side, repeat for each word in the query, and sum up these partial scores to get an average of how well is the query contextually captured in the document. 
> You can think of ColBERT as two things: You can use it out of the box as search, or you can think of it as the key idea of doing late-interaction between fine-grained representations. Along that spectrum, there are a bunch of modular components: The encoder (a BERT model or other language encoder that takes in text and spits out a bag of vectors), the search stack (Here, you could be very modular, but the more modular you are, you might be leaving some e2e optimizations. For a long time, ColBERT was a modular wrapper around FAISS).
> - Omar Khattab (https://youtu.be/8e3x5D_F-7c?si=eMh4Z0FdEhUwZJy_)

# Paper Figures


# Additional Figures
![[Pasted image 20240414150952.png]]

