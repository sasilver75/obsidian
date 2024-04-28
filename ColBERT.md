April 27, 2020 -- [[Omar Khattab]] and Matei Zaharia
Paper: [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
See also: [[ColBERTv2]], ColBERT-QA
#zotero 
Significance: ...

----

Takeaways:
...

Abstract
> Recent progress in Natural Language Understanding (NLU) is driving fast-paced advances in Information Retrieval (IR), largely owed to fine-tuning deep language models (LMs) for document ranking. While remarkably effective, the ranking models based on these LMs increase computational cost by orders of magnitude over prior approaches, particularly as they must feed each query-document pair through a massive neural network to compute a single relevance score. To tackle this, we present ColBERT, a novel ranking model that adapts deep LMs (in particular, BERT) for efficient retrieval. ColBERT introduces a late interaction architecture that independently encodes the query and the document using BERT and then employs a cheap yet powerful interaction step that models their fine-grained similarity. By delaying and yet retaining this fine-granular interaction, ColBERT can leverage the expressiveness of deep LMs while simultaneously gaining the ability to pre-compute document representations offline, considerably speeding up query processing. Beyond reducing the cost of re-ranking the documents retrieved by a traditional model, ColBERT's pruning-friendly interaction mechanism enables leveraging vector-similarity indexes for end-to-end retrieval directly from a large document collection. We extensively evaluate ColBERT using two recent passage search datasets. Results show that ColBERT's effectiveness is competitive with existing BERT-based models (and outperforms every non-BERT baseline), while executing two orders-of-magnitude faster and requiring four orders-of-magnitude fewer FLOPs per query.


> Q: Why is ColBERT superior to traditional embedding models?
> A: The idea that you can accurately boil down the nuances of ~256 tokens (2/3s of a page, e.g.) into a single vector is a pretty wild proposition. No matter how good the model, semantic nuance and details will inevitably be lost. Instead, ColBERT's approach is to allocate a small, efficient representation to EACH TOKEN within the passage; this way, you're not crossing your fingers that your compression strategy isn't crushing a lot of semantic value. 
> At a high level, you embed the query and the passage, getting vector representations for every token in both. Then, for each query token, you find the token in the passage with the largest dot product similarity, which is called the "MaxSim" for each token. Finally, the similarity scores between the query and passage is just the summation of all the MaxSims you found. So while you might have to compute many such dot products, each vector is much smaller than usual (eg dimensionality of 4), so it scales much better than you'd think. The "RAGatouille" library is much easier to grok than the original library.
> Source: https://x.com/marktenenholtz/status/1751406680535883869


# Paper Figures


# Additional Figures
![[Pasted image 20240414150952.png]]

