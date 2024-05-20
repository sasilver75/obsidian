January 2, 2021 (8 months after [[ColBERT]])
[[Omar Khattab]], [[Christopher Potts|Chris Potts]], [[Matei Zaharia]]
Paper: [Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval](https://arxiv.org/abs/2101.00436)
#zotero 
Takeaway: Baleen is a system for [[Multi-Hop]] question-answering, using [[ColBERT]] as a retriever. It introduces:
- **==condensed retrieval==** (where we summarize  retrieved passages after each hop into a single compact context)
- A ==focused late interaction== retriever that allows different parts of the same query representation to match disparate relevant passages.
- **==Latent hop ordering==**, a strategy where the retriever itself selects the series of hops.

---

Notes: 
- 

Abstract
> ==Multi-hop reasoning== (i.e., reasoning across two or more documents) is a key ingredient for NLP models that leverage large corpora to exhibit broad knowledge. To retrieve evidence passages, ==multi-hop models must contend with a fast-growing search space across the hops, represent complex queries that combine multiple information needs, and resolve ambiguity about the best order in which to hop between training passages==. We tackle these problems via ==Baleen==, a system that improves the accuracy of multi-hop retrieval while learning robustly from ==weak training signals== in the many-hop setting. To tame the search space, we propose ==condensed retrieval==, a ==pipeline that summarizes the retrieved passages after each hop into a single compact context==. To model complex queries, we introduce a ***==focused late interaction retriever==*** that allows different parts of the same query representation to match disparate relevant passages. Lastly, to infer the hopping dependencies among unordered training passages, we devise ==latent hop ordering==, **a weak-supervision strategy in which the trained retriever itself selects the sequence of hops**. We evaluate Baleen on retrieval for two-hop question answering and many-hop claim verification, establishing state-of-the-art performance.