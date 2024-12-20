Ranking is one of the fundamental problems in IR, which usually occurs after the initial retrieval of documents that (perhaps approximately) match the query.

Given a query $q$ and a collection of documents $D$ that match the query, the problem is to sort the documents in $D$ according to some criterion so that the "best" results appear early in the result list displayed to the user.

Paradigm of "==listwise reranking==" and "==pairwise reranking==" and "==pointwise reranker=="