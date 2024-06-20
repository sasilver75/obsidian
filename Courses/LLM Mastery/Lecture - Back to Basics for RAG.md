With [Jo Bergum](https://x.com/jobergum) from Vespa (June 18)
Distinguished engineer @ Vespa AI (18 year tenure @ Vespa)
Has been working in search/recommendation for 20 years
- Vespa was recently spun out of Yahoo (2017)

The majority of this talk is about setting up evaluation for your search so that you have something to iterate on -- interestingly, LLMs can actually help us with this!

![[Pasted image 20240619235829.png|300]]

If you have 1 million annotated training examples, you can use retrieval to retrieve relevant examples, and have the LLM reason around those, to predict a label.
But most people think around RAG in the context of building this sort of question-answering model.

![[Pasted image 20240620000053.png|300]]
Lots of things goin on here, as well as a lot of hype and methods around RAG
![[Pasted image 20240620000109.png|300]]

The retrieval step in RAG has been around for decades.

How do we evaluate IR systems?
![[Pasted image 20240620000333.png|200]]

The basic idea of evaluating IR systems is that you take a query, and retrieve documents, and then have human annotators (eg) judge the quality of documents
- Binary judgements (Good, bad)
- Graded judgement label (very relevant, slightly relevant, irrelevant, my boss will fire me) -- ((Likert Scale))

![[Pasted image 20240620000549.png|400]]
- TREC is the text retrieval conference, spanning multipel documents every year.
- [[MS MARCO]] is one of the largest datasets from Bing that you can publish results on -- many embedding models are trained on this dataset.
- [[BEIR]] from [[Nils Reimers]] et al which evaluates models in a zero shot setting on many domains/tasks

The solution to actually measuring how well you're doing is to build your own relevancy dataset -- look at what users in production are searching for, and look at the results.
- Put in a few hours, and judge the quality of those results -- are they actually relevant?
- If you don't have traffic, you've at least played with the product, right?
- At hte minimum, given some content, ask a LM what question would be natural for an LM to retrieve this passage.


![[Pasted image 20240620001441.png]]
Microsoft people find that with some good prompting, they can have LLMs be pretty good at judging query/passage relevance!
- If you find a prompt combination that actually correlates with YOUR domain/dataset, then you can start using this at a more massive scale.

![[Pasted image 20240620001705.png]]
UMBRELA paper a week ago or so.
- This prompt works very well at assessing the relevancy of queries.

![[Pasted image 20240620003033.png|300]]
Nice to say "We deployed this change and it increase nDCG@12 by 30%, on our own in-domain dataset that we've created."

Regarding representational approaches in IR
![[Pasted image 20240620003134.png]]

We want to try to avoid scoring *all* documents in the collection.
- Some might have heard about Cohere Rerankers, where you input a query and a collection of documents, and they score everything... 

Instead, we'd like to do something where we havea technique to represent documents so that we can index them, so when a query comes in, we can efficiently retrieve in a sublinear time, top-ranked docs, and then feed them into subsequent ranking phases...

Two primary representations: Sparse representation (sparse vector of size vocabulary, with counts for word occurrences) vs dense representation (using neural models to learn semantic representations of text in a latent representation space, and then comparing using some kind of distance metric. Typically off-the-shelf models that you then try to apply to your use case, though you can fine-tune on your own data)
![[Pasted image 20240620003318.png|450]]

![[Pasted image 20240620004026.png]]
Take input text, tokenize into fixed vocabulary, and we have learned representations of each fixed token, and then for each token, we have output vectors. There's a pooling step where we average each of these output embeddings into a single embedding (or just use the CLS token representation). The issue is that the representation becomes quite diluted (representing a large paragraph with a relatively small single vector). ==Especially for precision retrieval, this doesn't work as well, and if you're using BERT trained in 2018, its vocabulary doesn't know a lot of words! These get mapped to UNKs, which then results in quite weird results==.

![[Pasted image 20240620004244.png]]
BM25 is a great baseline

![[Pasted image 20240620004433.png]]
Having the mindset that you can evaluate and see what works, and remembering that BM25 CAN be a strong baseline is an important takeaway.

![[Pasted image 20240620004509.png]]
Lots of enthusiasm about Hybrid search; we can overcome the vocabulary issue discussed above with dense retrieval models... but it's nota silver bullet (though it does avoid the common failure scenarios with single-vector representations)


![[Pasted image 20240620004628.png]]
Nil Reimers: If you go beyond 256 tokens, you start to get bad results for high-precision search; the embeddings become diluted because of the pooling operations of embeddings.
- Models generally haven't even been trained with longer sequences.

![[Pasted image 20240620004716.png]]

![[Pasted image 20240620004814.png]]


This was kind of a dogshit talk
- "I think IR is more than just about single-vector representations. You should look at building your own evals ((THOUGH HE GIVES NO GOOD INFORMATION ON DOING THIS)), don't ignore good baselines, think about hybrid search, etc."


Questions:
- What kind of metadata is most useful to put into a vector DB for RAG?
	- If you're only concerned about text (no freshness component), eg users asking healthcare questions... you definitely want some filtering for authority of sources (you don't want to retrieve off reddit). But it really depends on use case.
- Thoughts on Rerankers?
	- The great thing about rerankers is that ... it's nice to invest more compute into fewer hits. The great thing about rerankers like Cohere Rerankers is that they offer rich interaction between tokens in the query and document.
- Advice on combining usage data (eg # views) along with semantic similarity?
	- If you have interaction data, it becomes more of a learning to rank problem; you need to come with labels from that usage data. When you convert that to a labeled dataset, you can train a model where you include the semantic score as well as a feature. (???)
- What are some favorite recent advancements in search technologies?
	- I think embedding models will become better. I hope we'll have models with larger vocabularies. 
	- I'd love to see a [[DeBERTa]] model trained on more tokens (+ more params) and with a larger vocabulary. I'm not hyped about increasing the context length, because research shows that we aren't good at turning long contexts into good single embeddings.
- Does query expansion for out of vocabulary words work... for search? Are people going as far with classical techniques (eg query expansion) as they should?
	- You can get really good results by starting with BM25 and classical techniques, and adding a reranker on top of that. You won't get the magic if you have a single-word query and there are no words in your collection, but you don't get into the nasty failure modes of just using single vector search.
- ColBERT?
	- Instead of learning one representation, you learn token-level representations. This is a bit more expensive at inference time in terms of compute ((?)), but it still suffers from vocabulary problems, because it uses the same vocabulary as other models. If we can get better pretrained models trained with a larger vocabulary, we hope that it's a path towards better neural search with ColBERT and other models as well.






