#lecture 
Link: [Information Retrieval Lecture](https://youtu.be/enRb6fp5_hw?si=HReeZswa891GA5Mb)
Guest speaker: [[Omar Khattab]]

----

# Part 1: Information Retrieval

NLP is revolutionizing [[Information Retrieval]]
- Soon after [[Bidirectional Encoder Representations from Transformers|BERT]] was launched, Google announced that it was incorporated aspects of BERT into its search technology; Microsoft soon followed, re: Bing.
- Later, we saw that LLMs might play a role in search as well.
	- The startup You.com (and now, Perplexity) were both visionaries in this sense.
	- Microsoft has partnered with OpenAI and now uses OpenAI models as part of the Bing search experience.


Information retrieval is a hard [[Natural Language Understanding|NLU]] problem!

Query: What compounds protect the digestive system against viruses?
Document retrieved: In the stomach, gastric acid and proteases serve as powerful chemical defenses against ingested pathogens.

See above that the relationship between the query and document above is entirely semantic!


Also, IR is revolutionizing NLP (as well!)
- Standard QA
	- Given a Title and a Context passage and a Question, the task is to answer the Question, but the Answer is guaranteed to be a literal substring of the entire passage.
	- [[SQuAD]] 
	- ![[Pasted image 20240425204447.png|250]]
	- This used to be a hard problem, but has grown quite easy, and is even disconnected from actual things that we want to do with question-answering in the world (we rarely have this substring guarantee)
- Open QA
	- Maybe there's a title, context, and question, and you must answer, but now *only the question and the answer* are given at train time, but the context and title need to be *retrieved*.
	- You pose a question and need to retrieve relevant information to answer the question.
	- ![[Pasted image 20240425204610.png|250]]


Question-Answering is just a single incarnation of "Knowledge-intensive tasks"
1. Question answering
2. Claim verification
3. Commonsense reasoning
4. Long-form reading comprehension
5. Information-seeking dialogue
6. Summarization (arguable; Standard summarization is closed, but it could be made a knowledge-intensive task by augmenting with retrieved information to improve summarizations
7. Natural Language Inference (arguable; Usually it's posed as a closed classification problem (Premise, Hypothesis), and you give one of three labels (entailment, contradiction, neutral). But wouldn't it be interesting if you could inject world knowledge that would help us make better predictions?)


## Classical Information Retrieval

![[Pasted image 20240425204943.png|250]]
- Query: "When was Standford University Founded?"
- We do *==term lookup==*, where we map terms to related documents.
- On the basis of that index, we can then do ==document scoring==, and give back to the user a list of documents, ordered by relevance.

## LLM for Everything
There's now a movement to replace this all with pure language models, the "==LLM for everything==" approach:
![[Pasted image 20240425205010.png|250]]
This is just relying entirely on the parametric "knowledge" of the language model... and we know that models can confabulate and hallucinate. 
We're deeply concerned about this model; we should be pushing in a different direction.

## Neural information retrieval
![[Pasted image 20240425205751.png|250]]
- We have a big language model, but we use it somewhat differently; we take all of our documents we have in our collection of documents, and create (eg) dense numerical representations of those documents. This is essentially our document index from the classical mode
- Query comes in; we process it into a dense numerical representation, and do scoring and extraction as usual; The only twist from the classical is that we deal with dense numerical representations instead of words.
- We give the user back a ranked list of pages, so the user has the same experience.
	- We hope that we offer better pages, because we operate in a richer semantic space.


## Retrieval-Augmented In-Context Learning

Q: Who is Bert?

![[Pasted image 20240425210019.png|250]]
How can we effectively answer this question using Retrieval?
- We can retrieve from a document store a context passage for the question and inject it into the prompt
	- Context: Bert is a muppet who lives with Ernie
- We can add other demonstrations (eg [[Few-Shot Prompting]]) to the prompt

We'll see that this is just the start of a very rich set of options that we can employ to effectively develop in-context-learning systems to process/develop evidence.

## IR is more important than ever!
- We see a lot of worrisome behavior from LLMs that are deployed as part of search technologies; Google got a real hit from making a minor factual error in one of its demo videos; maybe that's appropriate!
	- (At the same time, OpenAI models fabricate evidence all over the place!)


---

# Part 2: Classical Information Retrieval
![[Pasted image 20240425210614.png]]
- The standard starting point is the ==term-document matrix==
	- Terms along rows, documents along columns, cells record how often the terms occur in the documents
	- Usually very large, very sparse matrices, but encode latently a lot of information about which documents are relevant to which query terms.

![[Pasted image 20240425210752.png]]
==TF-IDF== is a way of massaging those values to get more information about relevance
- Begin from a corpus of documents $D$ 
- Term frequency of word, given document, is the number of times the word appears in the document, divided by the length of the document
- Document frequency of words is just counting the number of document that contain the target word, regardless of how many times it occurs
- Inverse document frequency is the log of the total size of our document corpus divided by the document frequency value above.
- [[TF-IDF]] then is just the product of the TF and IDF values.

Above: Term A gets a TF IDF value of 0, because it occurs in all documents, so ends up with an IDF value of zero.


![[Pasted image 20240426134723.png|200]]
==With TF-IDF, we're looking for words that are *truly* distinguishing indicators of a document.== 
- TF-IDF reaches *maximum values*  for terms that are *very frequent* in a small number of documents
- TF-IDF reaches *minimum* values for terms that are *very infrequent* in a *very large* number of documents (but it's also very small for terms that are very frequent in almost every document, like "and")


To calculate relevance scores for a given query with multiple terms, we do a 
![[Pasted image 20240426134914.png]]
Where here, Weight would be the TF-IDF score.


## BM 25
- [[BM25]] stands for "Best Match, Attempt #25", a classical IR approach whose name suggests a lot of exploration of the hyperparameters of the model that work best
	- It's an enduringly good solution taht's sort of an enhanced version of TF-IDF

![[Pasted image 20240426135115.png|400]]
- We begin with smooth IDF values (with a little bit of adjustment to handle the undefined case that we might worry about)
- We then have a Scoring component, which is sort of analaogus to Term Freqeuncy; we also have two hyperparameters k and b
- The BM25 weight is then a combination of the adjusted, smooth IDF values and the Scoring values (analagous to term frequency).

Let's look at the individual components, starting with the Smoothed IDF

#### BM25: Smoothed IDF term (+s hyperparameter)
![[Pasted image 20240426135151.png|300]]
- This very closely matches the standard IDF values

![[Pasted image 20240426135210.png|300]]
- As we vary the $s$ hyperpameter, we see that we very closely mirror the results of the usual IDF from TF-IDF, with small differences.

#### BM25: Scoring Function component (+b, k hyperparameters)

![[Pasted image 20240426135301.png|400]]
The scoring function is more nuanced, as result of having more hyperparameters
- Term Frequency on the X axis, BM25 score on the Y axis
- If the document has average document length of 10, then as our example document becomes long relative ot its average doc length of 5 or 3, we can see that the score goes down; when our document is large relative to the average, score is dimished.
	- Intuition: Long documents, as a result of being long, contain more terms; we should "trust" the terms they *do* contain less, as a consequence

![[Pasted image 20240426135434.png|400]]
- The $b$ hyperparameter controls the amount of the document length penalty described above. Higher values of b mean more of a penalty given to long documents.
- Again, we have a target document of length 10, and an average document of length 5; As we increase the $b$ value, we see a lower BM25 score as a result of a greater penalty.
- Right: If your document has length 10, and the average length in the corpus is also 10, then the b hyperparameter doesn't make any difference, because the parenthetical penalty term just becomes 1.


![[Pasted image 20240426140547.png]]
- The $k$ hyperparamter has the effect of flattening out higher frequencies.
	- Think about the extreme situation in which k is very very low. In this situation, you're essentially turning the scoring function into an indicator function (see right side, red line). "You appeared, I don't care how many times you appear."
	- As you make k larger, you get less and less of a dramatic effect, and you care more and more about how many times the document appears. 
	- Something like 1.2 is a more realistic value; as you get *very frequently* occurring terms, we begin to sort of taper off our weighting of it.



## Inverted Indices
![[Pasted image 20240426142835.png|300]]
Let's return to our inverted index (going from terms:documents, rather than documents:terms) from classic information retrieval!
- We can now ==AUGMENT THIS with precomputed IDF values==:
	- ((It sounds like we can do this offline? that makes sense, perhaps, because you only have TF-IDF scores for the words in your document, and you know what those are (and what the other documents are) ahead of time))

![[Pasted image 20240426142907.png|300]]

This is an essential ingredient as to why these approaches are so scalable.



## Beyond Term Matching
- We can expand the Query and Document, augmenting what the user gives us, and what's in our corpus, with metadata.
- Move to phrase-level search; we've moved to UniGrams, but we could use N-Grams, or more sophisticated notions.
- We haven't considered "term dependence"; Bigrams like "New York" are important; we should be thinking about how these terms have their own statistical interdependencies and bring them into search functionality.
- Documents aren't homogenous; words in Titles and words in Body are different, and might inherently have different relevance models. Our best search technologies should be aware of this
- Link analysis (eg Pagerank); How do the documents in our corpus form an implicit graph based on how they hyperlink to eachother.
- Learning to rank! Learn functionality for what's relevant, given queries. An important feature of Neural IR models that we've discussed.

Tools for classical IR 
- ElasticSearch
- Pyserini, PrimeQA (research repositories that could be useful when setting up classic search things as baselines, or using them compositionally in hybrid search)


----

# Part 3: IR Metrics

- There are many ways to assess the quality of IR systems
	1. ==Accuracy-style metrics==: these will be our focus
	2. ==Latency==: Time to execute a single query; incredibly important in industrial contexts -- users demand low-latency systems. You often see accuracy/latency paired.
	3. ==Throughput==: Total queries served in a fixed time, perhaps via batch processing. Sometimes related to latency; you might sacrifice per-query speed to process batches efficiently.
	4. ==FLOPs==: A hardware-agnostic measure of compute resources; Hard to measure/reason-about, but might be a good summary method
	5. ==Disk usage==: For the model, index, etc. If we're going to index the whole web, the cost of storing it all on disk might be very large.
	6. ==Memory usage==: For the model. index, etc.
	7. ==Cost==: Total cost of deployment for a system; summarizes all of 2-6, in a way.
		- If we want great latency, so we store everything in memory, but this is very expensive.
		- We might then cut costs my making our system smaller, but that would lead to a loss in accuracy.


## Relevance Data Types
- Given a query q and a collection of N documents D:
	1. One data type you might have is a complete partial gold ranking **D** = [doc1, ... docN] of D, with respect to a query q; you'd need to do this for every query in your dataset.
		- Inordinately expensive; if you have these, it was likely automatically generated.
	2. An incomplete partial ranking of D with respect to q.
	3. Labels for which passages in D are relevant to q
		- Simply binary judgements for whether documents in our corpus are relevant to a given query. Could be based on human labeling or on some [[Weak Supervision]] heuristic; i.e. whether the document contains parts of the query as a substring. This can be noisy, but also powerful when it comes to training good IR systems.
	4. A tuple consisting of one positive document $doc^+$ for q, and one or more negative docs $doc^-$ for our query.


## Metrics: Success and Reciprocal Rank

![[Pasted image 20240426144456.png]]
[[Rank]], [[Success]], [[Reciprocal Rank]], and [[Mean Reciprocal Rank]]

![[Pasted image 20240426144829.png]]
Comparing [[Success]] and [[Reciprocal Rank]]
Which of these is the best metric? It's not clear. For either of these metrics though, we really only care about *one star* (where the star here represents that some retrieved document is ground-truth *relevant* to a query)


## Precision and Recall

The ==[[Return Set]]== Ret(D, K)  of a ranking at value K is the set of documents at or above K in D.

The ==[[Relevance Set]]== Rel(D, q) for a query, given a document ranking, is the set of all documents that are relevant to he query (anywhere in the ranking).

[[Precision]] in an IR context is thus:
![[Pasted image 20240426145216.png]]
If we think about the values >= K as the "guesses" we made, Precision says how many of those were "good" ones.

And [[Recall]] is the dual of that:
![[Pasted image 20240426145220.png]]
Out of all of the relevant documents, how many of these relevant document bubbled up to be > K, in the ranking.

![[Pasted image 20240426145603.png]]
@K=2: Document ranking D_3 isn't doing so hot, right?
![[Pasted image 20240426145636.png]]
@K=5: Now document ranking D_3 is clearly in the lead! ==This shows you how important our value of K was to our assessment of quality.==


## Average Precision
Average Precision is a less-sensitive metric to values of K than Precision is.

Average precision for a query, relative to a document ranking
![[Pasted image 20240426145734.png]]
For the numerator, we get precision values for every step where there *is* a relevant document (every place where there is a star; we sum those up). The denominator is the set of relevant documents.
It's more clear with a picture:
![[Pasted image 20240426145858.png]]
See: D1 is the clear winner 

## So which metric should we use?
1. Is the cost of scrolling through K passages low? Then perhaps Success@K is fine-grained enough.
2. Are there multiple relevant documents per query? If so, Success@K and RR@K may be too coarse-grained...
3. Is it more important to find *every relevant document?* If so, favor [[Recall]] (Cost of missing documents is high, cost of review is low)
4. Is it more important to *only view relevant documents?* If so, favor [[Precision]] (Cost of review is low)
5. [[F1 Score]] F1@K is the harmonic mean of Prec@K and Recall@K. It can be used when there are multiple relevant documents but their relative order above K doesn't matter much.
6. [[Average Precision]] will give the finest-grained distinctions of all metrics discussed; it's sensitive to rank, precision, and recall.



![[Pasted image 20240426150320.png]]
The ColBERT systems, to achieve that MRR, need pretty heavy hardware; 4 GPUs! Whereas the SPLADe systems are comparable in terms of quality, but require fewer hardware resources. And then within those SPLADE systems, there are other tradeoffs around hardware and latency.

![[Pasted image 20240426150405.png]]
See tha bM25 is 

----

# Part 4: Neural Information Retrieval

IThe name of the game is to take a pre-trained [[Bidirectional Encoder Representations from Transformers|BERT]] models and finetune it for information-retrieval purposes.
-  In that context, [[Cross-Encoder]]s are the simplest approach you could take

![[Pasted image 20240429180317.png]]
Concat the ==query== and ==document== text into a single text, process it with BERT, and use representations in the BERT model as the basis for IR finetuning.
- We might take the final output state above the CLS token, add some task-specific parameters on top, and then finetune it on our objective.
- This allows for many interesting interactions between query and document.

The richness comes from us using the BERT model to jointly encode the query and documents, but that would mean that we need to encode EVERY DOCUMENT at query-time!
- If we had 1B documents, we'd have to do 1B forward passes with our BERT model to get scores, which is simply unfeasible; ==it doesn't scale==.
- There's ==something conceptually right about this approach,== but it's clearly wrong/intractable for modern search.

### Dense passage retrieval
![[Pasted image 20240429181202.png]]
[[Dense Passage Retrieval]] (DPR) is on the other side of the spectrum
- In this mode, we ***separately*** encode queries and documents.
- We're going to do scoring as a dot product of the two vectors produced (eg) above the CLS token in each encoder.
- The loss is as before, the negative log-likelihood of the positive passage.
- This is ==highly scalable==, but ==limited in terms of the query/document interactions== that it provides!
	- The core of the scalability is that now we can encode all of our documents offline, ahead of time.
	- At query time, we just encode the query once, and do a fast dot product with all of our documents.
	- At the same time, ==we've lost all of the token level interactions that we had with cross-encoder==; we better hope that those boiled single-vector representations produced for our query and key vectors contain *all* the important information needed to correlate queries and documents! It's perhaps a loss in expressivity for our model, in some way.


## Shared Loss Function
- We have a little bit of modularity; we used the negative log likelihood of the positive passage
- ![[Pasted image 20240429181645.png]]
- This general form is freeing, because if we develop variants of DPR or Cross Encoders, the way it plays out might be just that you have a new comparison function.


## ColBERT
![[Pasted image 20240429182009.png]]
- [[ColBERT]]  is near and dear to Potts; developed with Omar Khattab and Mattei Zaharia
- First, we encode queries using BERT
- Similarly, we encode the document again with BERT
- Then the basis for ColBERT scoring is a matrix of *similarity scores* between query tokens and document tokens, as represented by these final output tokens.
- The basis for scoring is a "MaxSim" comparison; for every query token, we get the value of maximum similarity over document tokens, and sum these up to get the MaxSim value that's the basis for the model.
- ==Highly scalable,== with late contextual interactions between tokens in the query and tokens in the document.
	- Scalable, because we can store all of our encoded documents ahead of time
	- At query time, we encode the query, and perform a multiplication with a bunch of MaxSim scoring.
	- At the same time, it's ==semantically rich,== because we have token-level comparisons between queries and documents (unlike the DPR example above)
	- ColBERT has indeed proven to be a 

### ColBERT as a re-ranker (or any of these neural models)
- How can these be effective as search technologies?
- We have semantic expressiveness, but it comes at the price of forward inference in (large) BERT models, which can be prohibitively expensive for some applications.
- As a result, we employ these models as ==Re-Rankers==
![[Pasted image 20240429182839.png]]
- For ColBERT, we have an index of token-level representations.
- Given an index structure like this, we take our query:
	1. We get the top K documents for our query $q$ using a fast, term-based model like [[BM25]].
	2. Score each of these top K documents using ColBERT, reranking them by the resulting score.
- This sounds like a small thing, but the reranking that happens in the second phase can be incredibly powerful.
- ==We can control our costs, too==; we can set $K$ very low, to do very little processing with ColBERT, or set $K$ to be very high, and do much more reranking with ColBERT.

Now we have two retrieval mechanisms in place; we might hope for a more integrated solution


### Beyond Reranking for ColBERT
- ![[Pasted image 20240429183839.png]]
- Let's make a slight adjustment to how we set up the index
- We have an index of token-level vectors that are associated with documents.
- When a query comes in, we encode it into a sequence of vectors.
	- For each of these query representation vectors, we retrieve the $p$ most similar token vectors, and travel through them to their associated documents.
	- Then we score each document using ColBERT.


We can do slightly better, even, with ==Centroid Ranking==

### Centroid-Based Ranking
- This begins from the insight that the index that we construct (of word vectors from our documents) will have a lot of semantic structure, so we can begin by *clustering* the token-level vectors representing our documents into clusters, and taking their centroids to be the representative vectors of each cluster, which we then use as the basis for search.
- So a query comes in, and we again generate a sequence of token vectors for every token in the query. 
	- For each query token vector, we retrieve the $p$ relevant centroids
	- We then retrieve the $t$ most similar token vectors to any of the centroids
	- We then use ColBERT to rerank the docs related to each of those document tokens. 
- This gives us huge gains, because instead of having to search over the entire index, we search over a very small number of centroid representations that we then use as the basis for getting down to a small set of documents that we rerank with ColBERT.


Some interesting discussion about tradeoffs that they choose to make to optimize ColBERTv2's latency

![[Pasted image 20240429185142.png]]
- Only the Blue and Purple bits are the Core ColBERT model steps; most of the time isn't spent on the core modeling steps, or even with the centroids (orange). The bulk of the work is looking things up in the giant index and doing decompression (which we haven't mentioned yet; The index gets large because of token-level represntations, so we shrink the index by using quantized versions of representations; we decompress them at some point to get back to full semantic richness).

![[Pasted image 20240429185354.png]]
The team did a lot of work to reduce the heavy-duty decompression, trading that off against using more centroids. They were able to remove almost all the overhead coming from the large datastructures and corresponding decompressions, and got the latency down to 58ms.
- Shows: A lot of innovative work can come from working on systems and realistic concerns.


### SPLADE
- [[SPLADE]] offers another perspective on how to use neural representations.
![[Pasted image 20240429185728.png]]
- The core shift in perspective is that we do scoring *not to some other text,* but instead *with respect to our entire vocabulary* (eg 10,000s of items).
- We have an encoding mechanism for sequences (agnostic as to whether query or document).
- We form a matrix of scores, where the scoring is now with respect to the {tokens in the sequence we're processing} and {all of our vocab items}
- The SPLADE scoring function is a kind of sparsification of the scores that we get out of that. The essential insight is that we get a score for *every vocabulary item* with respect to the sequence that we processed (the orange thing, above). This is a vector with the same dimensionality as our vocabulary, giving scores.
- We do this for queries and documents.
- The similarity function:
	- Sim_SPLADE is a dot product between the SPLADE representation for the query and the SPLADe representation of the document.
The loss is a usual negative log-likelihood plus a regularization term that leads to sparse, balanced scores.


![[Pasted image 20240429185807.png]]
Advances around making systems more efficient and more multilingual


# Part 5: Datasets and Conclusion

### TREC (Text Retrieval Conference (TREC)
- Has annual competitions for comparing IR systems
- Tends to emphasize careful evaluation with a very small set of queries (eg 50 queries, each with >100 annotated documents)

### MS MARCO ranking tasks
- [[MS MARCO]] is the largest public IR benchmark, adapted from a public IR benchmark, based on more than 500k Bing search queries.
- Sparse labels: Approximate one relevance label per query!
- For passage ranking, you have 9M short passage; sparse labels
- For document ranking: 3M long documents; sparse labels
- ==Fantastic for training IR models==

### BEIR ("Beer"): Benchmarking IR
- [[BEIR]] is about doing diverse, zero-shot evaluations across IR models, across a bunch of different domain and task settings.

### [[LoTTe]]: Long-Tail, Topic-stratified Evaluation
- The idea here is to rely on StackExchange to explore complicated, diverse questions.
- Meant for zero-shot evaluation; We released the dataset with topic-aligned dev-test pairings; You can do some development work with the dev, and try to transfer performance into the comparable test set.
- Subpart oriented around the things you see in web search, and a subpart more oriented to forum queries (eg StackExchange)

XOR-TyDI
- An effort to push IR into a more multilingual setting


## Other Topics
- There's a large literature on different techniques for sampling negatives.
- Weak supervision can often create effective retrieval labels!
	- A passage is relevant in a QA context if it contains the answer as a substring anywhere in the passage.
- Santhanam et al use [[Dynaboard|Dynascore]]s to create unified leaderboards measuring diverse IR metrics, including costs, latency, and performance. We'll discuss Dynascores in detail later in the course.
