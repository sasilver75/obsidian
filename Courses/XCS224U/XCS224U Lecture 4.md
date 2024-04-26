#lecture 
Link: [Information Retrieval Lecture](https://youtu.be/enRb6fp5_hw?si=HReeZswa891GA5Mb)
Guest speaker: [[Omar Khattab]]

----

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


# Retrieval-Augmented In-Context Learning

Q: Who is Bert?

![[Pasted image 20240425210019.png|250]]
How can we effectively answer this question using Retrieval?
- We can retrieve from a document store a context passage for the question and inject it into the prompt
	- Context: Bert is a muppet who lives with Ernie
- We can add other demonstrations (eg [[Few-Shot Prompting]]) to the prompt

We'll see that this is just the start of a very rich set of options that we can employ to effectively develop in-context-learning systems to process/develop evidence.

# IR is more important than ever!
- We see a lot of worrisome behavior from LLMs that are deployed as part of search technologies; Google got a real hit from making a minor factual error in one of its demo videos; maybe that's appropriate!
	- (At the same time, OpenAI models fabricate evidence all over the place!)


---

# Classical Information Retrieval
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

TF-IDF values are at their lowest for words that 
























