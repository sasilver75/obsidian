# Topic: Knowledge Bases and Language Models
https://www.youtube.com/watch?v=IwEYCbdgJ9U&list=PL8PYTP1V4I8DZprnWryM4nR8IZl1ZXDjg&index=17

This look like a promising lecture!

----

==Knowledge Bases== are strutured databases of knowledge, often containing:
- Entities (nodes in a graph)
- Relations (edges between nodes)

1. How can we learn to create and expand knowledge bases with neural network-based methods?
2. How can we *learn* from information in knowledge bases to improve neural representations?
3. How can we use structured knowledge to answer questions?

---

Types of Knowledge Bases
- They come in several different varities!

A classical one is called WordNet (Miller 1995)
- Basically, this used to be a really big thing in NLP, but it's not so much any more.
- It's a large database of words, including parts of speech and semantic relations.
	- Specifically, each word (or something called a synset (?)) is a node, and there are relationships between nodes (can be nouns, verbs, adjectives).
![[Pasted image 20240617165933.png]]
Nouns have different types of relations:
- Is-A relations
- Part-of relations
- Types and instances (Joe Biden is an instance of President)

WordNet was used for trying to do things like figure out all of the cars identified in a series of documents.

But why don't we use WordNet very much anymore?
- Probably because we use contextual representations with dense vectors; we could find all of the things close to it in embedding space.
- We might just download Mistral and say "Find all the cars mentioned in this sentence"; it's expensive, but really easy!


Cyc was a manually-curated database from 1995 that was a manually curated database attempting to encode all common sense knowledge!
![[Pasted image 20240617172132.png]]
Unfortunately, it was just too ambitious of a project -- it got part of the way there, but that part of the way there wasn't enough to be useful in practical systems.

DBPedia (2007) was a followup; the basic idea is that while Cyc is too difficult because they had people on the Cyc project go in and curate rules for machines... instead, places like Wikipedia have a large number of humans curating the structured data in the world *for humans.*
![[Pasted image 20240617172306.png]]

Now, people use something called WikiData (2008)
- Its name is a misnomer because the data doesn't only come from Wikipedia.
- It's a curated database of entities, linked, and is extremely large scale, and is multilingual.
- ![[Pasted image 20240617172358.png]]
https://www.wikidata.org/wiki/Wikidata:Main_Page
Another good thing about this that we didn't mention directly in the lecture notes... is that there's a query language (SPAQL) for it (SPARQL is a language for querying knowledge bases and other things)

But even with extremely large scale, knowledge bases are by their nature incomplete.
- So can we perform ==relation extraction== to extract information from unstructured/structured text to put into our knowledge base?

There are a bunch of ways that people do this:
- Leveraging consistency in embedding space... we can learn embeddings from text, or we can use the fact that we have a knowledge base curated by humans to improve the embeddings of a model itself!
	- How do we get good embeddings of a knowledge graph? This is important if we want to do any sort of knowledge graph search, for instance!

Learning Knkowledge Graph Embeddings (2013)
- Motivation: Express KG triples as additive transformation, and minimize the distance of existing triples with a margin-based loss. (l in the equation is r in the diagram)
- ![[Pasted image 20240617173429.png]]

Relation Extraction with Neural Tensor Networks (Socher et al 2013)
- A first attempt at predicting relations: a multi-layer perceptron just predicts whether a relationship exists
- ![[Pasted image 20240617173638.png]]
Powerful model, but perhaps overparametrized!


## Learning from Text Directly
- Another method for relation extraction is just to learn from text directly!

But where do we get training data to learn about relationship extraction?

Distant Supervision for Relation Extraction (Mintz et al 2009)
- One of the first/most influential papers on synthetic data augmentation for NLP; the idea is that you already have a knowledge base with some entries in it (eg WikiData)... and given an (entity-relation-entity) triple, can you extract all text that matches this relation type, and use it to train?
![[Pasted image 20240617174033.png]]
Given the first sentence, it finds all sentences finding SS and SPR in them, and labeling that as a positive relation.
- In general, this is okay/works reasonably well, but there are also negative examples of it! It's easy to accidentally create noisy examples.

Relation Classification with NEural Networks (Zeng et al 2014)
- Most of thes methods work by extracting features and then classifying, somehow.
- One thing about erlationship extraction/information extraction in general, is that veryo ften you want to run it over a huge corpus. From that point of view... if you want to run GPT4 over the whole intenret, you might want to reconsider that -- there's a benefit to having cheap and lightweight methods!
- This paper extracted lexical features of entities themselves, and features of the whole span, and then classified.
- ![[Pasted image 20240617174245.png]]


![[Pasted image 20240617174639.png]]

