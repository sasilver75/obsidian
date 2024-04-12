#lecture 
Link: https://youtu.be/FFRnDRcbQQU?si=AaJEKTuA9WivCZKo
# Subject: Coreference Resolution
((==This is very not something that I'm interested in, but it's taught by Chris, so I'll give it a shot.==))

-------

![[Pasted image 20240411220115.png]]
One bug in our course design: Many years, we've had a whole lecture on doing [[Convolutional Neural Network]] for language applications. For applications in coreference, people also make use of character-level ConvNets! We'll do a quick touch on this.


# What is Coreference Resolution?
- Identify all mentions that refer to the same entity
	- Mentions don't have to be people!
		- Vanaja, Akhila, local park
		- Akhil's son
		- Prajwal
		- "her son"
		- Akash
		- "the same school"
		- pre-school play
		- Prajwal
		- ...

![[Pasted image 20240411221105.png]]

The question is: ==Which of these mentions are talking about the same entity?==
- "her" = Akhila
- ...

Chris hopes to illustrate that most of the time when we do coreference in NLP, we make it loook like the conceptual phenomenon is... kind of obvious. There's a mention of Sara, then it says "she," so we say "Aha, it's coreferent, this is easy!" But in real text, coreference resolution is very difficult (and arguable!)


# Applications of Coreference resolution
- Full Text Understanding
	- [[Information Extraction]], [[Question Answering]], [[Summarization]]
	- "He was born in 1988" -> Who was?
- Machine Translation
	- Languages have different features of gender/number/dropped pronouns, etc.
	- "Juan likes Alicia because he's smart." -> "A juan le gusta Alicia porque es intelegente"
- Dialogue Systems
	- "Book tickets to see James Bond" -> "Spectre is playing near you at 2:00 and 3:00"
		- Spectre is actually a coreference to the movie franchise of James Bond 😮



Coreference is done in two steps:
1. ==Detect Mentions in the text== (easy)
	- Types of mentions
		- Pronouns (he she it) -> Use a Part of Speech tagger!
		- Named entities (People, places, etc.) -> Use a Named Entity Recognition system
		- Noun phrases (a dog, the car) -> Use a parser, especially a constituency parser!
1. Figure out how to ==Cluster the mentions (hard)==
	- Marking all pronouns, named entities, and NPs as mentions OVER GENERATES mentions
		- "IT is sunny"
		- "Every student"
		- "No student"
		- "100 miles"
	- Maybe we train a ML classifier to filter out these spurious mentions?
	- More commonly, people keep all mentions as "candidate mentions", and then try to run your coreference system... discard all of the resulting singleton mentions afterwards.

Can we have systems (later in lecture) that do both mention detection and coreference resolution in one shot? Yeah! Later, though.

![[Pasted image 20240411223001.png]]
You can either have:
- Two mentioners referring to the same entity in the world ([[Coreference]])
- A different-but-related concept, called an [[Anaphora]]
	- The canonical case of this is pronouns 
		- In the context of this text, the "he" anaphor is textually dependent on the antecedent "Barack Obama," meaning they sort of refer to the same thing in the world, but "he" doesn't really have its own meaning.

![[Pasted image 20240411224028.png]]

But there are some more complex forms of Anaphora that aren't even coreferential!
![[Pasted image 20240411224130.png]]

Then there's this other complex case that turns up quite a bit, where the things we talk about *do have reference,* but the relation is more subtle than identity!
![[Pasted image 20240411224223.png]]
Above: "The concert" and "the tickets" are two different things, but in interpreting this sentence, what it really means is "the tickets to the concert"; there's this hidden pointer back to the coreference.

Note: The term "Anaphora" means that you're look "backwards" (ana) for your antecedent; In classical terminology, you have both Anaphora and [[Cataphora]], where you look *forward* for your antecedent.
==In modern linguistics, we usually just use Anaphora regardless of whether it's a forwards or backwards reference.==

### Takeaway
![[Pasted image 20240411224555.png]]

![[Pasted image 20240411224807.png]]


# (1/4) Rule-based models (Hobbs Algorithm)
- Pretty archaic; a relic of last millenium, but it's an interesting thing to reflect on.
![[Pasted image 20240411224950.png]]
![[Pasted image 20240411224955.png]]
Looks like a hot mess, right?
- ==This set of rules for determining coreference... were actually really good!==
- In the 1990s/2000s decade, even when people used ML system, they'd *hide* into the ML based systems that one of their features was the Hobbes algorithm! 😄
![[Pasted image 20240411225149.png]]
He's fit a lot of information about how English works into this!

What's still interesting in 2024 is what point Jerry Hobbs was trying to make with his (self-called) "naive algorithm" 
- If you want to try to crudely decode coreference... there are these preferences (recency, subject, gender, etc.

==Jerry Hobbs wasn't even a fan of his own algorithm! He wanted to argue that even something like this isn't sufficient! We have to build systems that really understand the text!==

![[Pasted image 20240411225844.png]]

==These [[Winograd]] Schema examples show that examples with the same grammatical structures can have some small ambiguities that require world knowledge!==
- You have to understand cups, pitchers, which are empty when you pour water!

# (2/4) Mention-pair and Mention-ranking models (and Clustering methods, which we're skipping)
- A classic ML technique


# (3/4) Interlude: ConvNets for language (sequences)



# (4/4) Current SoTA Neural Coreference systems




