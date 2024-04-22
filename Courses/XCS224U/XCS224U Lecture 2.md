#lecture 
Link: [Lecture](https://www.youtube.com/watch?v=J52Dtu40esQ&list=PLoROMvodv4rOwvldxftJTmoR3kRcWkJBp&index=2)
Course Page: https://web.stanford.edu/class/cs224u/
- Mentions that there's a CS224U podcast

-------

Reviewing some things from last time
- We tried to immerse ourselves in this moment of AI and think about how we got here

The first two units of the course
- Transformers
- Retrieval-Augmented In-Context-Learning

But there is more to our field than just big language models and prompting, and many ways to contribute!

The third course unit: Compositional Generation
- The COGS benchmark
- A relatively new synthetic dataset trying to see whether LMs have systematic solutions to Language problems

![[Pasted image 20240422114600.png]]
It's basically a parsing problem  ((Not interested!))
Given a sentence, map it to the logical form (Lina gave the bottle -> * bottle ( x _ 3); give ...)

In this class, we'll work with an improved dataset, ReCOGS, to create a new dataset that we think is fairer -- but it remains incredibly hard. We have more confidence that it's testing something about semantics -- but it's still incredibly hard for our best systems -- there needs to be a breakthrough for our systems to do well, even on these incredibly simple sentences.

Some topics that we hope will improve our final probjects:

The first topic
- Better and more diverse benchmark tasks
	- We need measurement instruments to understand how well our systems are doing
	- "Water and air, the two essential fluids on which all life depends, have become global garbage cans" - Jacques Cousteau

==We ask a lot of our datasets:==
1. To optimize models
2. To evaluate models
3. To compare models
4. To enable new capabilities in models
5. To measure field-wide progress
6. For basic scientific inquiry into language and the world.



![[Pasted image 20240422115236.png]]
Above: benchmark saturation
Partially, what we're seeing is that our evaluations were really about "machine tasks"
- So we had humans do these "machine tasks" to get a benchmark of how good humans are at these machine tasks
Perhaps we need to change these.

Reference to: [[Dynabench]]
Reference to [[Dynaboard]]

![[Pasted image 20240422115547.png]]
^ Sort of reminds me of [[HELM]]


----

# Guiding Ideas

Static Vector Representations of Words
- Back in the old days, the way we represented examples was with feature-based sparse representations
	- Feature function ("Yes or no, referring to animate thing," "Yes or no ending in chracters -ing"), etc.

Other things like [[TF-IDF]]: Instead of writing feature functions, I'll just keep track of co-occurrence in large bodies of text.



















