#lecture 
Link: https://www.youtube.com/watch?v=PSGIodTN3KE&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=4

Focusing much more on linguistics and NLP in this lecture, looking at the topic of dependency parsing


Agenda:
- Syntactic structure and dependency parsing
	1. Syntactic Structure: Consistency and Dependency
	2. Dependency Grammar and Treebanks
	3. Transition-based dependency parsing
	4. Neural dependency parsing

----

How do people put structure on sentences to highlight how human language contains meaning?


![[Pasted image 20240402120717.png]]
Above:
- Word are obviously an important part of human languages... but there's more than we can do with them when think about the structure of sentences.
- The most basic way we think about words are their ==part of speech==
	- Cat is a noun
	- By is a preposition
	- Door is another noun
	- Cuddly is an adjective
- For the word "the" -- if was as given a different PoS, you might have been told it was an "article," or something. In modern linguistics, these are referred to as ==determiners==, along with *this* and *that*.

It's not the case that when we want to communicate about language that we just have a word salad where we have a bunch of words, and let the other person put it together
- We put words together in a particular way to express units
- So language has these larger units than words that we construct

In modern work, in modern US linguistics, or what you see in CS with formal languages, the most common way to approach this is with ==Context-Free Grammars ([[Context-Free Grammar|CFG]]s)==.
- There are bigger units like phrases
	- The cuddly cat (a noun phrase)
	- by the door ("the door" is a noun phrase, and we build something bigger around it with a preposition, into a prepositional phrase)
- Phrases can combine into bigger phrases
	- the cuddly cat by the door
		- The door is a noun phase
		- the cuddly cat is a noun phrase
		- by the door is a prepositional phrase
		- the whole sentence is a noun phrase
	- These smaller phrases contained within are called ==non-terminal phrases==

Two views of linguistic structure:
- Constituency: Phrase structure
- Grammar: Context-Free Grammars (CFG)

![[Pasted image 20240402121705.png]]
Above:
- Showing some examples for how the sentence can be modified


A somewhat different way of looking at grammar is ==dependency grammar/structure==
![[Pasted image 20240402122333.png]]
Above:
- In modern NLP starting in ~2000 or so, NLP people have really swung behind dependency grammars. ==By far the most common thing being used is dependency grammars.==
- The idea of [[Dependency Grammar]]s is that, for each word, what other words modify it?
	- Large modifies Crate
	- The modifies Crate
	- The modifies Kitchen
	- The modifies Door
- The representations we look at at those that follow *Universal Dependencies,* which was an attempt to make a dependency grammar that works across many languages. Chris Manning helped with this!
	- Many other languages make use of case-markings (genitive, dative cases) that cover most of the role of prepositions


![[Pasted image 20240402123009.png]]
- ((Semantics are sparsely represented among words; the meaning of a sentence, and the words in a sentence, needs context))
- The listener just gets a sentence one-word-after-another. The listener has to be able to figure out which words modify other words and construct the structure (and thus the meaning) of the sentence.
	- If we want to build NN models that understand the meaning of sentences, those NN models must necessarily also understand the structure of the sentence to interpret the sentence correctly!

The choices that we make re: building up the structure of language impact the meaning of sentences!
![[Pasted image 20240402123200.png]]
Above:
- Did the cops have knives, and use them to kill a man? Or did the man that was killed have a knife?

![[Pasted image 20240402124212.png]]
Above:
- Are the whales from space? :) 

Many sentences in english have prepositional sentences all over the place:
![[Pasted image 20240402124300.png]]
If you look at the structure of this sentence: You notice that there's this like... telescope of prepositional phrases. 
- There are some rules for how you attach them
	- "Things to the left, providing you don't create crossing attachments."

![[Pasted image 20240402124520.png]]

Note:
- The grammars used in programming languages are designed to be unambiguous, or where they're ambiguous there are rules where the compiler chooses one specific parse tree. 
- Human languages aren't like this -- we hope that we're just smart enough to pick out what was intended.

![[Pasted image 20240402125012.png]]
Above:
- Ambiguity in language

![[Pasted image 20240402125136.png]]

![[Pasted image 20240402125230.png]]

Above:
Syntactic ambiguities in terms of which things modify other things


More formally, what is a dependency garmmar?
- Consists of relations between pairs of words
	- It's a ==binary, asymmetric relation== -- we can draw directed arrows between words, which we call dependencies.
	- These arrows are commonly ==typed== to express the name of the grammatical relation
		- subject
		- prepositional object
		- apposition
		- etc.

Usually, dependencies form a tree (a connected, acyclic, single-root graph)

![[Pasted image 20240402125913.png]]


![[Pasted image 20240402130053.png]]

This idea of constituency grammars/context-free grammars are a pretty recent invention (~20th century)
- For modern work on dependency grammar... using the terminology and notation that we're just introduced, was introduced by this Lucien Tesniére fellow.
- Dependency Grammar widely used in the 20th century in many places. Natural to think about for languages that have a lot of different case markings for nouns like you'd get in a language like latin or Russian -- many of those language have much freer word order than English does, and instead use different forms of nouns to show you what the object of the sentence is.


A dataset of such dependency tree parsed sentences generated by human annotators is called a ==Tree Bank==

![[Pasted image 20240402130632.png]]

It's not seen as near-axiomatic that you need annotated data to make progress.
This effort to have a common (universal) dependency annotation (as introduced by Manning and others) is thus a good idea.

Why are Tree Banks good?
- It seems like it's bad news if you have to have people sitting around for weeks/months to hand-parse sentences, right?
- In practice, there's just a lot more that you can do with a tree bank -- Tree banks are great because:
	- Highly reusable
	- Typically, grammars are written for one particular parser, and are only used in that parser.
	- In contrast, tree banks are highly reusable resources that can be used by hundreds of researchers.

![[Pasted image 20240402131003.png]]
If you have an unbiased set of sentences in your treebank, you have to understand/parse *all* of language, even the annoying edge cases!

![[Pasted image 20240402131225.png]]

![[Pasted image 20240402131624.png]]So how do we build a parser?
- We have a sentence
	- For every word in the sentence, we have to choose another word that it's a dependent of.
	- We don't want cycles!

![[Pasted image 20240402131805.png]]


How do we go about building dependency parsers?
![[Pasted image 20240402132048.png]]

![[Pasted image 20240402132208.png]]

Details of above:
![[Pasted image 20240402132250.png]]
- start
	- We have a starting point with a stack with a root symbol on it
	- A buffer with a sentence that we're about to parse
	- So far we haven't built any dependency arcs
- At each timepoint choose one of three actions
	- Shift: moves a word onto the stack
	- Left-Arc (a reduce action): Take the top two items on the stack, and make one the dependent of the other one
	- Right-Arc (a reduce action):
- If one of the two reduce actions are chosen, the one that's the dependent disappears from the stack, and we add a dependency arc to our Arc set -- so we say we have a dependency from j -> i or i -> j
	- We commonly also specify the grammatical relationship that connects the two -- that's the relation r(wj, wi)


![[Pasted image 20240402132819.png]]


Honestly this shit is boring but I'm glad I'm putting eyes on it once





