
Morphology and FSTs

This is a linguistics talk, so he's going to teach us how words 

Morphology background 
- Finite estate transducers (FSTs)
	- FSMs take a string and either accept or reject it, according to a language.
	- FST takes a string and produces another string. You can think of it as ... it recognizes pairs of strings.
- Applications of FSTs to morphology

Historically all morphological analysis was done using FSTs

Morpohlogy
- The study of words and how they're formed. Words are made from building blocks called morphemes.

In the intro to the course, we explained the difference between the different areas of the linguistics hierarchy... and the traditional linguistics pipeline, this would be the first level of that pipeline.

Morpheme
- A minimal, meaning-bearing unit of language.
	- too small: p
	- Too big: processing
- In some languages (eg Chinese), words and morphemes are basically the same.
- In some languages (Czech, Turkish), most words are made of several morphemes.

English is somewhere in he middle, where we have some morphological complication, but not as much as other languages.

Why morphology?
- Needed for further processing -- might wnat to map words to lemmas (eg banks to bank)
- Some languages have very complex morphology, where the number of words is huge for these languages!
- In order to build applications, we need to analyze these words to their morphemes
- Language with complex morphology often have very little data -- you have to build morphological analyzers by hand (with FSTs). (because they're often pretty regular)

Morphemes are minimal, meaning-bearing units of language.
A ==lemma== is a kind of ==morpheme==.
Every lemma is a morpheme, but there are morphemes like "ing" which aren't lemmas.


Linguistic form: Words, sounds
Linguistic functions: Meaning

There's a map between what I'm saying and what I'm meaning.
There's a correspondence between the two... 

Morphemes can come in different types
- A free morpheme
	- Can occur on its own; it may be bound to something else, but it can occur on its own
	- bank, duck, open
	- Can have ones that are "open class" (the ones above, like car, spork)
	- On the other hand, you can have "closed class" morphemes that can occur by themselves... these are called Function Words -- and, or the, a, this, from, to, more, less. There's kind of a closed class of what these are -- it's hard to make up a new preposition.
- A bound morpheme
	- Cannot occur on its own.
		- reevaluate. You can think of evaluate as by itself being okay... and reevaluate means that we're going to redo it. The "re" is adding the "to do it again" meaning, and it's regular. You can "redo", "replay," "rerun." "un" is another example of it.
		- Includes prefixes suffixes (under the category of Affixes)


Phonemes are often written in some sort of linguistic notation




==Inflectional morphology== keeps the same class (noun satays noun, verb stays verb)
- Simple conjugation, like run -> ran


==Derivational morphology== can do thingsl ike turn a noun to a verb. 
- derive is a verb
- derivation is a noun
- derivational is an adjective


Compounding
- baseball, desktop

Cliticization
- I am is usually written as "I'm". The way it's pronounced changes. It's an I plus an 'm Clitic.
- Do not = Don't = Do + "n't" clitic. 
- Some languages have this a lot 

