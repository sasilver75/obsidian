
Morphemes
- Recall: A morpheme is a minimal meaning-bearing unit of language.
	- To small (in English) to be a morpheme: 'p'
	- Too big (in English) to be a morpheme: 'processing'
- In some languages, Chinese words and morphemes are basically the same.
- In some languages, most words are made of several morphemes.
- English is in the middle.


A ==Gloss== in linguistics is a series of brief explanations about the morphological analysis for each word...
It's a little visual diagram tool or something for understanding sentence structures.


Transparency of word-internal boundaries:
- Is it easy to tell the boundaries of morphemes in a word?
- In an ==Isolating Language==: There's a one-to-one correspondence between morphemes and words (eg Chinese)
- In ==Agglutinating Languages==, a word may consist of several morphemes, but boundaries and quite clearcut
- ==Fusional languages==, there's no clear boundary between morphemes -- semantically distinct features are merged into  a single morpheme. (Apache example) (English, maybe?)


AnaInternal complexity of words:
- ==Analytic Language==: One morpheme per word (same as isolating)
	- Actually more like "close to one morpheme per word.... this is a scale -- extreme analytic would be Isolating"
- ==Synthetic language==: More than 1 morpheme per word, but a small number (eg English)
- ==Polysynthetic Language== (very large number of words per morpheme allowed)


CLAIM:
- Morphology in human languages is roughly FINITE STATE
	-- There have been successes in modeling morphology in languages like Turkish and Finnish.
	Some difficult phenomena
	- Reduplication
	- Circumfixation
	- Root and Template morphology


Let's look at representing some morphology with finite state machines.

Adjectives become opposites, comparatives, superlatives, adverbs, etc.

cool, cooler, coolest, cooly ... We want to handle this phenomenon using finite state methods
red, redder, reddest, etc.



Finite STate Transducer
- An automaton that works with two tapes at the same time. An input tape and an output tape.
- FSMs only have one tape -- we could envision them as acceptors/recognizers, or as generators.....
- This languages is a "string-pair" languages.
- FSTs can be understood as reading or writing either or both tapes!
	- Recognizer: Take a pair of strings and accept if the pair is in the string-pair language; reject if not.
	- Generator: Outputs pairs of strings
	- Translator: Read one string and write out a string. (This is how  we'll use FSTs for morpholoogical parsing)
	- Set Relator: Compute relations between sets.


Example FFST:
in: aab
out: aab

in: bbaa
out:aaaa

(unsure about this one, not clear from the board)
in {bba, aaa, baa}
out: aaa

Let's now talk about FSTs and ==Regular Relations==\- A relation between two sets is regular if it can be represented as an FST.
- Operation: Projection: We extract only the input side or the output side.
	- The result of a projection is an FSA!



