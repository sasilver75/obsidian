We'll be covering for the next few lectures

- Finite State Automata (FSAs)
- Regular Languages
- Finite State Transducers (FSTs)
- Later: Weighted FSAs and FSTs (WFSAs and WFSTs)


Why learn FSAs?
- Other models (HMMs, n-gram LMs, etc) are a special cases of WFSAs
- FSA and FSTs are used model morphology; they're the first step in the formal language hierarchy.
- Useful for understanding the modeling capacity of deep NNs.


When we say a ==language==, there are two ways to give that meaning:
- Natural languages: Learned without explicit instruction by children from speakers in their environments (eg English, Hindi)
	- Natural languages link sounds or visual information (==linguistic forms==) with conceptual/intentional representations (==linguistic functions==)

Native speakers have a ==Grammar==, which is an (infinite) set of sentences that they accept as being "well-formed".
- ==A sentence can either be in the speaker's grammar, or not in the speaker's grammar.==
	- This is a personal thing (eg Ebonics, etc.)
	- In reality, it's a little but fuzzy as to whether you think something is "well formed," etc.
	- Linguistics use stars \* to indicate ungrammatical:
		- \*"They says 'hello'"
		- They say "hello"
	- Note that this isn't about "English Grammar" (prescriptive grammar, etc.). Linguists tend to focus on *descriptive grammar*, where we ask native speakers: "what do you accept as being grammatical or not?" We're asking them, instead of telling them.

Formal Language

Alphabet: The symbols in the lagnuage
Word: A string of symbols

For now let's not worry about spaces. 

Could also use the terms:
Vocabulary: The symbols in a language
Sentence: A string of symbols, separated by spaces



