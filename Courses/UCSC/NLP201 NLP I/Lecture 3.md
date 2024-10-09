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




![[Pasted image 20241008164452.png]]
This sigma star is the set of all finite words over Sigma (Sigma here is eg a vocabulary)
So L1\* is any combination of those letters..
And Sigma^i is any words of a particular length
Note L+ is just L star but it doesn't include the empty string


FSA

little delta: Takes the cross product of Q and Sigma (eg a letter in the alphabet) and gives us a new state/
The start state is just some initial starting state in Q
F is called the final or accept states -- the set of states that, if we're in it, we accept.

L(M), the language of M, is a subset of Sigma star, which is the set of strings that M accepts.

![[Pasted image 20241008164933.png]]
We have four states (q0, q1, q2, q3)
We start at q0  as an initial state
It's har d to see, but the right and bottom ones have a double ring around them, which denote them as final states -- which mean that if it ends up in that state, it's accepted, else it's not.

The set Q is all of these states.
The transition function delta tells me where to go based on what state I'm in , and what I see.
- From q0
	- If I see a 1, I go up to q1
	- If I see a 0, I go back to q0

The way we run the machien