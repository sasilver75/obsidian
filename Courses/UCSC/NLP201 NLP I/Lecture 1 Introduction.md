
A lot of talk on "what are all of the different NLP tasks?"

Classes of tasks:
- Document classification
	- Binary of multi-label classificaiton
- Tagging
	- Each token gets a label
- Parsing
	- Produce a tree or other structure over the words in sentence
- Generation
	- Produce a sentence from some representation of the desired output
- Sequence-to-sequence
	- Sequence of tokens to another sequence of tokens of possibly different length

-----

Traditional NLP pipeline
1. Tokenization (deciding the unit of processing; usually words or subwords)
2. Morphological analysis (analyzing structure of words)
3. PoS tagging
4. Syntactic Parsing
5. Semantic Parsing (optional)
6. Downstream task: Classification/QA/summarization/etc. (using information from previous stages)
7. Generation of answer (optional)

In deep learning, in contrast, sometimes tasks are done end-to-end without intermediate steps.

---

Areas of linguistics (depending on what you consider to be the signal)
- Speech
	- Phonetics (How *humans* produce the sounds)
	- Phonology (Sounds in a language)
- Text
	- Orthography (Spelling)

For each, from shallower to deeper
- Morphology (How the words are put together)
- Lexemes (The basic lexical units of a language)
- Syntax (The arrangement of words and phrases that create a well-crafted sentence)
- Semantics (The study of meaning in language)
- Pragmatics (Language and the context in which it is used)
- Discourse (Written and spoken language and the wider social contexts)

It seemed like Jeff Flannigan was saying that these were all constrained just the document itself, not including wider social context stuff. These parenthetical are my definitions, he didn't really give good ones, especially for the latter ones.
