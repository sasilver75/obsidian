

Common tokenization strategies include:

==Word-level tokenization== (Space-and-punctuation tokenization)
- Usually generates a large vocabulary; [[Transformer-XL]] uses space and punctuation tokenization, resulting in a vocabulary size of 267,735. Such a big vocabulary forces the model to have an enormous embedding matrix as the input and output layer, causing both increased memory and time complexity. In general, Transformers rarely have vocabularies greater than 50,000, especially when pretrained on a single language.
- I believe it's prone to have UNKs if those words aren't observed in the dataset.

==Subword-level tokenization==
- Subword tokenization algorithms rely on the principle that *frequently used* worlds shouldn't be split into smaller subwords, but *rarer* words should be decomposed into meaningful subwords.
	- "annoyingly" might be decomposed into "annoying" and "ly". Both of these two tokens appear more frequently, and the meaning of "annoyingly" are kept by the composite meaning of "annoying" and "ly."
- Allows the model to have a reasonable vocabulary size while still being able to learn meaningful context-independent representations for tokens. 
- It allows the model to generalize to process words it's never seen before, by composing them into known subwords. More effective in handling the complexities of natural language, especially for advanced models and diverse datsets.
- Has fewer UNKs than word-level tokenization. 

==Character-level tokenization==
- Very simple, and would reduce memory complexity, but results in sequences being much larger, slowing inference speeds. It also makes it harder for the model to learn meaningful input representation.
- Particularly useful for languages without clear word boundaries, like Japanese and Chinese, or for specialized applications requiring more-granular levels of analysis, like spelling correction. 
- Has fewer UNKs than word-level tokenization

==Byte-level tokenization==
- I believe GPT-2 used Byte-level tokenization

![[Pasted image 20240530114310.png]]


![[Pasted image 20240623222437.png]]
Above: The steps of a tokenization pipeline, generally
- Normalization involves general cleanup, such as removing needless whitespace, lowercasing, removing accents, and (eg) unicode normalization.
- Pre-tokenization involves splitting the text into smaller entities, like words (which will later be the boundaries of the subtokens the tokenizer can learn during its training)