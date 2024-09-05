https://www.youtube.com/watch?v=Bo0IdjorIUY&list=PLWnsVgP6CzafDszSy-njjdqnliv5qT0EW&index=20
# Topic: How does in-context learning work?

---


![[Pasted image 20240606190234.png]]
In the GPT-2 paper, we were surprised to see that this was possible!

![[Pasted image 20240606190626.png]]
What parts of these demonstration are actually useful to the model (sentiment analysis example)?
- Distribution of Inputs (Show all sorts of texts that we'll be actually sentiment analyzing in the real test time)
- Label Space (show both positives and negatives)
- Format (Here, we have an input, a newline character, and an output. This is the format that we want the model to generate in. At test time, we'll give {sentence} \n {and want an answer here})
- Mapping between input-output (Definition of the actual task, showing how the inputs map to the outputs through examples)
	- It turns out that this one doesn't seem to effect the performance of the model much, especially as you increase N. This is kind of weird to hear, right? Basically says the model can already do sentiment analysis and just needs to be prompted to get that ability.

![[Pasted image 20240606191625.png]]
In the end we don't seem to have a good understanding of which prompting strategies matter most.


![[Pasted image 20240606193012.png]]
Recent paper choose a language spoken on New Guinea (Kalamang) with like 200 speakers. A field linguist went to the island and created a grammar book about how the the language works, with some examples and a small dictionary and some example translations. This is all the data we have.

When we provided it to Gemeni (?), with all of the knowledge from the book IN THE PROMPT, it was able translate!

![[Pasted image 20240606193508.png]]
With no dictionary provided, none of our models were really able to translate into Kalamang. We see that GPT4 gets significantly better (going from .24 to 2.38) when we provide it with a dictionary.

It shows that memorization alone doesn't explain the abilities of models; they're certainly able to do some level of reasoning over the information that we provide them in context.









