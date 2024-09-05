https://www.youtube.com/watch?v=lSBG_JuhbPE&list=PLWnsVgP6CzafDszSy-njjdqnliv5qT0EW&index=17

# Topic: Scaling Laws for Large Language Models

----

What do bigger LMs buy us?
- Broadly, *emergent properties!* Those that only appear iwth larger LMs, but not smaller ones. Including:
	- [[In-Context Learning]]/few-shot prompting
	- [[Chain of Thought]] prompting
	- Instruction following
	- More memorized knowledge and patterns from training data

Scaling is not only associated with the number of parameters in the model, but we can pump more individual tokens/data into it, or directly more compute into it (by keeping the data and parameters constant, but doing more epochs).

![[Pasted image 20240605145242.png]]
Above: Seems to show the emergence of certain abilities as we increase the compute budget (measured in floating point operations)

![[Pasted image 20240605155259.png]]

If you can use 1 GPU for 1 day, would you:
1. Tran a 5M parameter LM on 100 books?
2. Or a 500M parameter LM on 1 book?
3. Or 100k parameter LM on 5000 books?

This is an important question for research labs to answer!
- If we wanted to empirically measure this, we'd have to train a huge number of modeles, varying dataset and parameter size, and fit some sort of curve(s) to determine what the ratio would be.

Luckily, both OpenAI and Google did experiments, and open-sourced their results!

OpenAI's 2020 paper on *Scaling Laws for Large Language Models (Kaplan et al)*
![[Pasted image 20240605155444.png|300]]
- The performance of the model depends strongly on scale, but less so on the shape (architecture, attention heads, etc.), within reason.
- The performance can be modeled as a function of scale.
- Performance improves MOST when you scale up BOTH model size and dataset size (so you can't just get the same performance by having a huge model with very little data, or vice versa)
- Larger models are more sample efficient!

![[Pasted image 20240605155816.png]]
Above:
- Each of these curves correspond to a different models' training run. You can see that the larger models improve sooner and MORE than smaller models!
- Note that after a certain point, the model stops benefitting (as much) from additional data. See that the larger models continue benefitting as dataset size passes the point at which smaller models "saturate"/"plateau"; Only larger models can benefit from huge numbers of training tokens, it seems.\
	- ((Curious how we square this with LLaMA3-7B being trained on 15T tokens))

![[Pasted image 20240605160155.png]]


----

Later, Google did a paper called [[Chinchilla]]

![[Pasted image 20240605160402.png]]

(More notes added to the Chinchilla note in Obsidian, for the rest of the lecture.)

