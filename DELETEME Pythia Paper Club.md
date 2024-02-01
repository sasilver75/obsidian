
They trained 16 LLMs on the Pile
Reproducibility: 154 checkpoints
Gender Bias: Last 7% and 21%... bias


Architecture and training:
- Use fully dense layers 

Results:
- Deduplication - No effect
	- Deduplication of training data has no clear benefit on language modeling performance.
	- They had 300B tokens from the pile; they deduplication it to 207B, and ran for 1.5 epochs on the smaller one (so the total was 300B in either case), and it didn't make a difference.
		- Matches up with the self-instruct paper from last week; including data that isn't high quality still improves model performance.
		- Swyx: Surprising that people still don't know what the effect of duplication is. I'd refer to the ==datablations== paper, where they say that up to ~4 epochs on the same data is fine. It seems the meta is shifting from 1 epoch on a dataset to 1-4 epochs on the dataset, and we don't know why. ==Maybe we can repeat data without suffering so much of a loss==.
	- People are moving towards inference-optimal training
- Task performance over size
	- ...
- No memorization
	- They had an interesting way of measuring memorizing that seeing whether a Poisson model fit the data in some way.
- Gender Bias
	- Swapped stereotypically male with female dataset to reduce the gender bias of the models that were trained, only at the end.
	- It's interesting that you can improve your gender bias metrics with this kind of simple intervention of just swapping male pronouns to female ones.
	- ![[Pasted image 20240131121429.png]]

That's pretty much it, the main part of it is 
They measured task performance over size; basically large models have more emergence as you train over the same amount of data. They controlled everything else, very scientifically. 

![[Pasted image 20240131121712.png]]
Techniques that they perceived as SoTA for model training
- The rotary embeddings paper is very mathematical :(

![[Pasted image 20240131121906.png]]
Papers before Pythia didn't have the goal of reproducibility
Collects the SoTA techniques that they think have been unresolved questions (sparse vs dense attention, flashattention, rotary embeddings, ...)

Paper sets the bar for what open research should look like