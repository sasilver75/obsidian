August 11, 2023 (<1 month after LLaMA 2)
[[Meta AI Research]]
Paper: [Self-Alignment with Instruction Backtranslation](https://arxiv.org/abs/2308.06259)
#zotero 
Takeaway: A method/model that applies the **==instruction backtranslation==** method, using a fine-tuned language model and web corpus to:
- Construct instruction prompts for web documents (**==self-augmentation==**)
- Then select high-quality examples from these generations (==**self-curation**==) to train the next iteration of the model.
This is applied iteratively in a self-alignment manner. On the Alpaca leaderboard, it sounds like this model (Humpback, 7B/33B/65B LLaMA 1 finetunes) outperforms all other *non-distilled* instruction-following models (Non-distilled models don't rely on a stronger LM to teach them to be good models).

Relevance: Is contrast to Distillation, Self-alignment methods don't rely on having a stronger model present to improve the current model. In contrast to other methods that might generating their own synthetic instructions *and/or responses*, this model simply generates instructions for already-existing human-written text "responses," and incorporates a discrimination component to filter out poor s

Aside: 
- The [[Self-Reward]]ing Language Models paper had critical things to say (see Obsidian references) about the LLM-as-a-Judge prompt used in this paper, where we had the model pick a score in a multiple-choice fashion, rather than using their explicit "add a point" criteria.
	- Yeah, this paper says the rating is "done using prompting, instructing the trained model to rate the quality of a candidate pair of a 5 point scale." It *does* give descriptions of what each should look like.
- I do like the "related work" section of this paper, which names a lot of good papers in Instruction tuning for LLMs, Instruction generation and curation, Self-alignment, Data quality, and Distillation.
- I can sort of see how papers like this inspired Nous Research's [[Genstruct]] model, and the [[Evol-Amplify]] technique from Capybara.
- In some way similar to the [[Self-Reward]]ing language model paper, in the sense that the model has two functions (in this case, generation and discrimination of good examples).
- In some ways similar to [[Web Rephrase Augmented Pre-training|WRAP]], in that it leverages a large corpus of text data as "inspiration." WRAP rephrases it, whereas Instruction Backtranslation augments it by using it as the instruction response and generating an appropriate instruction.
- Very similar to [[Longform]] (concurrent work), which takes human-written text as a natural response, and uses LLMs to generate corresponding instruction conditioning. The difference is that Instruction Backtranslation shows that the *self-curation* step is vital to improve such a procedure. Additionally, Longform uses distillation via an instruction-tuned LLM (InstructGPT) to generate instructions, whereas SAIB is an instance of self-alignment, not relying on a more powerful model.
- Similar to [[Alpagasus]], which also provided an algorithmic approach to selecting high-quality data, except they prompt a stronger model (ChatGPT) to score the quality of *model generations*, while our work scores the quality of *human-written data* as a response to a self-generated instruction.
- It seems like this model would only generate single-turn examples of (x, y)... which isn't amazing. I'm also interested in the divers

hu-po on Youtube said that the process for this model seemed pretty similar to what [[Segment Anything Model]] (SAM) does for training.

----

Note:
- Inspired by the classic backtranslation method from [[Machine Translation]]
	- A data augmentation technique in which, given an input sentence in (eg) English, we translate it to a temporary destination language (Chinese), and then translate *back* to the source language, English. 
- Our method starts with a seed instruction-following model (base model of LLaMA 7,33,65Bs)  finetuned on some seed instructions (3200 instructions from [[oasst1]])) and a web corpus. The model is used to *==self-augment==* the training set by, for each web document, creating an instruction-following training example by **predicting a prompt (instruction) that would be correctly answered by a (portion of) that document.**
	- Directly training on such data gives poor results in our experiments, because of the mixed quality of human-written web text, and noise in the generated instructions.
- Thus, we then ==*self-curate*== the set of newly-crated augmentation data **using the same seed model** by predicting their quality, so that, after filtering, we can train the next iteration of our model only the highest-equality `(instruction, output)` pairs. (This, and the previous step, is performed iteratively)
	- Note: The authors choose to train M_2 from M_1, rather than training all models from M_0.
	- ((I  don't see that it's obvious that a better instruction-following model would be a better discriminator of what good instructions are -- I wonder what sort of technique they're using?))
- Key Assumptions
	- That there exists some subset of a large human-written text that would be suitable as gold generations for some user instructions.
	- That we can predict instructions for these candidate gold answers that can be used as high-quality example pairs to train an instruction-following model.
- Self-Augmentation
	- We run inference on the unlabeled examples $y_i$ to generate a candidate instruction $\hat{x_i}$. We now have a bunch of {($\hat{x_i}$, $y_i$)} pairs, but many of them are probably not of high quality.
	- It seems that they choose to use [[Top-P Sampling|Nucleus Sampling]] (Top-P Sampling) for generation
- Self-Curation
	- We further propose an iterative training method to produce higher-quality predicitons.
	- On iteration *t*, we finetune our model on the curated augmentation data from the previous iteration. This model in turn is used to rescore the augmented examples for quality, resulting in a new augmentation set.
	- We perform two iterations of data selection and finetuning to get the final model $M_2$. 
	- When combining both seed data and augmented data for finetuning, we use tagging to distinguish these two data sources. 
		- We append an addition sentence to the system prompt of "Answer in the style of an AI Assistant" for seed data, and "Answer with knowledge from web search" for augmented data. ((I don't really understand the reasoning behind this.))
	- Successive models can be used to ==*rescore*== augmented examples for quality, resulting in new augmentation sets.
- Authors call the resulting model ==Humpback==.
- As is shown in Figure 5, when training on self-augmented data alone (without seed data), and without self-curation, the quality of instruction following does not improve, or even deteriorates with more data. However, training on the higher quality self-curated data brings improvements as training set size increases
- 

Abstract
> We present a ==scalable method to build a high quality instruction following language model by automatically labelling human-written text with corresponding instructions==. Our approach, named ==instruction backtranslation==, starts with a language model finetuned on a small amount of seed data, and a given web corpus. ==The seed model is used to construct training examples by generating instruction prompts for web documents== (==self-augmentation==), and ==then selecting high quality examples== from among these candidates (==self-curation==). This data is ==then used to finetune a stronger model==. Finetuning LLaMa on two iterations of our approach yields a model that outperforms all other LLaMa-based models on the Alpaca leaderboard not relying on distillation data, demonstrating highly effective self-alignment.


# Paper Figures
![[Pasted image 20240509155923.png]]
Above: The process 🔄 

![[Pasted image 20240509163352.png]]
Above: Interestingly, it seems that they're claiming that the dataset produced for Humpback in this paper is more efficient training fodder than the data created via [[Evol-Instruct]] in [[WizardLM]] (and many others).

![[Pasted image 20240509163651.png]]
Above: This is kind of an interesting figure. If you look at the last two rows, it shows that the system prompt being present during inference doesn't really matter, as long as it was present during the pretraining? Huh, never really thought about that.

![[Pasted image 20240509170634.png]]
Above: It seems like performance was continuing to improve even at $A_5$ (the fifth generation)

![[Pasted image 20240509171047.png|300]]
Some examples of $y$ text "responses", and $\hat{x}$  generated instructions for them.

![[Pasted image 20240509172704.png]]
Above: Prompt used in the ==Self-Curation== step. See note at the top that the people from the [[Self-Reward]]ing LM paper were negging this LLM-as-a-Judge prompt. See that this one is basically a multiple choice between 5 different options. It *does* request that you write  a brief reasoning, though. Annoyingly, it doesn't seem like they share the prompt they used for the Self-Augmentation step.




# Non-Paper Figures

- 