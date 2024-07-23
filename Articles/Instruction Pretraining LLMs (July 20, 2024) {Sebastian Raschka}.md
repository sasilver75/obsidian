https://magazine.sebastianraschka.com/p/instruction-pretraining-llms?utm_campaign=email-half-post&r=764e6&utm_source=substack&utm_medium=email

---

In the last month, there have been a few interesting announcements:
- Apple integration of on-device LLMs
- Nvidia's [[Nemotron-4]] model family
- [[FlashAttention]] 3
- Google's [[Gemma 2]]
- More!

Let's cover some stuff


## Generating an Instruction Dataset from Nothing (Magpie)
- The [[Magpie]]: *Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing* paper highlights a practical exploit that seems super useful.
![[Pasted image 20240722232225.png|400]]
- As shown above, we just prompt *LLaMA 3 8B Instruct* with a pre-query template, and it generates an instruction for us. Then we feed the instruction back into the LLM, and it will generate a response. We do this a few thousands of times to get a dataset for instruction finetuning!
- It's strange that, with the resulting dataset, ==authors finetuned a LLaMA 3 8B *base* model with just IFT (no preference tuning) and found that it beats the original LLaMA 2 8B Instruct model ==! 
![[Pasted image 20240722234604.png|500]]
See that Magpie-Pro (distilled using Magpie from LLaMA 3 70B, finetuned LLAMA 3 8B) handily beats the LLaMA 3 Instruct.

The Magpie results shown in the figure above wer achieved with ==300 thousand samples only. In comparison, The original Llama 3 Instruct model was finetuned and aligned on 100 million samples!==

Sebastian has a [notebook here](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/05_dataset-generation/llama3-ollama.ipynb) implementing the generation process.

Authors created a "Pro" and "Air" version of the dataset, using either a 70B or 8B LLaMA Instruct model. The Pro dataset results in slightly stronger models compared to Air.

![[Pasted image 20240723003222.png|400]]
Interesting to see that the ratios are pretty similar for both.

Furthermore, the paper contains an analysis showing that the ==breadth or diversity in this dataset is much larger than that of other popular datasets for instruction finetuning==, such as Alpaca, Evol Instruct, and UltraChat.


-----

## Instruction Pretraining LLMs

- The *Instruction Pre-Training: Language Model are Supervised Multitask Learners* paper has authors investigate whether LLM ***pretraining*** can be made more efficient by including synthetic instruction-response pairs instead of raw text from books/websites/papers/etc.
	- Note: *Genie: Achieving Human Parity in Content-Grounded Datasets Generation* is a similar paper. Added to list.

![[Pasted image 20240723125701.png|500]]

- Specifically, the researchers experiment with generating instruction-response data from the raw training itself using an =="instruction synthesizer"== LLM finetuned for this task.

For an instruction synthesizer, they use a [[Mistral 7B]] v0.1 LLM finetuned to generate instruction-response pairs from raw text.

To finetune this synthesizer, the researchers use datasets like [[HotpotQA]], (also RACE and [[SQuAD]]) which consists of passages from Wikipedia associated with questions and answers.

![[Pasted image 20240723130013.png]]
((Above: Note that we're generating both the instruction and response; not sure if this differs from [[Genstruct]]))

A noteworthy detail regarding instruction synthesizers is that multipel raw texts and instruction-response pairs are concatenated as few-shot examples:
![[Pasted image 20240723130308.png]]
Multiple raw texts and instruction response pairs are concatenated as few-shot examples ...

Pretraining with Instruction Data
- Now that we have a method to generate instruction-response pairs, let's get to the interesting part -- how well do models train on this augmented dataset?
- The first set of results look at two small models (based on Mistral arch) trained from scratch:
	- 500M parameters
	- 1.3B parameters

![[Pasted image 20240723130955.png]]
As we can see in the table above, the model trained via the proposed instruction pretraining approach performs best on most benchmark tasks... 
- ((But is this expected? Now instruction-train both the vanilla and instruct PT on a few million additional instruction-tuned. Ah, answered next!))

![[Pasted image 20240723131246.png]]

Continual pretraining with instruction data
- Pretraining from scratch is interesting because that's how LLMs are created in the first place, but practitioners care more about continual pretraining and finetuning.
- Continual pretraining means that we take an existing pretrained model and pretrain it further on new domain data. For instance, think of a LLaMA 3 8B base model that's been trained on a general text corpus... and then adapting it to finance, medical, legal, or other domains.

![[Pasted image 20240723131736.png]]
We can see that instruction pretraining approach clearly outperforms the vanilla pretraining approach.

The instruction pretraining approach is a refreshing change to the usual pretraining pipeline; though it might be expensive to create the instruction-augmented corpora..


---

## Gemma 2
- In the [[Gemma 2]] models (2.6B, 9B, 27B), the main theme is exploring techniques without necessarily increasing the size of training datasets, but rather focusing on exploring techniques to develop small and efficient LLMs.
- The blend three main architectural/training choices for the 2.6B/9B models:
	1. [[Sliding Window Attention]]
	2. [[Grouped Query Attention]]
	3. [[Distillation|Knowledge Distillation]]

Sliding window Attention
- Popularized by [[Mistral]], a technique using a fixed-size attention block that allows a current token to attend to only a specific number of previous tokens, instead of all previous tokens.
![[Pasted image 20240723132202.png|400]]
In the case of Gemma 2, authors alternated between regular attention and [[Sliding Window Attention]] layers. It results in improved speed with a barely-noticeable difference in perplexity.
![[Pasted image 20240723132308.png]]


Group-Query Attention
- [[Grouped Query Attention]] (like in [[LLaMA 2]] and [[LLaMA 3]]) can be regarded as more generalized form of [[Multi-Query Attention]].
	- Motivation: Reduce the number of trainable parameters by sharing the same keys and values heads for multiple query heads, thereby lowering computational requirement.
![[Pasted image 20240723132420.png]]


Knowledge Distillation
- The general idea of [[Distillation|Knowledge Distillation]] is to transfer knowledge from a larger model (the teacher) to a smaller model (the student).
- In Gemma, they train a 27B teacher model from scratch, and then train the smaller 2B and 9B (student) models on the outputs of the larger teacher model.