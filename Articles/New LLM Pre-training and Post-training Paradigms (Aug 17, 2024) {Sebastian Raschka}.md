https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training?utm_source=post-email-title&publication_id=1174659&post_id=147749119&utm_campaign=email-post-title&isFreemail=true&r=764e6&triedRedirect=true&utm_medium=email

[[Sebastian Raschka]]

----

<<<<<<< HEAD
![[Pasted image 20241126143049.png]]

There are hundreds of papers a month that propose new techniques and approaches for post-training.

Let's look at the pre-training and post-training pipelines of the most recent SoTA models!


Let's focus on the following pre-training and post-training pipelines of the following models:
1. [[Qwen 2]]
2. [[Apple Intelligence Foundation Language Models|Apple Foundation Model]]
3. [[Gemma 2]]
4. [[LLaMA 3.1]]

These models are presented in order based on the publication dates of their respective papers.

# Alibaba's Qwen 2
- Comes in 5 flavors
	- Dense 0.5B
	- Dense 7B
	- Dense 62B
	- MoE 57B (14B active)

One of the standout features is good multilingual capabilities in 30 languages:
- Large, 151,642 token vocabulary
	- For comparison [[Llama 2]] had a 32k vocab and [[LLaMA 3.1]] has a 128k vocabulary
- "==As a rule of thumb, increasing the vocab size by 2x reduces the number of input tokens by 2x, so the LLM can fit more [text] in the same input?=="
	- I think this is true if you're just raising your vocab count using 




# Apple's Apple Foundation Model
- 


# GDM's Gemma 2
- 


# Meta's LLaMA 3.1
- 
=======
Training methods have evolved since ChatGPT was released -- let's check in on the latest advancements in both pre-training and post-training!

![[Pasted image 20241120182333.png]]
Above: The LLM development/training pipeline

Four major new LLMs have been released in the last few months, with relatively detailed technical reports:
1. [[Alibaba Research]]'s [[Qwen 2]]
2. [[Apple]]'s [[Apple Intelligence Foundation Language Models|Apple Foundation Model]]
3. [[Google Research]]'s [[Gemma 2]]
4. [[Meta AI Research]]'s [[LLaMA 3.1]]

These models are presented in the order of their publication date of their respective technical papers on arxiv.

-----

# Qwen 2
- The [[Qwen 2]] family of models is competitive with many major LLMs, but is for some reason less popular than open-weight models from Meta/Google/Microsoft.
- Qwen 2 family comes in 5 flavors:
	- 0.5B Dense
	- 1.5B Dense
	- 7B Dense
	- 72B Dense
	- 57B MoE, 14B active at the same time

One of the standout features of Qwen 2 LLMs are their good multilingual capabilities in 30 languages. They also have a surprisingly large 151,642 token vocabulary.
- LLaMA 2 has a 32k vocabulary
- LLaMA 3.1 uses a 128k token vocabulary

==Rule of thumb: Increasing vocab size by 2x reduces the number of input tokens by 2x so that the LLM can fit more information into the same input.==
- Also helps with multilingual data to cover words outside the English vocabulary.


![[Pasted image 20241120182814.png|600]]
Above: [[[MMLU]] scores for various models, by model size. See that the Qwen-2 models are roughly on the frontier of those compared here.
- Interesting that the AFM-server model from Apple lies above the usual frontier.

Qwen 2 Pre-training
- The Qwen 2 team trained the 1.5B, 7B, 72B parameter models on ==7 T training tokens==, which is a reasonable size. Interestingly, the .5B model was trained on ==12 T training tokens==... but they didn't do this for other models because ==they did not observe any improvements during training, and the additional computational costs were not justified.==
	- For comparison, LLaMa 2 models were trained on 2T tokens, and LLaMA 3.1 models were trained on 15T tokens.

Pretraining data also included synthetic data from previous-generation [[Qwen]] models.

They performed training in two stages:
- Regular p-retraining
- Followed by long-context training (increased context length from 4,096 to 32,768 tokens at the end of pre-training using "high quality, lengthy data")

![[Pasted image 20241120183212.png]]
Above:
- The "continued pretraining" refers to the two-stage pretraining, where we start with regular pre-training and then follow up with a long-context continued pre-training.

### Qwen 2 Post-training
- Employed the popular two-phase post-training methodology of ==[[Supervised Fine-Tuning|SFT]] followed by [[Direct Preference Optimization|DPO]]== for alignment with human preferences (interestingly, they used "RLHF to describe this).
- It seems like the ==SFT+DPO approach is the most popular preference tuning strategy at the moment, due to the ease of use== compared to other methods, like RLHF with PPO.
- If you want to learn how DPO works, Seb implemented it from scratch [here](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)

The alignment phase itself was also done in 2 stages:
1. First using DPO on an existing dataset (offline stage)
2. Second, using a reward model to form the preference pair (online)

>>>>>>> origin/main




























