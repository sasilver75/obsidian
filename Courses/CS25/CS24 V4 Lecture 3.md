April 18, 2024
Speaker: [[Nathan Lambert]], from [[Allen Institute|AI2]]

Topic: Aligning Open Language Models

---

![[Pasted image 20240418163127.png]]

Since ChatGPT, a lot has happened -- it's been a blur! Let's retell what happened since ChatGPT.

This is not really a 101 lecture, but it will probably give you a lot of context as to why people are mentioning things, what matters, etc.



Claude Shannon had a paper about arranging characters to create language models in 1948.
A lot has happened since then!
Since then, they've been built on something called the "autoregressive loss function"

"I saw a" --> {?}
The idea is that there' one token that's the correct token, so we'll increase the probability of that token, and decrease the probability of everything else. This has enabled wild things.

In 2017: Attention is all you Need -- the [[Transformer]] arrives
- It's a great mechanism to dig into what attention is actually doing, but not he focuso n this talk!

2018: [[ELMo]] (Contextualized word embeddings)
[[GPT-1]], [[Bidirectional Encoder Representations from Transformers|BERT]] get released -- getting these better models, training on large internet-scaled corpa

2019: [[GPT-2]], we learn about scaling laws from Kaplan et al in 2020
- Pioneered discussions on how to *release* language models, because of risks of language models.

2020: [[GPT-3]] - Language models can really do interesting things if you prompt them correctly! [[In-Context Learning]], [[Few-Shot Learning]]
- With this power came many harms -- this comes back to the discussion of risks.

2021: Stochastic Parrots paper
- Argues whether LMs can be too big

2022: ChatGPT
- Reshaped the narrative about language models AGAIN
- How does this notion of "alignment" emerge in ChatGPT, and what happens after it?

Can ChatGPT exist without RLHF?
- It seems to be necessary, but not sufficient! The post lists a bunch of limitations (the things that we'll takl about in this talk), but notes that it was very important.
- You can't do something like ChatGPT without something like RLHF


RLHF is relied upon elsewhere:
- A key factor in many popular models, both on and off the record, including ChatGPT, Bard/Gemeni, Claude, Llama 2.

![[Pasted image 20240418163709.png]]

![[Pasted image 20240418163808.png]]
A big endorsement of RLHF from the [[LLaMA 2]]
For NLP researchers to say these things... given that it's cost and time-effective, this is *shocking* to an RL person like Nato, for whom RL has really never been either of these! But in the fine-tuning zone, where we aren't training with RL from scratch, it turns out to vaguely be true!

![[Pasted image 20240418163902.png]]
A [[HuggingFace]] collection from Nato to document some of the artifacts that he's going to talk about


![[Pasted image 20240418163127.png]]
Let's cover some of the substantial developments since ChatGPT!

Chapters:
1. Kickstart: A few months after ChatGPT, we just didn't have any models
2. Instruction-Tuning boom: After release of [[LLaMA]], [[Alpaca]], [[Vicuna]] show what you can do with open-weight models and fine-tuning on your own
3. Evaluations and Expectations: At the end of this phase, Mistral's first model came out.
4. [[Reinforcement Learning from Human Feedback|RLHF]] works!: Models trained with some sort of preference-learning technique; either [[Proximal Policy Optimization|PPO]] or [[Direct Preference Optimization|DPO]]. A lot of models al at once!
5. Expansion: A lot of different types of models, a Cambrian explosion; there's so much out there at once, and it's hard to make a narrative out of it!

Some of these models are Base Models; these are extremely important (none of this happens without LLaMA, LLaMA2; they're the bedrock!)
But the Aligned/fine-tuned/preference-trained models are those that you can really play with, so we'll talk more about these.


![[Pasted image 20240418164208.png]]
No one likes listening to definitions, so we'll just post these here.
- Difference between [[Instruction-Tuning]] and [[Supervised Fine-Tuning]]
- *==Preference fine-tuning==* (a term Nato is trying to grow) -- How do we differentate non-RL preference-aligning methods from the RLHF ones?


![[Pasted image 20240418164344.png]]
It's hard to retell how crazy things were when ChatGPT dropped; people were almost losing their minds, and there was a lot of uncertainty about what the future held! There were many articles: "We're going to reproduce open ChatGPT!" -- you can't really have an open model that does what a closed ***product*** does (the product is more than just the model)!
- People learning the basic teams -- what is [[Red-Teaming]]?
- What is a dialogue agent? What makes one useful?
- What tools can we use?

In retrospect, it was wild that people said: "We need to build this thing in the Open!"

![[Pasted image 20240418164517.png]]
LLaMA gets released, and instruction-tuning models show up
- [[Alpaca]] did a bunch of things that are still used today; Trained on 52k instruct-style data distilled from text-davinci-003.
	- Key idea: [[Instruction-Tuning|Instruction-Tune]]: Adapting the model to knowing what its role is, what day it is, etc. It's where we make the model capable of having these dialogue agent behaviors.
	- How do we do it? We continue training with our autoregressive loss function on instruction-following data. This is [[Supervised Fine-Tuning]], specifically [[Instruction-Tuning]]
	- [[Self-Instruct]]: A paper from AllenAI/U













