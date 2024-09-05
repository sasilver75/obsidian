#lecture 
Link: https://youtu.be/SXpJ9EmG3s4?si=GSKTiXVH_7wx_uoB

----
# Subject Prompting, RLHF

![[Pasted image 20240410185517.png]]
Let's take this idea much further; if we follow this idea of: We train a giant LM on all the world's text, you begin to see LMs as rudimentary world models.


![[Pasted image 20240410185540.png]]
Above: Describing an explanation of Pat watching a bowling ball and leaf falling. When we condition the LM by adding the information that Pat is a physicist, the LM fills in that he correctly figures out the experiment.

Language models are also surprisingly good at:
1. Math
2. Code
3. Medicine

This is what happens when you really take the LM model seriously! There's a resurgence of interest in building LMs that are basically assistant. LMs should be able to take a good stab at doing this.


# Lecture Plan

1. Zero-Shot and Few-Shot [[In-Context Learning]]
2. [[Instruction-Tuning]]
3. [[Reinforcement Learning from Human Feedback|RLHF]]


---

# Few Shot and Zero Shot Learning


### Emergent Abilities of LLMs
- These are decoder-only language models just trained to predict the next token in a corpus of text.

[[GPT]], 2018 (11M parameters, 12 layers, trained on ~7000 unique books,  4.6GB text)
- Showed that language modeling at scale can be an effective pretraining technique for downstream tasks like natural language inference.

[[GPT-2]], 2019 (1.5B params, same arch, 4GB->40GB of internet text data (WebText))
- Paper title: "Language Models are *Unsupervised* Multitask Learners"
- One key emergent ability: Zero-Shot learning
	- The ability to do many tasks that the model wasn't explicitly trained for by simply specifying the right question
		-   > Passage: Tom Brady... Q: Where was Tom Brady Born? A: ...
- Comparing probabilities of sequences (Winograd Schema Challenge)
![[Pasted image 20240410203905.png]]
- Requires some world knowledge to resolve this. The way we get ZS predictions out of a LM is we just ask it which sequence is more likely
	- The cat couldn't fit into hte hat because the cat was too big
	- The cat couldn't fit into the hat because the hat was too big
- GPT-2 beats SoTA on language modeling benchmarks with no specific fine-tuning!

![[Pasted image 20240410204141.png]]
Even though GPT-2 was just a language model, we can get it to do summarization by prompting it in a certain way -- we append a "TL;DR" token at the end, and then ask it to continue! By asking the model to predict, after the TL;DR token, it will generate a summary!
- It's not actually that good; it does only a little better than the random baseline, but it *does* approach supervised/finetuned approaches that explicitly does finetuning.

[[GPT-3]], 2020 (15B parameters, >600GB)
- Paper Title: "Language models are few shot learners"
- ==Emergent few-shot learning was the key takeaway from the GPT-3 paper==
	- Give examples of a task being completed, before -- this is often called [[In-Context Learning]] to highlight the fact that no parameters of the model are being updated.
![[Pasted image 20240410204343.png]]
- The more "shots" we provide, the more its performance seems to increase!

Few-Shot Learning is an emergent property of model scale!
![[Pasted image 20240410204522.png]]
Above:
- We come up with a bunch of simple letter-manipulation tasks that are unlikely to explicitly exist in the data (for the words that are tested, at least)
- We see that as we increase model size, the ability to do few-shot learning seems to be an emergent property of random scale -- especially for random insertion
	- ((There are probably more dramatic "emergent" charts, but sure))


![[Pasted image 20240410204713.png]]
Above: Few or Zero-Shot prompting/ICL versus Traditional Finetuning


![[Pasted image 20240410204940.png]][[Chain of Thought]]

![[Pasted image 20240410204951.png]]
CoT is effective! As we scale the model for GPT and other models, the ability to do CoT *==emerges==*. We see ability approaching prior supervised bests, using these models.

But wait, do we even need to provide examples of humans reasoning out questions for our few-shot examples?
What if we tried a [[Zero-Shot Prompting]] version of [[Chain of Thought]] -- Zero-Shot CoT!
- We ask it: "==think step by step!=="
![[Pasted image 20240410205226.png]]
This results in impressive performance!

![[Pasted image 20240410205317.png]]
==Just asking the model to think through things is better than actually providing a few exemplars (without CoT)==
- The authors tried a bunch of things, and found that the specific "Let's think step by step" thing was about the best they tried.

![[Pasted image 20240410205421.png]]

Is Prompt Engineering becoming a new profession?
- Asking models for reasoning
- "Jailbreaking" LMs
- Diffusion art using interesting prompts
- Using Google code headers to generate better prompts

((Makes me think: What if we reformulate our training data to be in CoT format, or something? Does that make for better training data?))

![[Pasted image 20240410205959.png]]

![[Pasted image 20240410213005.png]]
[[Super-NaturalInstructions]] dataset!

How do we evaluate such a model that's been instruction-tuned?

![[Pasted image 20240410213114.png]]
Above: [[MMLU|MMLU]]
See: At the time, GPT-3 wasn't thaat good, but certainly better than the random baseline!

![[Pasted image 20240410213150.png]]
Above: [[BIG-Bench]]
- Contains some esoteric task, including translating various kanjii into ASCII asrt and having the language model guess what the character means.

![[Pasted image 20240410213424.png]]

![[Pasted image 20240410213823.png]]


# Reinforcement Learning from Human Feedback

![[Pasted image 20240410213946.png]]
Let's say that we can score outputs, giving reward to the model depending on the quality of the generation. The objective that we want to maximize is maximize the expected reward of samples from our language model.

We can use [[Reinforcement Learning]]!
![[Pasted image 20240410214112.png]]
New RL algorithms seem to work well for large neural language models, eg [[Proximal Policy Optimization]]

![[Pasted image 20240410214215.png]]
Humans rewards are non-differentiable!

[[Policy Gradient]] methods in RL (eg [[REINFORCE]], 1992) give us tools for estimating and optimizing this objective!
- We'll describe a very high level mathematical overview of the simplest policy gradient estimator.

![[Pasted image 20240410214500.png]]
What happened here is we shoved the gradient inside of the expectation
- Why is this useful?

![[Pasted image 20240410214742.png]]
It's okay if you don't understand this 🙂
Can you see any problems with this objective? 🔍

## Problems with Human Preferences (in the context of RLHF)

![[Pasted image 20240410215027.png]]
- We label some data with expensive human labelers, in terms of human preferences
- Then we train a reward model that picks up the behavior of our human labelers
- This is actually what we finetune our model, during RLHF!

![[Pasted image 20240410215256.png]]
So we can make pairwise preference labeling, and still have a reward model that outputs a number (usually mean of 0).

![[Pasted image 20240410215448.png]]
So how good are our reward models at predicting human preferences??
- They're pretty much approaching the same accuracy as humans are!

![[Pasted image 20240410215654.png]]
(We use the KL divergence, because if you end up adhering too closely to the reward model's preferences, you'll get weird mode collapses and strange behavior.)

![[Pasted image 20240410215729.png]]
It does work!

![[Pasted image 20240410215908.png]]

![[Pasted image 20240410221446.png]]

![[Pasted image 20240410221955.png]]


![[Pasted image 20240410222337.png]]
Above: [[Reinforcement Learning from from AI Feedback|RLAIF]] and [[Constitutional AI|CAI]]
- Also: Finetuning LMs on their own outputs
	- This has been explored a lot in the context of [[Chain of Thought]] reasoning
		- eg: "Large Language Models can Self-Improved"
			- Produce a bunch of reasoning, finetune on that as if it were new data -- can it get any better?













