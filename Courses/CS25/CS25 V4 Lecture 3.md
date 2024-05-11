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

2020: [[GPT-3]] - Language models can really do interesting things if you prompt them correctly! [[In-Context Learning]], [[Few-Shot Prompting]]
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
	- [[Self-Instruct]]: A paper from AllenAI about creating synthetic Question/Answer pairs from a human-written seed of prompts. It sounds obvious today.
- [[Vicuna]]: really they just added new sources of prompts to the distribution, using ShareGPT. Also introduced the idea of [[LLM-as-a-Judge]].
	- [[ShareGPT]] was one of the only datasets that got open language model builders *prompts* that were similar to what people were actually asking ChatGPT! 
		- You would install a Browser plugin that would let you share your prompts/conversations from ChatGPT.
		- Legally gray -- not clear that users knew that the data was going to be used to train models. But incredibly important to the open source development
- [[Koala]]: Mostly known for having a diverse set of datasets. (Alpaca, Anthropic Helpful/Harmless dataset, ShareGPT, WebGPT). 
- [[Dolly]]: Finetuned from a different baes model; from the [[Pythia]] models from [[Eleuther]];
	- Added some 15k human-written data to the loop!
		- Almost all the projects we talk about today talk about either distilling data from OpenAI or creating synthetic data. Only some, like Dolly, go the extra distance.

![[Pasted image 20240418165458.png]]
In the slides above, we've been talking about "weight differences" versus LLaMA


![[Pasted image 20240418165244.png]]
Annoying phase: You had to clone it, convert it with some delta to the original model, and then upload to HuggingFace. Pretty onerous to researchers, but changed in LLaMA2. Today, we see different license restrictions on how LLaMA is used; If we finetune a model today on LLaMA3, it has to be called "LLaMA3-{model name}"


![[Pasted image 20240418165515.png]]
(Above: blocked out, over 10,000 trees, and over 1,000 volunteers)
Still the biggest community-coordinated human-generated dataset
- It was so incredibly important to the process of alignment, and is still used today.
- The first majorly-successful project of the era; the dataset is still used today. ==We need more things like this!==

Stable Vicuna (April 28, 2023): Trained with [[Proximal Policy Optimization|PPO]] on popular datasets, using OpenAssistant1 dataset for SFT/PPO, and the Antrhopic HH + Stanford Human Preferences (SHP) for RL.

![[Pasted image 20240418165723.png]]
The idea of [[Low-Rank Adaptation|LoRA]] and [[Quantized Low-Rank Adaptation|QLoRA]] enabled a bunch of players to actually finetune models.

![[Pasted image 20240418165849.png]]
Note: Guacano was the model that was released alongside QLoRA as a testament to its abilities.


![[Pasted image 20240418170015.png]]
It seemed like things were slower on the ground in this era, but if you look back it was quite important.

![[Pasted image 20240418170031.png]]
A lot of people were continuing to build on these LoRA and QLoRA methods
- At HuggingFace, they could now do RLHF on 7B models... almost on consumer GPUs!

Weeks and weeks would go by, and it was like... "Why hasn't anyone picked up what we released in the blog post and trained a great model with it?"
- There's something weird about LoRA being used during pretraining that makes it hard to get a good model out of it -- it's better to scale using $ than it is to use.

![[Pasted image 20240418170217.png]]
Another defining moment of this era is the LLaMA2 backlash: People would (eg) ask how to kill a Python process, and the model would refuse to do it -- this caused a large discussion about what Models should/should not be able to do.
- Should models be safe, or follow the instructions that I give it?

![[Pasted image 20240418170247.png]]
It led to the idea of "Uncensored models"
- It's a popular category on HuggingFace
The idea is to remove the points from our finetuning dataset; LMs really aren't censored to begin with, but the dataset needed more filtering or some way of becoming unbiased. A lot of people now only try to make models that don't refuse anything. 

![[Pasted image 20240418170342.png]]
Solid models being trained, but they didn't have the right press/release tactics to make them really stick
- The team behind [[WizardLM]], creating [[Evol-Instruct]] method, *CLEARLY* worked well, but for some reason the narrative wasn't actually changed.

![[Pasted image 20240418170509.png]]
In the background, we were establishing new evaluation tools that ended up being the standard of today.
All of these things were created about the same time, when we were desperate to understand how our models were actually doing in the open (without having the $ of Anthropic to sit humans in front of the models and review them).
Perspective: What can I use when I try to align models, and what can give me immediate feedback on this?

![[Pasted image 20240418170604.png]]
[[ChatBotArena]]: Fantastic, defining the biggest langauge models performance questions (GPT vs Claude)
- Hard to get your model in there if you're a small shop
- Takes weeks to get results

![[Pasted image 20240418170725.png]]
[[AlpacaEval]]: If you have a list of prompts that you compare to some other strong base model (Claude, GPT) and then you ask a language model which is better.
- Includes datasets from OpenAssistant, Vicuna, Anthropic

Asking a LM to provide some rating is going to have some ceiling to it; we aren't going to be good at comparing frontier models (necessarily)
- Easy to use, because of single-turn generation (?)
- Potentially biased towards language models that produce longer responses

Question: "What does it really mean when I say model A beats model B 80% of the time?"

![[Pasted image 20240418170815.png]]
- Just improved the model
- What does it mean to beat GPT4, when it already answers questions very well? What exactly does a 20% -> 30% mean in AlpacaEval?

![[Pasted image 20240418170930.png]]
- Generate a completion to 80 diverse prompts, and then ask the model: 0-10, how good were these completions?
- Runs into same problem: What if our model is getting really good? How do we use LLM-as-a-judge?
	- GPT-4 only gets like 90%
	- Only 80 prompts in evaluation set

We can tell if a model gets really bad if MTBench and Alpaca give it really low scores, but what does it mean when we get middling scores? How do we interpret and compare them?

![[Pasted image 20240418171025.png]]
OpenLLM Leaderboard came out of the team at HuggingFace; we just wanted to evaluate more models to get more signal, and get some ballpark estimate.
Grew into an ecosystem-supporting discovery tool, because getting ANY signal was so useful!
- Gave everyone access to a little more signal on the models... but didn't really solve any of the fundamental problems.
- Didn't show us that RLHF would make scores go up 🤔

![[Pasted image 20240418171142.png]]
- It's very hard to make sense of these evaluations.
- AlpacaEval/MT Bench solve this by being cheaper and more accessible
- How do we push this along? Colleague at AI2 launched WildBench as sort of a hybrid of these. It's still an open question

Question frmo class: About model versus system. Day one, ChatGPT has 
- LLaMA3 released with LLaMA-Guard that does this moderation on the fly too; the actual model being generated does no reasoning over what is an unsafe topic.


![[Pasted image 20240418171232.png]]
We asked: Can we even use RLHF in the open? Do we have the skills/tooling?

Reviewing RL fundamentals, leading into [[Direct Preference Optimization|DPO]]

![[Pasted image 20240418171453.png]]
This is what we're optimizing when we optimize RLHF. It loks nebulous, let's break it down.
- We maximize some reward, subject to a policy $pi$ , penalize
- We want to increase our reward, but don't go too far

Primary questions: ==How do we implement a good reward function? How do we optimize the reward?==
- The environment has the reward function built-in in a usual RL scenario; but here, we're artificially creating one!

![[Pasted image 20240418171549.png]]
We learn a preference/reward model;
- We take a LLM that predicts the separation between two preferences.
	- This is called a Bradley Terry model
- The reard is proportional to the probability that we choose our text over other text.

What if we just use gradient ascent on this equation? What if we just try to use Gradient descent directly?
![[Pasted image 20240418171641.png]]
- Back in May, still talking about OpenAssistant, the DPO paper came out!
	- A great paper to learn about language model math.
The core idea: Why spend all this time learning a reward model when we can use Gradient ascent and solve a loss function?


![[Pasted image 20240418171718.png]]
It's simple to do: As long as you have access to the log probabilities from the model, you can compute teh DPO loss!
- Scales nicely with existing libraries. It trains an implicit reward function
- Reward is a function of the log-probabilities

![[Pasted image 20240418171818.png]]
Debate that's gone on:
- Should we use PPO? [[REINFORCE]]? DPO?
- They're all very different styles of optimization
	- Using RL update rules: Learning a value function, taking gradient functions with respect to value funcitons
	- DPO: Taking gradient steps directly from the probabilities of the LM
- Very different!

Both of them are continuing to progress, and that is good.

What made this kick into high gear:
![[Pasted image 20240418171918.png]]
[[Zephyr]] was the first model to make a splash with DPO; it felt really good to use it, and it was building on this better base model that [[Mistral]] was coming out with, and used the excellent [[UltraFeedback]] dataset.

==Core thing getting [[Direct Preference Optimization|DPO]] to work: LOW LEARNING RATE!==

It was a validation proof that DPO worked, 4 months after the paper (people were losing hope!)

![[Pasted image 20240418172026.png]]
[[Allen Institute|AI2]] was the first to scale DPO to 70B parameters
- Will DPO scale to larger models? The answer is yes!
- Scores continued to climb; this model was so close to beating GPT-3.5 on ChatBot arena; a couple ELO points below.

Open models started to get the chatty behavior that for so long had eluded them...

Now it's like, okay: RLHF methods can really work!


![[Pasted image 20240418172122.png]]
Other important projects:
- NVIDIA's [[SteerLM]]: collected feedback data with attributes on it (How helpful? How concise?), showed that PPO is better than DPO?
- Berkeley's [[Starling]] model; introducing a new dataset [[Nectar]]



![[Pasted image 20240418172226.png]]
In 2024, we've seen more companies come into the space.

![[Pasted image 20240418172452.png]]
Way more types of models, and types of players
- [[Genstruct]] from [[Nous Research]]: A finetuned model for rephrasing any text into instructions!
- [[OLMo]] from [[Allen Institute|AI2]]: Very open-source models with a bunch of information
- Databricks [[DBRX]]
- Cohere's [[Command R]] and [[Command R+]], the latter of which beat GPT-4 on human evaluations! 😮❗
- Research models like Microsoft's [[Rho-1]] (reward model weighted pretraining)
- Multilingual fine-tuning with Cohere's [[Aya]]
- More MoE models: [[JetMoE]], Qwen MoE
- State-space model like [[Jamba]]

![[Pasted image 20240418172704.png]]
LLaMA3's release is more about scaling than it is about alignment.
- The LLaMA2 paper was extremely detailed about alignment. We'll get a LLaMA3 paper soon, and when that paper comes out is when we'll learn all the interesting alignment things they've done; but they're very unlikely to release the human preference data
- He's yet to get them to release the Reward Model they trained
- Can we get them to support the open alignment ecosystem like they did the pretraining ecosystem with the release of the models themselves?

People like Mistral's models because there's a Magnet link and it's funny; we're used to it, and it's really fun to follow. It's not a time to worry about being scooped; figure out whether you can contribute on evaluation or on these techniques/models



# Current Directions
![[Pasted image 20240418172936.png]]

Q: Will open models catch up to closed models?
A: Probably not. Open model weights aren't inherently unsafe; given the territory we're going with in AI where we're uncovering capabilities that we've never seen, Nato thinks its fine that there's a few months delate.

[[Maxime Labonne]]'s line above seems to show the gap narrowing.

![[Pasted image 20240418173035.png]]
We have 2-3 datasets driving all the research in alignment:
- [[Helpful and Harmless|HH]] is from 2022
- [[UltraFeedback]]
- [[Nectar]]
We need more, especially human-written, to add robustness!

There's a SHIT TON of DPO-related papers coming out:
- [[Odds Ratio Preference Optimization|ORPO]]
- [[Conservative Direct Preference Optimization|cDPO]]
- [[Identity-Mapping Preference Optimization|IPO]]
- [[Binary Classifier Optimization|BCO]]
- [[Kahneman-Tversky Optimization|KTO]]
- [[Direct Nash Optimization|DNO]]
- sDPO (sequential DPO)

More model sizes!
- Most alignment at the 7B/13B scale; there's a large drive to make smaller models aligned -- it's an opportunity where there aren't that many playing in the space!

Specific evaluations
- how do we get more specific evaluations than ChatBotArena?
- Personalization: A large motivation behind local models -- a young area academically

![[Pasted image 20240418173227.png]]

Berkeley-Nest
OpenBMB in China
Argilla, a startup building tools around annotating data, focused on preference data

[[Jon Durbin]]
Model Merging is sos accessible; you don't need a GPU to do it, it's a for loop.

Pitfalls of Synthetic Data: Repetitiveness of the data being created. The models are going to generalize less-well and probably get smaller boosts from alignment.

Advice on grad school: You should be very wary of listening to advice, but the most important thing you can do is keep trying to develop skills and build things that matter -- you'll never be able to keep track of everything. Grad school is about learning to do research, and that still has value - but industry is also fun!

You indicated that making LoRA methods working with RL is tricky -- does DPO work well with LoRA or its variants?
- He hasn't seen this be particularly successful. He waits until there's a model release "in the ballpark" with that method... There seems to be some weirdness preventing this from happening.

You mentioned GPT4 being used as an evaluation method, but perhaps it causes data contamination.

For stuff like LLaMA3 training on so many tokens, does that make it harder to align models without losing some capabilities? You will need different LR, Batch Size, for different models, because models are released at different levels of training. It's not that LLaMA is overtrained/harder to finetune, the model can keep learning, it just takes more data to get improvements. But this should only help, not hurt! 

Q: Do you think synthetic data generation like [[Cosmopedia]] is the way to go in the future?
- He thinks it will be good, also a way to get around the fact that Google is paying Reddit to use their data so we can't train on it
- Cosmopedia and other Synthetic datasets can be a way around this










