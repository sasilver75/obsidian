Link: https://thegradient.pub/ai-is-domestification/

-----

## Introduction
- As internet-scale AI models mature rapidly from coarse research demos to productionized user-facing systems, expectations have increased drastically.
- In a few months, the AI community has shifted from being impressed by proof-of-concept zero-shot capabilities to tackling the challenging last-mile of improving the quality and reliability of fine-tuned capabilities.
	- It turns out that it's not (yet) sufficient to just dump ever larger amounts of compute, tokens, and parameters to ascend scaling curves!
	- The trillion-dollar question is how to make these base foundation models useful and performant for *specific downstream tasks!*

Increasingly, ==modern AI is now the study of digital domestification==! The art (and science) of taming wild internet-scale data distributions!

## Prior Amplification Methods
- The process of training LLMs and vision-language models (VLMs) critically relies on a vast amount of internet-scale data.
- High-capacity models like [[Transformer]]s have shown the ability to effectively model these extremely diverse data distributions -- perhaps all too well, sometimes!
- ==These large models train on a virtual stew of *all kinds* of data -- elegant prose, toxic 4-chan posts, brilliant software projects, bug-ridden homework code, gorgeous professional photography, and amateur social media selfies.==
	- And so these models train and soak up all the glory *and* the imperfection of these web-scale datasets, and ==they begin to act as mirrors raised to the face of the digital human experience==!
	- While these "raw" models might offer some unique sociological tool to study human culture, **they're ==a far cry from producing high-quality, desirable, and consistent outputs==** -- These are the things that are necessary for full productionization in user-facing applications at scale!
- These models aren't *bad models* -- they're doing exactly what they were designed too - to exhaustively model the distributions that they were trained on!
	- These underlying data distributions -- the ==dataset priors== -- may *indeed* contain MANY undesirable properties! But also many of the good properties (and the diversity and scale) requisite for performant final models!
		- A popular recent paper (LIMA: Less is more for Alignment) emphasizes that a model's knowledge and capabilities are almost entirely learnt during pretraining, while alignment teaches it ==*which subdistribution of priors should be used during inference*==. 
		- So how do we amplify the good priors in the dataset, and suppress the bad priors? How do we *tame* the *raw* models captured directly from wild heterogenous data into what we want?


==[[Prior Amplification]]==: How a set of desired priors can be projected and amplified onto a model's understanding of internet-scale datasets.

In the past year, a few major approaches have gained traction. While their technical underpinning and advantages vary, they all share the common goal of *prior amplification*

1. [[Prompting]]
2. [[Supervised Fine-Tuning]] (SFT)
3. [[Reinforcement Learning from Human Feedback]] (RLHF)
4. Incorporating AI Feedback: AI Critics
5. Synthetic Data Generation


![[Pasted image 20240120220158.png]]

----
### (1/5): Prompting
- The most obvious starting point for trying to *steer* a foundation model towards some desired prior is just to *nicely ask the model!*
- The intuitive concept is simple:
	- If the model has learned about all sorts of diverse data during training, can you guide the model *at inference time* by ==carefully crafting the context to make your query look more like *specific high-quality examples* in the training data==?
		- This ==takes advantage of correlations and priors seen during training==.
	- Chess games correlated with high participant ELO ratings will mostly likely have much stronger moves than those with low participant ELO ratings, so at test time, ==an effective prompt make make it abundantly clear to the model that it's in the high ELO chess playing regime==, and should accordingly make strong grandmaster-caliber predictions!
- There are some clear limitations of zero-shot prompting! Prompting is an opportunistic strategy that's strongly dependent on the patterns, correlations, and priors seen in the original training dataset...
	- Successful prompt engineering is a ==tug of war== between prompts that are too generic (but easily followable, like "play like a Chess AI") and prompts that are too specific (to the extent that the model can't generalize to it, like "play like a 9000 ELO chess AI").
- Prompting's reliance on underlying data distributions becomes challenging when wild data distributions contain many more undesirable data correlations than desirable correlations.
	- Some side discussion on the [Waluigi Effect](https://www.lesswrong.com/posts/D7PumeYTDPfBTp3i7/the-waluigi-effect-mega-post), where undesirable regions in training data sometimes act as absorbing states/traps.
- Regardless of whether these issues will go away with "better prompting," it's clear that zero-shot methods force a model to operate at inference time with all of the baggage of the arbitrary priors that were contained in the training distribution.
	- Can we amplify priors more effectively if we look *beyond* gradient-free prompting, and instead consider fine-tuning the raw model itself?

### (2/5): Supervised Fine-Tuning (SFT)
- In supervised finetuning (SFT), raw models are pretrained on diverse datasets, and then are ==subsequently trained on *smaller but high-quality datasets*==, which may or may not be subsets of the original dataset.
- SFT is the epitome of *show, don't tell*, ==where the finetuning dataset acts as the gold standard== that contains all the final model's desired properties.
	- This simplicity makes a compelling argument: provide the model with some target dataset, and SFT promises to bring the raw model closer to this target distribution.
- Since SFT (aka =="behavior cloning"==) is supervised learning, if the data is good and the models are large, success is guaranteed.
- The regime of SFT is flexible to *what* the finetuning dataset source is, too!
	- It could be a subset of the original, diverse dataset, or it could be a new, custom dataset altogether. It could be ==painstakingly crafted and verified manually== by human labor, or it could be ==automatically sourced using crawlers== and heuristic rules... As we'll see later, it could also be ==generated synthetically==!
- Let's, assume we've selected a particular finetuning dataset that represents all of the nice priors that we want to distill into our model: *how* do you mechanically finetune the base model? There are a few options!
	- ==Standard SFT== finetunes the *entire base model*, updating the weights of the entire network. This is the *most exhaustive* type of update possible, with the potential for *significant changes* in underlying model behaviors.
	- Sometimes, a lighter touch is needed is needed, and just a *subset* of the network can be finetuned!  ([LiT](https://ai.googleblog.com/2022/04/locked-image-tuning-adding-language.html) paper from google) A related class of methods known as ==[[Parameter-Efficient Fine-Tuning]]== (PEFT) takes this concept further and freezes large parts of the original model, only finetuning a relatively tiny set of (extra) model parameters!
### (3/5): Reinforcement Learning from Human Feedback (RLHF)
- 

### (4/5): Incorporating AI Feedback: AI Critics
- 

#### (5/5): Synthetic Data Generation
- 



---
## Conclusion



























