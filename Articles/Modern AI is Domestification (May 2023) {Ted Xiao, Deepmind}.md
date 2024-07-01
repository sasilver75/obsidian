---
tags:
  - article
---


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
		- PEFT methods like [[Low-Rank Adaptation|LoRA]] have unlocked tremendous open-source innovation, allowing consumer hardware to finetune respectably-sized foundation models!
- ![[Pasted image 20240120232035.png]]
- Clearly, the *how* of SFT is rapidly evolving; there are many ways to do it. Regardless of the exact SFT method, ==there remains a heavy reliance on the composition and *quality* of the underlying finetuning dataset!==
- In SFT, *what priors you amplify* matters just as much as *how you amplify them*! 
- Here are some examples of SFT methods and high-quality datasets that enable distilling desired human priors:
	- [[LAION-Aesthetics]]: Aesthetic images from LAION-5B
	- [[Video PreTraining]]: Task-specific Minecraft gameplay
	- [[FLAN]]: 60 high-quality NLP datasets turned into instruction-following datasets
	- [[Interactive Language]]: Language-annotated robot **trajectories**
	- [[CodeXGLUE]]: Popular code repositories from GitHub; Well-written code.
	- [[Locked-Image Tuning]] (LiT) finetunes text to match a frozen, pretrained image encoder.
	- [[Parameter-Efficient Fine-Tuning|PEFT]] methods like [[Prefix Tuning]],  [[Prompt Tuning]], [[Low-Rank Adaptation]], [[ControlNet]] all freeze the main network and add new tunable weights that can be rapidly adapted to new datasets.

### (3/5): Reinforcement Learning from Human Feedback (RLHF)
- In contrast to SFT, [[Reinforcement Learning]]-related finetuning introduces a *reward model*, a *separate component* that aims to directly provide granular feedback signals to model outputs during training.
- One of the most popular RL finetuning paradigms is ==RLHF==, where the reward model is trained directly on human preference labels.
	- ==Insight:== While SFT took the non-parametric approach of "show, don't tell," RLHF is the opposite; explicitly learn good priors via a parametrized reward model, and then directly *tell* the raw model about these preferences during training.
	- Formulating autoregressive token prediction as a reinforcement learning problem has two very compelling technical benefits:
		   1. Direct on-policy feedback
			- On-policy learning signals are extremely useful and qualitatively very different from those seen during standard offline off-policy training. 
				- On-policy feedback gives the model information on "how good is your *best* prediction?" compared to off-policy feedback, which tells tells the model "how could would this *other* prediction have been?"
				- In addition to on-policy feedback being the most informative, sometimes off-policy feedback can be stale and incorrect; pre-collected training datasets contain target labels that exist in a vacuum and don't consider the model's current capabilities!
		   2. The ability to train on suboptimal data
			- RLHF provides granular rewards that enable training on suboptimal data. 
			- Whereas SFT only allows for a hard boundary between including or excluding data of varying quality, RLHF enables a more flexible approach of utilizing the suboptimal data both during the reward model training, as well as during finetuning using a reward model. During foundation model finetuning, the reward model is able to output multiple grandular reward scales 
				- eg 1.0 for "correct+confident", .5 for "correct+unconfident", and -2.0 for "incorrect+confident", which allows for effective utilization of different types of suboptimal data.
- In addition to these two technical benefits, there's also the systems-level benefit of viewing the reward model as an independent component that can be studied and improved on iteratively.
	- ==This offers the potential of very nuanced reward modeling, which could then propagate very fine-grained feedback to the raw base model.==
		- This is empirically backed by ==SFT seeming to cause larger shifts in a  base model's instruction following behavior.==
- Here's some examples of RLHF that *amplify* human-preference priors:
	- [[InstructGPT]]  trained a text alignment reward function using contractor-collected instruction-following demonstrations as well as human-labeled model output rankings.
	- [[Text-to-Image Alignment]] trained an image generation reward function using samples of discrete human preferences of images generated from text with [[Stable Diffusion]].
	- [[Few-Shot Preference Learning for Human-in-the-Loop RL]] pre-trains a robot manipulation reward model and adapts it to new tasks using human feedback.

### (4/5): Incorporating AI Feedback: AI Critics
- While RLHF provides a powerful mechanism to transfer human knowledge to AI models, it also faces practical limitations -- human feedback can be noisy and inconsistent, as well as being expensive to collect!
- To tackle these challenges, [[Reinforcement Learning from from AI Feedback]] (RLAIF) was developed, to bring existing AI models into the loop by ==using prompted, pre-trained models to generate preference data for training reward models!== 
	- RLAIF capitalizes on the asymmetric property that ==solution verification is easier than solution generation== ("I can't describe it, but I know it when I see it").
	- Even if existing foundation models aren't good enough to generate outputs corresponding to some desired prior, perhaps they're good enough to know good answers when they see them, and provide some on-policy preference labels? ==RLAIF thus captures good priors contained in prompted foundation models to generate automated preference data, with no humans in the loop, for downstream reward model training.==
- Foundation models acting as AI critics can go beyond generating data for reward models -- ==they can also BE the reward model, directly!==
	- At inference time, foundation models can give their best shot at completing the task and then self-reflect on whether they succeeded!
	- AI Critics are inference time can enforce additional structure, such as being combined with tree-structured search that prunes the logical reasoning plans that don't stand up to AI critic scrutiny -- or even using *multiple* AI critics in a sort of "Society of Minds" to *debate and discuss* potential outputs.
	- At training time, these AI critics can provide direct on-policy feedback, aiming to automatically distill the good AI critic priors into the finetuned models. There's a clear parallel here to lessons in Actor-Critic methods in RL, where critics are easier to learn but can provide great regularization and bootstrapping benefits to the actor policy.

Here's a few examples of AI feedback that amplify existing AI priors onto other AI models:
- [[Claude]] introduced [[Constitutional AI]], which starts with a human-produced prompt of rules and principles that is used during AI feedback generation and preference ranking of outputs, which are then used during downstream reinforcement learning to reduce harmfulness and increase helpfulness of instruction-following LLMs.
- [[ALMoST]] uses LLMs of different quality and sizes to generate contrasting responses which can be used to train a ranking-based reward model.
- LLM Self-Reflection has been a rapidly accelerating area. LLMs understand their own uncertainty -- [[Reflexion]] (and followups) use AI feedback during inference time, and [[LLMs Self-Improving]] incorporates AI feedback during training.
- [[Tree of Thought]] uses structured search at inference time to utilize LLMs to propose and search for the most promising reasoning chains.
- [[Society of Minds]] utilizes multiagent debate between LLMs to use an almost ensemble-like approach to improve factuality and reasoning.
- [[Inner Monologue]] uses expert models to provide textual feedback for LLMs that iteratively plan robotics tasks.
- [[AutoGPT]] combines AI feedback with digital tool use to autonomously execute tasks during inference time until self-judged completion.

#### (5/5): Synthetic Data Generation
- We've already mentioned examples of prior amplification that included AI in different parts of training, whether it was dataset filtering like LAION-Aesthetics using CLIP embeddings, or using AI critics using feedback generated by foundation models.
- But can AI models also improve how we *acquire* and *label* entirely new datasets? ==Could AI models actually *generate useful data* that's high-enough quality to subsequently train on?==
- A starting place might be to not entirely replace humans in the data engine loops, but rather *augment* human abilities with a shared autonomy paradigm!
	- Predictions from AI models might not be perfect, but are perhaps a good enough starting point to save human labeling time.
	- You don't even need to go whole-hog! Recently, Meta released an SA-!B segmentation mask dataset, which was made possible by an interactive model-assisted labeling process that was 6.5x faster than a completely-manual data labeling approach. 
- Advances in generative modeling are starting to enable creation of useful synthetic data without humans in the loop at all! This has been studied in the past as something related to [[Semi-Supervised Learning]] or pseudo-labeling in the past.
- In the post-2021 proliferation of performant, internet-scale models in language and vision, the potential of synthetic data generation has been dramatically increased.
	- Whereas in the past, synthetic labels relied on narrow, domain-specific models, now synthetic labels can potentially be produced by general foundation models that might not even be specifically fitted for the task at hand!
		- This lowers the cost of *trying out* synthetic data generation, and has the potential to import internet-scale common sense into specific training domains.
- The narrative of "==general large models" being used for narrow synthetic generation"== has been increasingly explored in a variety of contexts, ranging from vision to robotics. Especially exciting results that show the *power* of positive transfer of general model capabilities from the data generation to the consumption model.
	- [[InstructPix2Pix]] created a synthetic image editing instruction dataset by combining the instruction understanding capabilities of LLMs with text-to-image generative models!
- Synthetic data generation could also be used as data augmentation for *existing* ground-truth labels! 
	- This is explored in [[DIAL]], which augments language-conditioned robot trajectories with instructions prediction by [[CLIP]].
- Finally, synthetic data generation can also be used for [[Distillation]] between models of very different scales, such as [[Alpaca]] fine-tuning a 7-B parameter [[LLaMA]] model on instruction following outputs from 175B-parameter [[GPT-3]].

The trend seems clear:
- Although the usefulness and quality of synthetic data was often called into question in the past, it seems that there are at least a few compelling domains where ==synthetic data is able to combine *low-cost efficiency* with *sufficient quality for training*==, and in some cases even bring *positive transfer* from the data labeling model to the data consumption model.

Here's some examples of synthetic data generation:
- The [[Segment Anything Model]] was trained on a 1.1 billion example segmentation mask dataset collected with model-assisted annotations.
- Tesla Autopilot's vision models utilize model-assisted labeling for segmentation and detection tasks.
- VPT is a minecraft agent that uses an inverse dynamic model automatically label Minecraft gameplay videos with their original keyboard action inputs.
- [[Goat]] finetunes LLaMA on a generated arithmetic dataset that encompasses accurate and precise mathematical rigor.
- ROSIE and CACTI are robotic visual data augmentation methods that use diffusion models for semantic visual data augmentation.
- [[DIAL]] is a robotic language augmentation method that uses [[CLIP]] for generating language instructions language instructions or augmenting existing instructions for robotic trajectory datasets.
- [[Alpaca]] and [[Vicuna]] are instruction-following LLMS that finetunes [[LLaMA]] on GPT-3 and ChatGPT outputs.
- [[InstructPix2Pix]] is an instruction  following text-to-image generation model that *generates a dataset* by combining instructions from LLMs with [[Stable Diffusion]] to generate images.
- Synthetic generated images from Stable Diffusion can improve downstream classification accuracy.

---
## Conclusion
- So what's the optimal finetuning strategy for projecting desired priors *onto existing foundation models*?
	- ==This is the trillion-dollar question==
	- And it's being actively explored by a plethora of exciting research touched upon in this overview.

![[Pasted image 20240121014202.png]]



But already, there are some actionable suggestions one can conclude:
- Does the original training corpus contain *all the capabilities and priors you desire?*
	- If YES: Try [[Prompting]]
	- If NO: Try [[Supervised Fine-Tuning|Fine-Tuning]]
- Is it easy to source different finetuning datasets?
	- If Yes, try [[Supervised Fine-Tuning]]
	- If No, try [[Reinforcement Learning from Human Feedback|RLHF]] or [[Reinforcement Learning from from AI Feedback|RLAIF]]
- Do you have access to a lot of compute?
	- If YES: Finetune the whole model
	- If NO: use [[Parameter-Efficient Fine-Tuning|PEFT]]
- Are existing AI models good enough for data generation or data verification?
	- If good enough for data generation, try creating synthetic data!
	- If good enough for *verification, but not generation*, try using [[Reinforcement Learning from from AI Feedback|RLAIF]]or *self-reflection*
		- If neither, stick to RLHF


Zooming out a bit -- it's important to recognize AI acceleration of prior amplification as a double-edged sword:
- As models become increasingly utilized in various components of the data curation process... the pre-existing priors in these AI models ALSO get passed on -- both the desirable and undesirable priors.
- Each of the finetuning methods discussed can be applied iteratively *many times*, with each generation of finetuned "student" models acting as the "teachers" of the next generation!
	- So over time, the original source of specific priors starts to get obfuscated compared to the simple lineage of model training in the past.

These are difficult problems to think about, but they're now one of the core problems in modern AI. ==Priors are everywhere and everything.== Shaping and amplifying them correctly in the context of massive internet-scale data distributions is now the next frontier in modern AI -- the study of digital domestification.


# What a great article! Nice!
































