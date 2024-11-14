May 27, 2024
[[Meta AI Research]] (Bordes et al.; large number of authors)
Paper
#zotero 
Takeaway:  ...

This was [recommended](https://x.com/andrew_n_carr/status/1856140174134259925) by Andrew Carr @ Cartwheel as a good paper to skill up on VLMs with.

----

# (1/6) Introduction
- Connecting language models to vision will unlock several key applications... but it's still not a solved problem.
	- Models struggle to understand spatial relationships
	- Models struggle to count without complicated engineering overhead
	- Models lack an understanding of attributes and ordering
	- Models often ignore some part of the input prompt, leading to significant prompt engineering efforts

==This work should not be considered as a survey or a complete guide on VLMs.== (but the footnotes do link out to multiple of these)

We aim to provide a clear and easy-to-understand introduction to VLM research, and highlight effective practices for research in this area.

We will:
1. Present different VLM training paradigms
2. Discuss how contrastive methods changed the field
3. Present methods that leverage masking strategies or generative components
4. Present VLMs which use pre-trained backbones (eg LLMs)
5. Which datasets are appropriate, given a research goal?
6. What data curation strategy to use?
7. Is contrastive loss enough for vision understanding, or do we need a generative component?
8. Grounding and alignment techniques
9. Strengths and weaknesses of VLM benchmarks
10. VLMs that process videos


# (2/6) The families of VLMs
- We categorize recent initiatives into four different training paradigms:
	- ==Contrastive training==
		- Leverages positive and negative examples
		- VLM is then trained to predict similar representations for the positive pairs, while predicting different representations for the negative pairs.
	- ==Masking==
		- Leverages reconstruction of masked image patches given some unmasked text
		- Leverages reconstruction of masked text, given an unmasked images
	- ==Pretrained backbones==
		- Leverage open-source LLMs like LLaMA to learn a mapping between an image encoder (Also often pretrained) and the LLM.
	- ==Generative VLMs==
		- Can generate images or captions. Given the nature of these models, they're often the most expensive to train.

==These approaches are not mutually exclusive -- many approaches rely on a mixture of contrastive, masking, and generative criteria.==

In the beginning, there was early work to extend [[BERT]] to process visual data. visual-BERT and ViL-BERT combine text with image tokens, with the models trained on two objectives:
1. Classical masked modeling task that aims to predict the missing part in a given input
2. A sentence-image prediction task that aims to predict the missing part in a given input
Model learns to associate words with visual clues.


Contrastive-based VLMs
- Contrastive-based training is often better explained through an ==Energy-based Models (EBM)== point of view, in which a model is trained to assign low energy to *observed* variables and high energy to *unobserved* ones.
- Data from a target distribution should have low energy while *any other data points* should have high energy.


# (3/6) A guide to VLM training
- 


# (4/6) Approaches for Responsible VLM Evaluation
- 


# (5/6) Extending VLMs to Videos
- 


# (6/6) Conclusion



------

# Paper Figures

![[Pasted image 20241114014408.png]]
