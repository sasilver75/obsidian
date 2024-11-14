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


## Contrastive-based VLMs
- Contrastive-based training is often better explained through an ==Energy-based Models (EBM)== point of view, in which a model is trained to assign low energy to *observed* variables and high energy to *unobserved* ones.
- Data from a target distribution should have low energy while *any other data points* should have high energy.

There are various ways to get $x \sim P_\theta(x)$, which is required by a few methods:
- [[Markov Chain Monte Carlo]] (MCMC)
- Score Matching
- Denoising Score Matching
- ==[[Noise-Contrastive Estimation]] (NCE)== 
	- This is what most recent works on self-supervised learning and VLM are based.

The original ==NCE== framework can be described as a ==binary classification problem== in which a model should predict the label C=1 for samples from the real distribution, and C=0 for those coming from the noise distribution. By doing so, the model learns to discriminate between the *real* datapoints and the *noisy* ones.
- The loss function is for a binary classification, using cross entropy. 


Oord (2018) built on some work from Wu (2018) and coined an approach known ==as [[InfoNCE]],== such that:
![[Pasted image 20241114110236.png]]
==Instead of predicting a binary value==, the InfoNCE loss leverages a *==distance metric==*, such as cosine similarity, computed in model representation spaces.
- This requires ==computing the distance between the positive pairs of examples and all negative pairs of examples==.
- ==The model learns to predict==, via the softmax, ==the most likely pair of examples that is closest to the representation space==, while associating lower probability to all other pairs of negative examples.

For self-supervised methods like [[SimCLR]] (2020), a positive pair of examples is defined as one image and its corresponding handcrafted data-augmented version (eg apply grayscaling on the original image), while the negative pairs of examples are built using one image and all other images that are present in a mini-batch.

==The major drawback of InfoNCE-based methods== is the introduction of a dependence on *mini-batch content*. This often requires *large mini-batches* to make the contrastive training criterion between the positive and negative samples more effective.
- ((I've heard varying claims about this -- is it the NUMBER of negatives, or is it the QUALITY of negatives? Unsure))

### CLIP
- A common contrastive method using an [[InfoNCE]] loss is [[CLIP|Contrastive Language-Image Pretraining]] (CLIP; Radford 2021)
- The positive pairs of examples are defined as one image, and its corresponding ground-truth caption
- The negative examples are the same image but with all other captions contained in the mini-batch.
- ==A novelty of CLIP: Incorporates vision and language in a shared representation space.==
	- Uses a contrastive loss to map the representation of an image and its caption ti similar embedding vectors
- Train on 400 million caption-image pairs.

- [[SigLIP]]
	- Zhai (2023)'s SigLIP is similar to CLIP with the exception that it uses the ==original NCE loss== based on a binary cross-entropy, ==rather than CLIP's multi-class objective based on InfoNCE==.
		- ((==That's interesting! So is InfoNCE not as "good"?==))
	- This change enables better 0-shot performance on smaller batch sizes than CLIP

- [[Llip]] 
	- Lavoie (2024) ==accounts for the fact that an image can be captioned in several different ways==
	- Proposes to ==condition the encoding of an image on the target caption via a cross-attention module==.
	- Accounting for the caption diversity increase the representation's expressivity and it generally improves the downstream zero-shot transfer classification and retrieval performance.

## VLMs with masking objectives
- Masking is a commonly-used technique in DL research
	- It's a specific form of a denoising autoencoder, in which the noise has a spatial structure (( ok sure, lol ))
	- It's related to *inpainting strategies* used to learn visual representations, and to [[Masked Language Model|Masked Language Modeling]] strategies like those used to train [[BERT]].
- There have been several works on the vision side to use [[Masked Image Model|Masked Image Modeling]] (such as MAE (2022) or I-JEPA (2023)). 
- There have been works that have combined both techniques to train VLMs:
	- FLAVA (2022) leverages several training strategies including masking to lean text and image representations.
	- MaskVLM (2023) is a standalone model.

### FLAVA
- A first example of the masking-based approach is the Foundational Language and Vision Alignment ([[FLAVA]]) paper (Singh 2022).
- Its architecture comprises three core components, each transformer-based:
	1. The Image Encoder employs a [[Vision Transformer|ViT]] to process images into patches for linear embedding and transformer-based representation, including a classification token (\[CLS\]). Trained using a masking approach.
	2. The Text Encoder tokenizes textual input using a transformer and embeds them into vectors for contextual processing and outputting hidden state vectors alongside a classification token \[CLS_T\]. Trained using masking approach.
	3. A Multimodal Encoder that fuses hidden states from both the image and text encoders, leveraging learned linear projections and cross-attention mechanisms within the framework to integrate visual and textual information, highlighted by an additional multimodal classification token (\[CLS_M\])

Comprehensive training regimen that combines multimodal and unimodal masked modeling losses along with a contrastive objective. It's pretrained on a dataset of 70M publicly-available image and text pairs.

### MaskVLM
- A limitation of FLAVA is the use of pre-trained vision encoders such as dVAE. ((??))
- To make a VLM that's less dependent on third-party models, [[MaskVLM]] (Kwon 2023) applies masking directly in the pixel space and in the text token space.
- One of the keys to make it work across both text and image is to ==use the flow of information coming from one modality to the other==... the text reconstruction task receives the information coming from the image encoder, and vice versa.

## Generative-Based VLMs
- In contrast ot predvious training paradigms, where we operate on latent representations to build image or text abstractions that are then mapped between eachother, the *genreative* paradigm considers the *generation* of text/images!
- Some methods like
	- [[Contrastive Captioner|CoCa]] (Contrastive Captioner; Yu 2022b) learn a complete text encoder and ***==text decoder==***, enabling image captioning.
	- [[Chameleon]] (2024) and CM3leon (Yu, 2023) are multi-modal generative models that are explicitly trained to generate ***==both text and images==***.
	- [[Stable Diffusion]] (2022), [[Imagen]] (2022) and others are trained to generate images based on text.

### Text generator: CoCa
- Besides the contrastive loss that works well in CLIP, the [[Contrastive Captioner]] (CoCa) model (Yu 2022) *ALSO* employs a generative loss, which is the loss corresponding to captions generated by a multimodal text decoder that takes in:
	- (1) image encoder outputs
	- (2) representations produced by the unimodal text decoder as inputs.
		- ((Wait, there's a unimodal text decoder AND a multimodal text decoder? What text is the unimodal text decoder processing? Isn't this an image captioning model? Hmmm))
- The new loss allows for the ability to perform multimodal understanding tasks (eg VQA)
- Pretrains on two dataset:
	- [[ALIGN]] (I guesss the dataset from it), with 1.8B images with alt text
	- JFT-3B (An internal dataset at Google Research)

### Multimodal generative models: Chameleon and CM3leon
- [[CM3Leon]] (Yu et al, 2023) is a foundation model for text-to-iamge and image-to-text generation!
	- Tokenizastion approach that allows the model to process interleaved text and images; they tokenized images and texts are then passed to a decoder-only transformer model.
	- Two stage training process:
		1. ==Retrieval-augmented pretraining==: Uses a [[CLIP]]-based encoder as a dense retriever to fetch relevant and diverse multimodal documents and prepends these documents to an input sequence. The model is then trained using NTP on the input sequence. 
		2. ==Supervised Fine-tuning (SFT)==, where the model undergoes multi-task instruction tuning. Allows the model to process and geneate content across different modalities.

And extension to this work is [[Chameleon]]
- A new series of mixed-modal foundation models that can generate and reason with mixed sequences of *==interleaved==* textual and image content.
- Designed to be mixed-modal from the beginning, trained from scratch in an end-to-end manner on a blend of all modalities -- image, text, and code.
- By converting images into discrete tokens, similar to words in text, the same transformer architecture can be applied to sequences of both image and text tokens without needing separate encodes for each modality. This is an [[Early Fusion]] strategy. It enables seamless reasoning across modalities, but also introduces significant technical challenges.

### Using generative text-to-image models for downstream vision-language tasks
- Models like [[Stable Diffusion]] and [[Imagen]] are diffusion models, but there are also autoregressive image-generation models like Parti.
- The focus has been on their GENERATIVE abilities, but they can actually be directly used for *discriminative* tasks like classification or caption prediction without retraining.
- (Skipping some of this. It's sort of interesting, but not my thing right now)

## VLMS from Pretrained BAckbones
- A downside of VLMs is that they'er costly to train from scratch, requiring hundreds to thousands of GPUs while having to use hundreds of millions of images and text pairs.
- Instead, let's leverage existing LLMs and existing visual extractors, and learn a mapping between them!

### Frozen
- Frozen (2021) is a first example of a model leveraging a pretrained LLM. Proposes to connect vision encoders to a FROZEN language model through a lightweight mapping network which projects visual featuers to text token embeddings.
- ==The vision encoder and the linear mapping are trained from scratch, while the language model is kept frozen== (this is crucial to maintain the features that the pretrained model had already learned).
- This model is supervised with a simple text generation objective on the Conceptual Captions dataset (2018); At inference time, the language model can be conditioned on interleaved text and image embeddings.
- ==Achieved only modest performance, but was a first step towards the current multimodal LLMs==

### MiniGPT
- Starting from models like [[Flamingo]] (2022), a recent trend is to train multimodal language models where the input contains text and images, and the output contains text.
- In [[MiniGPT-4]] (Zhu, 2023), a simple linear projection layer is used to align image representations (using the visual encoder from [[BLIP-2]], which is based on Q-Former and a ViT backbone) with the input space of teh [[Vicuna]] language model.
- 




 
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
Above:
- Contrastive
- Masking
- Generative
- Pretrained backbones
These are not exclusive means of training a VLM; you can use either.

