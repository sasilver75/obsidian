September 25, 2024
[[Allen Institute|AI2]]
[Molmo/Pixmo Blog Post](https://molmo.allenai.org/blog)

References:
- [Hannaneh Hajishirzi - OLMo: Accelerating the Science of Language Modeling (COLM)](https://youtu.be/qMTzor0j418?si=9Wrct3p5baAfq5Nx)

Released as part of the [[Molmo]] release from the Allen Institute.
- Contains less than 1M image-text pairs, in two broad categories:
	1. Dense captioning data for multimodal pre-training
	2. Supervised fine-tuning data for enabling a wide array of user interactions, including behaviors like [[Question Answering]], Document Reading, and Pointing.
		- Fine-tuning data comes from both standard academic datasets as well as newly-collected datasets which will be released.

Datasets include:
- ==PixMo-Cap== (For pretraining VLMs to understand images in great detail; 712k distinct images, 1.3M dense image captions. Captions generated by human annotators via 60-90s spoken descriptions, transcribed by language models)
- ==PixMo-AskModelAnything== (162k QA pairs for 73k images)
- ==PixMo-Points== (2.3M question-point pairs from 428k **images**)
- ==PixMo-CapQA== (214k QA pairs generated from 165k image captions using a LM; diverse topics/styles)
- ==PixMo-Docs== (Text and figure-heavy images (charts/documents/tables/diagrams) with corresponding code, as well as 2.3M QA pairs based on generated code.)
- ==PixMo-Clocks== (826k diverse clock images with questions and answers around time)

Blog Post Excerpts:
> "Our primary constraint in the collection of this data is to avoid making use of existing VLMs, since we want to build a performant VLM ==from the ground-up, rather than by distillation of an existing system== (note that we do make use of language-only LLMs, but we never pass images to these models)."
> "In practice, ==it is challenging to collect dense captioning datasets from human annotators==. If asked to write an image description, the result often only mentions a few salient visual elements and lacks detail. If a minimum word count is enforced, annotators will either take too long to type, making collection uneconomical, or copy-and-paste responses from proprietary VLMs, circumventing our goal to avoid distillation. As a result ==the open research community has struggled to create such datasets== without relying on synthetic data from proprietary VLMs."
> "Our key innovation is a simple but effective data collection methodology that avoids these problems: ==we ask annotators to describe images in speech for 60 to 90 seconds rather than asking them to write descriptions=="
> "We prompt the annotators to describe everything they see in great detail and include descriptions of spatial positioning and relationships. Empirically, we found that with this modality switching "trick" annotators provide far more detailed descriptions in less time, and for each description we collect an audio receipt (i.e., the annotator's recording) proving that a VLM was not used. In total, we collected detailed audio descriptions for 712k images that were sampled from 50 high-level topics."
> "Pointing provides a natural explanation grounded in image pixels resulting in new and improved capabilities for Molmo. We believe that in the future pointing will be an important communication channel between VLMs and agents."




-----

# non-Paper figures

![[Pasted image 20241101021048.png]]

![[Pasted image 20241101021055.png]]

![[Pasted image 20241101021104.png]]