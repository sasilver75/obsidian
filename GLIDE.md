---
aliases:
  - Guided Language to Image Diffusion for Generation and Editing
---

December 20, 2021 -- [[OpenAI]] (Authors include [[Ilya Sutskever]])
Paper: [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741)

A paper on [[Diffusion Model]] from OpenAI where they compare [[CLIP]] guidance and [[Classifier-Free Guidance]], and find that the latter (CFG)  is preferred by human evaluators.
They also do some playing around with image inpainting (think: Dingboard)


Abstract
> Diffusion models have recently been shown to generate high-quality synthetic images, especially when paired with a guidance technique to trade off diversity for fidelity. ==We explore diffusion models for the problem of text-conditional image synthesis== and compare two different guidance strategies: ==CLIP guidance== and ==classifier-free guidance==. We find that the ==latter is preferred by human evaluators== for both photorealism and caption similarity, and often produces photorealistic samples. Samples from a 3.5 billion parameter text-conditional diffusion model using classifier-free guidance are favored by human evaluators to those from DALL-E, even when the latter uses expensive CLIP reranking. Additionally, we find that our models can be fine-tuned to perform image ==inpainting==, enabling powerful text-driven image editing. We train a smaller model on a filtered dataset and release the code and weights at [this https URL](https://github.com/openai/glide-text2im).

![[Pasted image 20240420191747.png]]
GLIDE generations

![[Pasted image 20240420191820.png]]
Inpainting

![[Pasted image 20240420191835.png]]
Iterative Inpainting