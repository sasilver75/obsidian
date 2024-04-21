
May 23, 2022 -- [[Google Brain]]
Paper: [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487)

A relatively-early text-to-image [[Diffusion Model]] from Google.
It seems from the abstract like the use the Encoder portion of a [[T5]] model to create text latents, and then use a diffusion process to generate images from this latent.
- Intriguingly, they find that the images produced from a text latent are as aligned as the image/text alignment found in the [[Common Objects in Context]] dataset!

Abstract
> We present ==Imagen==, a ==text-to-image diffusion model== with an unprecedented degree of photorealism and a deep level of language understanding. Imagen builds on the power of large transformer language models in understanding text and hinges on the strength of diffusion models in high-fidelity image generation. ==Our key discovery is that generic large language models (e.g. T5), pretrained on text-only corpora, are surprisingly effective at encoding text for image synthesis==: increasing the size of the language model in Imagen boosts both sample fidelity and image-text alignment much more than increasing the size of the image diffusion model. Imagen achieves a new state-of-the-art FID score of 7.27 on the COCO dataset, without ever training on COCO, and ==human raters find Imagen samples to be on par with the COCO data itself in image-text alignment==. To assess text-to-image models in greater depth, we introduce DrawBench, a comprehensive and challenging benchmark for text-to-image models. With DrawBench, we compare Imagen with recent methods including VQ-GAN+CLIP, Latent Diffusion Models, and DALL-E 2, and find that human raters prefer Imagen over other models in side-by-side comparisons, both in terms of sample quality and image-text alignment. See https://imagen.research.google/ for an overview of the results.

![[Pasted image 20240420193155.png]]