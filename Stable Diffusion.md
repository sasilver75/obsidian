Paper: December 20, 2021
Paper:  [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (Latent Diffusion Models)
*Stable Diffusion* is actually a 2022 open-source reimplementation of the paper described above.

Abstract:
> By ==decomposing the image formation process into a sequential application of denoising autoencoders==, diffusion models (DMs) ==achieve state-of-the-art synthesis results on image data== and beyond. Additionally, their formulation ==allows for a guiding mechanism to control the image generation process== without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, ==we apply them in the latent space of powerful pretrained autoencoders==. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Our ==latent diffusion models (LDMs)== achieve a new state of the art for image inpainting and highly competitive performance on various tasks, including unconditional image generation, semantic scene synthesis, and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs. Code is available at [this https URL](https://github.com/CompVis/latent-diffusion) .


# Paper Figures



# Non-Paper Figures
![[Pasted image 20240710034747.png]]