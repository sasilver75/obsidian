May 9, 2023 -- [[Meta AI Research]]
Paper: [ImageBind: One Embedding Space to Bind Them All](https://arxiv.org/abs/2305.05665)

A paper on multimodal models where they're able to learn a joint embedding across six different modalities! Interestingly, you don't need to have every record in your training corpus have *all six modalities* -- it turns out that just having Image-{ModalityA}, Image-{ModalityB} is enough to (eg) bind {ModalityA}-{ModalityB}. It's the power of images!

This enables cool cross-modal retrieval applications.

Abstract
> We present ==ImageBind==, an ==approach to learn a joint embedding across six different modalities== - ==images, text, audio, depth, thermal, and IMU data==. We show that ==all combinations of paired data are not necessary to train such a joint embedding, and only image-paired data is sufficient to bind the modalities together==. ImageBind can ***leverage recent large scale vision-language models, and extends their zero-shot capabilities to new modalities*** just by using their natural pairing with images. It enables novel emergent applications 'out-of-the-box' including cross-modal retrieval, composing modalities with arithmetic, cross-modal detection and generation. The emergent capabilities improve with the strength of the image encoder and we set a new state-of-the-art on emergent zero-shot recognition tasks across modalities, outperforming specialist supervised models. Finally, we show strong few-shot recognition results outperforming prior work, and that ImageBind serves as a new way to evaluate vision models for visual and non-visual tasks.

![[Pasted image 20240420192217.png]]

![[Pasted image 20240420192345.png]]

![[Pasted image 20240420192405.png]]