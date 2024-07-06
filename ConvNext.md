January 10, 2022 (same year, but after [[Vision Transformer]] paper)
Paper: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

See also: ConvNeXt V2

Abstract
> The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers (ViTs), which quickly superseded ConvNets as the state-of-the-art ***image classification*** model. A vanilla ViT, on the other hand, faces difficulties when applied to general computer vision tasks such as object ***detection and semantic segmentation***. It is the hierarchical Transformers (e.g., [[Swin]] Transformers) that reintroduced several ConvNet priors, making Transformers practically viable as a generic vision backbone and demonstrating remarkable performance on a wide variety of vision tasks. However, the effectiveness of such hybrid approaches is still largely credited to the intrinsic superiority of Transformers, rather than the inherent inductive biases of convolutions. ==In this work, we reexamine the design spaces and test the limits of what a pure ConvNet can achieve==. We gradually "modernize" a standard ResNet toward the design of a vision Transformer, and discover several key components that contribute to the performance difference along the way. ==The outcome of this exploration is a family of pure ConvNet models dubbed ConvNeXt==. Constructed entirely from standard ConvNet modules, ConvNeXts ==compete favorably with Transformers== in terms of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, ==while maintaining the simplicity and efficiency of standard ConvNets==.



![[Pasted image 20240701111548.png]]
A 4x4 konvolution kernel with a stride of 4 extracts 

![[Pasted image 20240701111604.png]]
Like [[MobileNet]], uses Depthwise Seperable Convolution

![[Pasted image 20240701111618.png]]

![[Pasted image 20240701111629.png]]
Other tricks: Using MobileNet inverted bottlenecks, Layernorm instead of Batchnorm, new activations, etc.

![[Pasted image 20240701111650.png]]
Still perform well for high-resolution images, because high-resolution images will scale quadratically with attention in a Transformer, but only linearly for convolutions.