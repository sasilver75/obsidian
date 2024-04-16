#article 
Link: https://magazine.sebastianraschka.com/p/ahead-of-ai-10-state-of-computer?utm_source=publication-search

---

Large language model development (LLM) development is still happening at a rapid pace, but at the time of writing (July 2023), they were in a mini-lull, so Sebastian thought it'd be a good time to cover computer vision, after having just got back from CVPR '24 (Computer Vision & Pattern Recognition)

---

# Articles and Trends
- CVPR 2023 had a total of 2359 papers. resulting in a vast array of posters for participants to explore! 

Sebastian saw that ==most research focused on one of the following 4 themes:==
1. Vision Transformers ([[Vision Transformer|ViT]]) ((Sorta New))
2. Generative AI for Vision: Diffusion Models and GANs ([[Diffusion Models]], [[Generative Adversarial Network|GAN]]s) ((Sorta New))
3. [[Neural Radiance Fields]] (NeRFs)  ((New!))
4. Object Detection and Segmentation ((The Classics))


# (1/4) Vision Transformers
- Following in the footsteps of successful language transformers and LLMS, [[Vision Transformer]]s originally appeared in 2020 with *An Image is Worth 16x16 Words* paper.
- ==The main concept of ViTs is similar to that of language transformer; it uses the same self-attention mechanisms in its multi-head attention blocks -- however, instead of tokenizing words, ViTs are tokenizing *IMAGE PATCHES*==
- The original ViT model resemble encoder-like architectures similar to BERT, encoding embedding image patches. We can then attach a classification head for an image classification task.
- ==Note that ViTs typically have many more parameters than convolutional neural networks (CNNs)==, and, as a result, generally require more training data to achieve good modeling performance -- this is why we typically adopt *pretrained* ViTs instead of training them from scratch.
	- ((This sort of makes sense that they might require more parameters than convolutional neural networks, since it feels like (maybe?) they have less of an inductive bias))

ViTs like [[DeiT]] and [[Swin]] are extremely popular architectures due to their state-of-the-art performance on computer vision tasks.

Still, one major criticism of ViTs is that they are relatively resource-intensive and less efficient than CNNs.


### Efficient ViT: Memory-Efficient Vision Transformer with Cascaded Group Attention
- ViTs are relatively resource intensive, holding them back from more widespread adoption in practice -- in the CVPR paper *==Efficient ViT: Memory Efficient Vision Transformer with Cascaded Group Attention==*, researchers introduce a new, efficient architecture to address this limitation.
	- (Matches the accuracy of EfficientNet (a CNN), and is faster than MobileNet!)

The main innovations in this paper include using:
1. A single memory-bound multihead self-attention (MHSA) block between fully connected layers (FC layers)
	- ((This sounds strange to me, since I'm usually not used to attention layers being *sandwiched* by FFNNs in the way it sounds like they're describing.))
2. ==Cascaded group attention==

Let's start with point 1: The MHSA sandwiched between FC layers. According to other studies, memory inefficiencies are mainly caused
- Studies show that memory inefficiencies ((?)) are mainly caused by MHSA, rather than FC (fully connected) layers. Addressing this, researchers use additional FC layers to allow more communication between feature channels... but reduce the number of attention layers to 1, in contrast to (eg) the popular Swin Transformer 🤔

Point 2: The cascaded attention group here is inspired by ==group convolutions==, which were used way back in [[AlexNet]].
- [[Grouped Convolutions]] are a variation of the standard convolution operation. In contrast to regular convolutions *group convolutions* divide the input channels into several groups! Each group performs its own convolution operations independently!
	- So if an input has 64 channels and the grouping parameter is set to 2, the input would be split into two groups of 32 channels each, and these groups would then be convolved independently. This approach reduces computational cost and can also increase model diversity by enforcing a kind of regularization, leading to potentially improved performance in some tasks.

![[Pasted image 20240415211855.png]]


# (2/4) Generative AI for Vision: Diffusion Models
- Open source models like [[Stable Diffusion]] reimplemented the model proposed in Dec 2021's *High-Resolution Image Synthesis with Latent Diffusion Models*.

Recap:
- Diffusion models are fundamentally generative models.
- During training, random noise is added over the input data over a series of "timesteps" to increasingly perturb it.
- Then, in a reverse process, the model learns how to *invert* the noise (predict the noise), effectively denoising the output, recovering the original data.
- During inference, we then use a diffusion model to generate new images *entirely from noised inputs!*

Most models employ a [[U-Net|UNet]] based on a [[Convolutional Neural Network|CNN]], encoding of an encoder part to capture context and an encoder that allows precise localization ((?)).

Diffusion models traditionally repurposed U-Net to model the conditional distribution at each diffusion step, providing the mapping from a noise distribution to the target data distribution. In other words, ==the U-Net is used to predict the noise to be added or subtracted at each step of the diffusion process.==
- They're particularly attractive because they can combine local and global information.

![[Pasted image 20240415213314.png|500]]


### All are Worth Words: A ViT Backbone for Diffusion Models
- In the ==All are Worth Words: A ViT Backbone for Diffusion Models== paper (Sep 2022), researchers try to swap the *convolutional* U-Net backbone with a ViT, which they refer to as a U-ViT (Note that this isn't the first attempt to do it -- but it seems to be the best so far)

- The main contributions are additional "long" skip connections and an additional convolutional block before the output. Note that the "long" skip connections are in addition to the regular skip connections in the transformer blocks.

Similar to regular ViTs, the inputs are "patchified" images, plus an additional ==*time token*== (for the diffusion step) and *==condition token==* (for class-conditional generation).

![[Pasted image 20240415214058.png|400]]

It's refreshing to see that the paper was accepted even though the model didn't outperform all other models on several tasks -- it's a new model with the new architectural ideas.


# (3/4) Neural Radiance Fields (NeRF)
- [[Neural Radiance Fields]] are a relatively new method (2020) for synthesizing novel views of complex 3D scenes from a set of 2D images.
- This is accomplished by modeling scenes as volumetric fields of NeuralNet-generated colors and densities.
- ==The NeRF model, trained on a small set of images taken from different viewpoints, learns to output the *color* and *opacity* of a point in 3D space, given its 3D coordinates and viewing direction.==
	- ((Oh, I didn't realize that was it. Cool.))

Sine there's a lot of technical jargon in *Neural Radiance Fields*, let's derive where these terms came from!
- *Neural Fields* are a fancy term for describing a neural network that serves as a trainable (or parametrizable) function that generates a "field" of output values across the input space -- a 3D scene!

The idea is to get the network to *overfit* to a specific 3D scene, which can then be used to generated novel views of the scene with high accuracy -- this is somewhat similar to the concept of *spline interpolation* in numerical analysis, where a curve is *overfit* to a set of data points to generate a smooth and precise representation of the underlying function.

![[Pasted image 20240415214803.png]]

In the case of a NeRF, the network learns to output *color and opacity values* for a given point in space, conditioned on the viewing distance, allowing for creation of a realistic 3D scene representation.
- Since the NeRF predicts *color and intensity of light* at every point along a specific viewing direction in 3D space, it essentially models a field of *radiance values,* hence *Radiance Field!*

Note that we technically refer to a NeRF representation as 5D, because it considers three spatial dimensions (x, y, z coordinates in 3D space) and two additional dimensions representing the viewing direction defined by two angles: theta and phi.

### ABLE-NeRF: Attention-Based Rendering with Learnable Embeddings for Neural Radiance Fields
- As described above, the basic idea behind NeRF is to model a 3D scene as a continuous volume of radiance fields.
	- Instead of storing explicit 3D objects or voxel grids, the 3D scene is stored as a function (a neural network) that maps 3D coordinates to colors (RGB values) and densities.
	- This network is trained using a set of 2D images from the scene taken from different viewpoints.
	- When rending a scene, the NeRF model takes as input a 3D coordinate and viewing direction, and outputs the RGB color value and the volume density at that location.

==One o the shortcomings though is that glossy objects often look blurry, and the colors of translucent objects are often murky.==

In the paper *ABLE-NeRF: Attention-Based Rendering with Learnable Embeddings for Neural Radiance Field.*
- Researchers address these shortcomings and improve the visual quality of translucent and glossy surfaces by introducing a *self-attention-based framework and learnable embeddings*.

One additional detail worth mentioning about regular NeRFs is that they use a volumetric rendering equation derived from the physics of light transport.
- The basic idea in volumetric rendering is integrating the contributions of all points along a ray of light as it travels through a 3D scene. The rendering process involves sampling points along each ray, querying the NN for the radiance/density values at each point, and then integrating these to compute the final color of the pixel.

The proposed ABLE-NeRF diverges from such physics-based volumetric rendering and uses an attention-based network instead, which is responsible for determining a ray's color. Added masked attention so that specific points can't attend occluded points, incorporating the physical restrictions of the real world.
They also add another transformer module for learnable embeddings to capture the view-dependent appearance caused by indirect illumination.

![[Pasted image 20240415220332.png]]


# (4/4) Object Detection and Segmentation
- Object detection and segmentation are classic CV tasks, so it probably doesn't require a lengthy introduction -- but to highlight the difference between these tasks:
	- Object detection is about drawing bounding boxes and the associated labels
	- Semantic segmentation classifies each pixel to distinguish between foreground and background objects.

 1. [[Semantic Segmentation]]. This technique labels each pixel in an image with a class of objects (e.g., car, dog, or house). However, it doesn't distinguish between different instances of an object. For example, if there are three cars in an image, all of them would be labeled as "car" without distinguishing between car 1, car 2, and car 3.

2.  [[Instance Segmentation]]. This technique takes semantic segmentation a step further by differentiating between individual instances of an object. So, in the same scenario of an image with three cars, instance segmentation would separately identify each one (e.g., car 1, car 2, and car 3). So, this technique not only classifies each pixel but also identifies distinct object instances, giving it a distinct object ID.

3. [[Panoptic Segmentation]]. This technique combines both semantic segmentation and instance segmentation. In panoptic segmentation, each pixel is assigned a semantic label as well as an instance ID. To differentiate panoptic segmentation a bit better from instance segmentation, the latter focuses on identifying each instance of each recognized object in the scene. I.e., instance segmentation is concerned primarily with "things" - identifiable objects like cars, people, or animals. On the other hand, panoptic segmentation aims to provide a comprehensive understanding of the scene by labeling every pixel with either a class label (for "stuff", uncountable regions like sky, grass, etc.) or an instance label (for "things", countable objects like cars, people, etc.).

Some popular algorithms for object detection include [R-CNN](https://arxiv.org/abs/1311.2524) and its variants ([Fast R-CNN](https://arxiv.org/abs/1504.08083), [Faster R-CNN](https://arxiv.org/abs/1506.01497)), [YOLO](https://arxiv.org/abs/1506.02640) (You Only Look Once), and [SSD](https://arxiv.org/abs/1512.02325) (Single Shot MultiBox Detector).

Models for segmentation include [U-Net](https://arxiv.org/abs/1505.04597) (discussed in the diffusion model section earlier), [Mask R-CNN](https://arxiv.org/abs/1703.06870) (a variant of Faster R-CNN with segmentation capabilities), and [DeepLab](https://arxiv.org/abs/1606.00915), among others.

### Mask DINO: Towards a unified Transformer-based Framework of Object Detection and Segmentation
- An extension of the DINO method (DETR with Improved deNoising anchOr boxes)
- DETR (DEtection TRansformer) is an end-to-end object detection model introduced by FB AI and uses a transformer architecture. 
	- Unlike traditional models, DETR treats object detection as a direct set prediction problem, eliminating the need for handcrafted anchor boxes and non-maximum suppression procedures and providing a simpler and more-flexible approach to object detection.

While CNN-based methods nowadays unify object detection (region-specific task) and segmentation (pixel-level task) to improve the overall performance on both tasks, this isn't the case for transformer-based object detection and segmentation masks.

DINO outperforms all existing object detection and segmentation systems.
![[Pasted image 20240415222121.png]]

