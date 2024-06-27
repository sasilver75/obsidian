https://www.youtube.com/watch?v=rCFvPEQTxKI&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB

The speaker, Song Han (from which the MIT Han lab gets its name), is a BS @ Tsinghua, PhD @ Stanford, 2x startup exit to AMD/NVIDIA

![[Pasted image 20240626180615.png]]


Why do we need to make AI models more efficient?
![[Pasted image 20240626180735.png|400]]
The green line shows computing capacity in modern GPUs, and the red line shows the number of parameters in language models (this comparison isn't straightforward, but it shows that model parameters are growing at a much faster rate than the hardware is improving)
- (Note that there are other ways of increasing aggregate hardware performance, e.g. through parallelization)

Model Compression and Efficient AI aim to compress the model, reducing the complexity of the model via:
- [[Model Pruning|Pruning]]
- Sparsity
- [[Quantization]]
As well as hardware systems that can handle running these techniques.


![[Pasted image 20240626181014.png]]
Lots of growth in the number of publications in these areas over time.

In this first lecture, we'll look at several amazing examples of progress across
- Vision models
- Language models
- Multimodal models

And highlight how much *computing* is under the rug, and why we need lightweight and fast machine learning models.

## Deep Learning for Image Classification

![[Pasted image 20240626181138.png|400]]
High accuracy comes at the cost of larger models; there's no free lunch!  (Note: MAC=Multiply-Accumulate operations; similar to FLOPs)
So how do we push these models into the top-left area, where performance is high, but cost is low?

![[Pasted image 20240626181251.png]]
These are some efficient models that have been recently introduced!
(Note that ImageNet is perhaps not the world's most difficult benchmark to near-saturate).

![[Pasted image 20240626181439.png]]
For use on mobile devices, we need to make our models small and efficient enough to run on limited hardware.

We can go even smaller than phones, too!
![[Pasted image 20240626181522.png]]
Maybe on a microcontroller or IOT device that only costs $5? Can we shrink models and improve the efficiency of our inference engines to enable the ability to run models on extremely limited hardware? System and algorithm co-design is of interest here.

Not only inference -- what about *training* models on the edge? We might want our models to constantly be able to quickly adapt to new data, collected locally!
![[Pasted image 20240626181727.png]]
On edge devices, we don't always want to transfer to the cloud -- perhaps for privacy issues! How do we fine-tune models on the edge to give us better privacy lower cost, and customization?
- Training is much more expensive than inference, so it's hard to fit this on limited hardware!

(Shows an example of a microcontroller with only 1MB of memory; they had a model detecting "person" vs "no person")

There's a big trend right now around incorporating prompts into image recognition tasks, like image segmentation.
- The [[Segment Anything Model]] lets us segment multiple things simultaneously, or interactively via point prompts.
- ![[Pasted image 20240626182109.png]]
- SAM runs at 12 images per second due to the large vision transformer model (ViT-Huge)

Q: What are vision transformers, and why are they so complex?
A: We'll cover more of these later!

![[Pasted image 20240626185623.png]]
An EfficientViT implementation was able to increase the number of images that could be segmented by a huge factor!


![[Pasted image 20240626185650.png]]
So far, we've been talking about discriminative models (classification, segmentation, etc.)

![[Pasted image 20240626185937.png]]
Accelerating the FPS of this CycleGAN style transfer video thing from 12 FPS to 40FPS

![[Pasted image 20240626185956.png]]
Running Photo-editing model locally on a limited-power laptop
- Training a smaller or larger sub-network that's low-cost and able to do fast prototyping
- At the end, running the full model to get the high quality result

![[Pasted image 20240626195750.png]]

1855 Giga Maccs to just 514 Giga Maccs (3.6x smaller)
MAC = Multiply and Accumulate (Add)

![[Pasted image 20240626200049.png]]
Can we generate something that exists in real life? Can we create a personalized image that looks like ourself, like our friend, like our pet, etc. We can do that using existing work where we finetune a model to learn a new subject.
Problems though:
- Generating multiple people that fit a description
- Overfitting to the provided subject; it's hard to generate FeiFei Li riding a horse, because the model is so fit to the idea of FeiFei.


![[Pasted image 20240626200726.png]]
Not only 2D images, but 3D objects from a natural language description!
- This is far less good than 2d object generation, but still making good progress

![[Pasted image 20240626200752.png]]
Not only static objects, but videos -- multiple frames! Shown is [[Imagen]].
- But this is super computationally heavy; Imagen is 5.6B params, which the person said is "really big"

![[Pasted image 20240626201018.png]]
LiDAR is an important sensor in self-driving; accelerating 3D perception with efficient techniques is important, because we need to make predictions with high frequency!
- The point clouds that are generated are very sparse!

![[Pasted image 20240626201152.png]]
How do we fuse information across many modalities from multiple sensors efficiently -- without having our car trunk full of computers?



---

## Deep Learning for NLP

![[Pasted image 20240626201358.png]]
Let's talk about language now!


ChatGPT and LLMs can produce human-like text based on past conversations.
- But the models are pretty popular, and often times ChatGPT has to be throttled ("You have 3 messages remaining until Noon") because they just don't have enough compute to support users, because the models are so intensive!


![[Pasted image 20240626201612.png]]
We perhaps don't want to upload all of our context of our coding projects (or our company's IP) to the cloud; we might want it to stay local to our company.


![[Pasted image 20240626201642.png]]
Neural Machine Translation software is available offline, on-device, in many cases now!
- HAN lab worked on a project called ==Lite Transformer== to reduce model size with pruning and quantization.


![[Pasted image 20240626201802.png]]
LLMs are able to do zero/few-shot learning. Both sort of forms of [[In-Context Learning]].
![[Pasted image 20240626201953.png]]
I dunno why he's showing CoT in an efficient NLP course lol

For these examples, we "change" the model, but we don't change any parameters, we just change the input to the model.
![[Pasted image 20240626202400.png|500]]
It costs a lot to train LLMs. The picture is one of his Lab's server. That's one rack, and there are a couple of them. Infiniband switches connects multiple machines so we can act as a large GPU. Each cable is a few thousand dollars.
- The Infiniband on his is 200Gbps and he bought it a year ago; now, the SoTA is 400Gbps!


![[Pasted image 20240626202539.png]]
Sparse attention/token pruning! 
- We don't have to attend to every token in the sentence!
We can remove a lot of redundant information from a sentence!


![[Pasted image 20240626202756.png]]
In cars, on robots, on spaceships where there is no internet connection!
We want to preserve privacy!

![[Pasted image 20240626202829.png]]
This is a Macbook Air running a LLaMa 7B model.
- This is achieved by compressing the model to 4Bit, and designing an efficient inference engine to run directly on the 4bit quantized model. This is on the M1 chip, but it also works on Intel chips.
- The project is called TinyChat; feel free to download on github.
	- It also funs Falcon MPT, and Vicuna

---

## Deep Learning for Multimodal Models

![[Pasted image 20240626203252.png]]


Combining vision, language, and more!

![[Pasted image 20240626203307.png]]
[[LLaVA]] combines vision and language understanding. We ask it who's in the painting, and it responds that it's Mona Lisa! We want to compress these multimodal models as well!
	- With ==[[Activation-Aware Weight Quantization|AWQ]]== , we quant







