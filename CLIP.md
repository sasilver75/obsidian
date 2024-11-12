---
aliases:
  - Contrastive Language-Image Pretraining
---
February 26, 2021 (8 months after GPT-3)
[[OpenAI]] - Including [[Alec Radford]], [[Ilya Sutskever]], [[Jack Clark]]
Paper: [Learning Transferable Visual Models from Natural Language Supervision](https://arxiv.org/abs/2103.00020)
#zotero 
Takeaway: A zero-shot image classifying model that makes it easy to create your own classes for categorization, "rolling your own classifier" without the need for re-training. Uses contrastive loss between image and text embeddings into a joint space. At inference time, we embed something like `A photo of a {label}` for every label/class in our dataset, and then for every image we want to classify, we compare the image embedding to the (cached) text embeddings for our labels. A great and exhaustive paper.

---
[[Douwe Kiela]] thinks that the CLIP paper is one of the best papers ever written in the field -- extremely thorough and worth a read.

Notes:
- NLP was revolutionized by using task-agnostic objectives like autoregressive/masked language modeling, along with using "text-to-text" as I/O, enabling zero-shot transfer to downstream tasks... but in other fields like CV, it's still standard practice to pre-train models on *crowd-labeled datasets* such as ImageNet. 
- At the time of this paper, using natural language supervision for image representation learning is still rare because demonstrated performance on common benchmarks at the time was much lower than alternative approaches (lacking scale in training).
- Authors create a new dataset of ==400 million (image, text) pairs,== and train simplified version(s) of the ConVIRT model, which they call ==CLIP: Contrastive Language-Image Pretraining== (There are actually 8 CLIP models spanning 2 orders of magnitude in size). The hope is to learn perception from supervision contained in natural language.
	- To construct this dataset, they search for images whose text includes one of a set of 500,000 queries, where this query list is all words occurring at least 100 times in the English version of Wikipedia. They approximately class-balance the results by including ***up to*** 20,000 image/text pairs *per query*.
- Emergent abilities with scale, like GPT: CLIP learns to perform a wide set of tasks during pre-training including [[Optical Character Recognition|OCR]], Geo-Localization, Action Recognition, and many others.
- Authors note that recent work in contrastive representation learnings for images have found that contrastive objectives can learn better representations than equivalent predictive objectives; Other work has found that although generative models of images can learn high-quality image representations, they require over an order of magnitude more compute than contrastive models with the same performance.
- Given a batch of N (image, text) pairs, CLIP is trained to predict which of the N x N possible (image, text) pairings across a batch *actually occurred*. 
	- To do this, CLIP learns a multimodal embedding space by ==jointly training an image encoder (tested ResNet, ViT) and text encoder (Transformer) to maximize the cosine similarity of the image/text embeddings of the N *real pairs* in the batch, while minimizing the cosine similarity of the embeddings of the N^2-N *incorrect pairs*==. They optimize a ***symmetric cross entropy loss*** over these similarity scores. They train CLIP from scratch without initializing either the image encoder or text encoder with pre-trained weights.
- In CV, zero-shot learning refers to the study of generalizing to unseen object categories in image classification (or in a broader sense, generalization to unseen datasets). 
	- In CLIP, we pre-train to predict if an image and text snippet are paired together in its dataset. We re-use this capability to perform zero-shot classification!
	- ==Inference==: For a dataset, we use the names of all classes in the dataset as the set of potential text pairings, and then predict the most probable (image, text) pair according to clip. So if our dataset has Dogs, Cats, and Birds, we compute 3 text embeddings for "A picture of a Dog", "A picture of a Cat", "A picture of a Bird," and then determine which text embedding is closest to the image embedding of a picture that we want to classify. We can reuse these three text embeddings for every classification of subsequent images, since the class labels we're trying to assign won't change (unless we want them to, then we just recompute text embeddings for those classes). In this sense, CLIP is a great zero-shot classifier, in that we can just choose whatever classes we want and (attempt to) classify any dataset of images.
- Prompt Engineering and Ensembling
	- Many standard image classification datasets have noisy labels on the images that are chosen somewhat haphazardly. 
	- A common issue is ==**polysemy**== -- when the *name of the class* is the only information provided to CLIP's text encoder, it's unable to differentiate which word sense is meant, due to the lack of context. "**==Crane==**" as a label: it's unclear whether you mean 🏗️ or 🐦 (polysemy).
	- It's uncommon that dataset captions are a single word/class label; usually the text is a full sentence describing the image someway. 
		- This is why we use the prompt template `A photo of a {label}` as a good default that improves performance mildly... but you can improve performance by customizing the prompt to each task, like `A photo of a {label}, a type of pet` or `A satellite photo of a {label}`.
	- They also experimented with ==ensembling== over multiple zero-shot classifiers another way of improving performance, where we'd use different context prompts, such as:
		- `A photo of a big {label}`
		- `A photo of a small {label}`
	- We then ***construct the ensemble over the embedding space instead of probability space*** (meaning take the average representation of each text embedding), allowing us to cache only a single set of averaged text embeddings.
	- On ImageNet, they ensembles 80 different context prompts, improving performance by an additional 3.5% over the single default prompt.
- To assess the quality of the representations learned by models, it's common to fit a linear classifier on the representations extracted from the model and measure its performance on various datasets. TLDR is that the CLIP models produce very good features.
- There are many pages dedicated to the robustness of CLIP models, which is higher than many supervised alternatives; but it still generalizes poorly to data that's truly out-of-distribution (eg it learns good OCR representations for digitally rendered text, but achieves quite poor performance on MNIST hand-written digits)... so it's still subject to the brittleness of deep learning models.
- Limitations
	- Image-text pairs that CLIP is trained on are unfiltered and uncurated, resulting in CLIP models learning many social biases.
	- Captioning is difficult for complex tasks and visual concepts, even given the abilities of natural language.

Abstract:
> State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. ==Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision.== We demonstrate that the ==simple pre-training task of predicting which caption goes with which image== is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of ==400 million (image, text) pairs== collected from the internet. After pre-training, ==natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks==. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is ==often competitive with a fully supervised baseline without the need for any dataset specific training==. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained model weights at [this https URL](https://github.com/OpenAI/CLIP).

# Paper Figures
![[Pasted image 20240505125232.png]]
Above: Jointly training an image/text encoder to produce the ~same representations in vector space of an image and its caption. Then, at inference time, for each of our N classes, we produce a text vector of "A photo of a {Dog}," "A photo of a {Cat}," etc. and we see which text-embedding (representing a class) our embedded image vector is closest to.

![[Pasted image 20240506105926.png|200]]

![[Pasted image 20240506112132.png|250]]
Above: CLIP Pseudocode. Given a batch of images and a batch of texts ("aligned", meaning that the first caption in the text batch corresponds to the first image in the image batch), embed each batch using the corresponding encoders. Create the N x N pairwise cosine similarity matrix, and then compute the symmetric cross entropy loss function.

![[Pasted image 20240506114213.png|250]]

![[Pasted image 20240506122932.png]]

![[Pasted image 20240506123415.png]]
Above: On Robustness of CLIP representations; "Intuitively, a zero-shot model should not be able to exploit spurious correlations or patterns that hold only on a specific distribution, since it is not trained on that distribution. Thus it's reasonable to expect zero-shot models to have much higher effective robustness." But other details like CLIP's large and diverse pre-training dataset and use of natural language supervision could also result in more robust models, regardless of whether they're zero-shot or finetuned...

![[Pasted image 20240506123830.png]]
Above: More on robustness




# Non-Paper Figures
![[Pasted image 20241112125844.png]]
[Link](https://youtu.be/zdejKiH06CU?si=n-pcOT9yzwSs9Y7y&t=577): We want to show some failure cases. The [[SugarCrepe]] benchmark is based on CoCo, has images with candidate captions. The main difference between the captions is that the object attribute changes. "One apple and several oranges" -> "several apples and orange"  (can the model count?), "A blue vase" -> "An orange vase" (Can the model see color?)