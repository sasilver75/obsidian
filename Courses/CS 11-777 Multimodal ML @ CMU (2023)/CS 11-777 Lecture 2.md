https://youtu.be/BO-N8tb3kmw?si=LfhHMT1Kk6dG5N9n
# Topic: Unimodal Representations

---

Objectives
- Unimodal basic representations
- Dimension of heterogeneity
- Image representations
	- Image gradients, edges, kernels
- Convolutional Neural Networks (CNNs)
	- Convolution and pooling layers
- Visualizing CNNs
- Region-based CNNs

---


![[Pasted image 20240612185418.png]]
How do we represent these three channels of color?
- Usually via 3 matrices, with each matrix having one channel (r, g, b)

![[Pasted image 20240612185935.png]]
In language, we can do a one-hot encoding where each position in our vector is a word. This bag-of-words vector doesn't preserve ordering or context of words, and there are a lot of problems related to it (scare, scary are different positions).-
- These days, representation isn't at the word level, but at the subword level.
This type of representation hasn't been popular since before [[Word2Vec]]/[[ELMo]], but for a while (late 00's), we lost a lot of the sequence structure but still had decent performance.



![[Pasted image 20240612191143.png]]
In speech, the simplest representation is waveforms.
- This acoustic waveform could be represented as a 1-dimensional vector (at some specific sampling rate, bit depth) where the value at each position represents the amplitude of the signal.
- In speech, although you could use the very raw representation of the wave signal, here we usually do some sort of pre-step before representing.
	- We usually create ==audio frames==, which are very local windows that we can slide along our waveform. The time window size and offset are parameters here.
		- Offset: by how much do I slide the window?
		- Window size: the width of the window
This spectrogram kind of looks like an imagine in a way. You can run a CNN on it, or again have a way of casting it into a vector to process (perhaps by time windows).
- Re: processing it as if it were an image. The main difference is that locality has different meanings, and that moving left to right on the image means going forward in time. 
From this, we can predict emotions, spoken words, voice quality, etc.

![[Pasted image 20240612191836.png]]
Sensors in robotics are interesting too; they do look like speech signals in that they're temporal, but they have different dynamics.

![[Pasted image 20240612191953.png]]
There's some work on taking tables and trying to summarize the content. There's a lot of ad-hoc ways that people are trying to use to represent tables.

![[Pasted image 20240612192039.png]]
There's a lot of work on graph representation too. The basic building blocks are nodes, edges (and optionally weights). From there, you can do a lot of things ("Is this group of friends from a certain culture? Do they like a certain thing?"), or you can use this graph as extra knowledge for input to another approach.

![[Pasted image 20240612192125.png]]
Sets, collections!


## Dimensions of Heterogeneity

![[Pasted image 20240612192146.png]]
What really makes multimodality interesting is the heterogeneity (but relatedness) of modalities!

![[Pasted image 20240612192226.png]]
The first one is the one to think deeply about; Given one or two modalities, what is the locally-cohesive-enough element that has enough information but ALSO doesn't have too much muddled information!
- Language is easy (words, subwords), but if you look at a different modality, think about what building block would be useful to both analyze that modality and *link* that modality to other modalities!


## Image Representations

![[Pasted image 20240612192434.png]]
The simplest way would be to take the image and just represent it as a vector, but in reality, most of the popular representations look at images as lists of objects, possibly with labels/descriptions/metadata:

![[Pasted image 20240612192653.png]]
- Each of these descriptors will be part of a feature vector describing the qualities of the object in the bounding box.

80% of the papers produced these days... we're going to only keep the 2D representation, but a lot of the depth information will be implemented in... the feature vectors (?).

![[Pasted image 20240612193239.png]]
Think: How is the human eye perceiving things? It looks like a Gabor filter; mostly looking for edges.

![[Pasted image 20240612193614.png]]

![[Pasted image 20240612193815.png]]
![[Pasted image 20240612193947.png]]
We learn these kernels at every level of these hierarchical representations that we build throughout our network. Has nice properties like translation invariance.


![[Pasted image 20240612194244.png]]
![[Pasted image 20240612194249.png]]
We use the same weights as we slide the kernel over the entire image, giving us translation invariance.

We can stack many of these kernels to get (eg) 50 response maps in a single forward pass, cool!
- Maybe one looks for eyes, and another looks for the corner of the mouth.

To make these features more high-level we can use pooling strategies like [[Max Pooling]].
![[Pasted image 20240612194943.png]]
The CNN does template matching. The response map is really about how good of a match we found, using our kernel. What's really important is "how good of a match." The Max operation here makes sense, because it says "This is the best match I found locally!"

![[Pasted image 20240612195209.png]]

![[Pasted image 20240612195216.png]]
[[Residual Network]] helps with deeper networks, blah blah.

## Region-Based CNNs
- How do we go from an image to all of those bounding boxes?
- ![[Pasted image 20240612195242.png]]

Ways:
1. A sliding window; we have a dataset of very well-segmented objects, and train a NN. Given an image, we try every possible size of bounding box around, and simply go around and try to find a dog, person, etc.
2. A better option: Start by identifying hundreds of region proposals, and then apply our CNN object detector:

![[Pasted image 20240612195337.png]]

One of the approaches for image segmentation in using ==Superpixels==; the idea is that we take our image and look for coherent patches, where the color/texture is coherent. We then slowly make them more and more coherent (merging?).

![[Pasted image 20240612195417.png]]
The general idea is to look at local coherence.
- As we go along, we start having fewer and fewer of these image patches. We don't want to go too far (because then smaller objects won't be there)... 
- Once we have (eg) 100 possible candidates, we then run our CNN on those regions.

[[Region-Based Convolutional Neural Network|R-CNN]] (Region-Based CNN)
- If I have two overlapping objects, the patch between two objects... the few layers of CNN could be re-used... because they're overlapping... so instead of running the CNN on each image patch, we run it once on the whole image, cache all this information, and then, for each image patch, we look at these local memory buffers that we recorded.
	- It's just an efficient way
- ![[Pasted image 20240612195618.png]]
- We run the CNN once on the whole image, and then segment on that, instead of doing bottom-up in the manner described earlier.

With CNNs, we start having representations that are meaningful, especially in the last few layers!
![[Pasted image 20240612195657.png]]
The simplest way is to take the last layer and do something like [[Principal Component Analysis|PCA]] or [[t-SNE]]. 
- Above: Doing it on MNIST digits, we see that there are 10 nice digit clusters.

----

# Lecture 2.2
Link: https://youtu.be/CfSX2dFinUI?si=Kgic6Q-WZoR78hNC

---









