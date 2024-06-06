Link: https://www.youtube.com/watch?v=ijqUUZI3osM&list=PLWnsVgP6CzafDszSy-njjdqnliv5qT0EW&index=19

# Topic: Vision-Language Models

(This was after their midterm exams; Lecture 18 was a midterm review)

---

We're interested in models that can either/both PROCESS images+text, or GENERATE images+text!

A SoTA system is [[GPT-4V]] (We play around with it in the class, asking it about pictures.)

![[Pasted image 20240605172047.png|300]]
![[Pasted image 20240605172055.png|300]]

We've seen how to compute representation of words, but what about images?

How do we even represent images?
![[Pasted image 20240605172144.png|300]]
In a grayscale image, we might represent each pixel in the image with an integer from 0-255. These are pixel values!
![[Pasted image 20240605172422.png|300]]
If we have color images (or other spectrums), we just add additional channels in a third dimension!

One of the critical components of computer vision inclute the [[Convolution]] operator:
![[Pasted image 20240605172526.png]]
We generally define a smaller matrix (called a kernel or filter), with generally learned values. We slide this (often smaller) kernel over every single pixel in the (larger) input image, performing an element-wise multiplication for every "covered" position. We'll sum these (eg, 9) products, and the sum will give the output of the resulting cell, in that location.
- We can use this formula of applying convolutions and shifting the formula over the image to influence the dimensionality of the output.

If we wanted a SMALLER output image, we can slide the filter faster, going over TWO (or more) spaces every time we slide the filter. We could also do Pooling, where we might, for each sub-region, output the maximum value of the subregion of the feature map covered by the filter.

We aim to to learn the weights on each of the filter "pixels" to accomplish our objective.

We might pass our image through multiple convolutional kernels:
![[Pasted image 20240605173136.png]]
If we wanted to get SMALLER outputs, we can increase the Stride hyperparameter.

If we set the stride to two, we see the image size is halved:
![[Pasted image 20240605173218.png]]
We can use these properties to get bottleneck representations where we have the model try to learn useful features.

The paper that started the deep learning revolution was the [[AlexNet]] paper from 2012. It showed that using CNNs combined with GPUs could beat the shit out of everyone else in the ImageNet LSVRC

![[Pasted image 20240605174529.png]]

![[Pasted image 20240605174537.png]]
Here's Alexnet's simple architecture! We'll compare it today's architectures, which are much deeper and (often) using Transformers.
- See that we apply a number of convolutions, turn it into a vector, apply some fully connected layers, and then add a classification head.

![[Pasted image 20240605174811.png]]
Above: [[Residual Network|ResNet]]s allowed for meaningful scaling of the depths of these networks!


Nowadays, we've shifted over to the Transformer to encode images.

![[Pasted image 20240605174839.png]]
One easy thing you could do is just linearize/vectorize an image and feed it into a transformer as usual, and use the resulting encoding to predict the class of the image!

![[Pasted image 20240605175430.png|300]]
->
![[Pasted image 20240605175443.png|300]]
Vision Transformers: *An Image is Worth 16x16 words* in 2021 said that we should split our image into N 16x16 patches. Each of these patches can be linearized, then passed through a linear layer to reduce the dimensionality of each of these patches.
- They also add positional embeddings; learned D-dimensional vector per position. They used Learned positional embeddings in the paper.

![[Pasted image 20240605175714.png]]

16x16 patches x 3 dimensions = 768-dimensional patch vectors.

![[Pasted image 20240605175746.png]]
With larger amounts of models, Transformers outperform ResNets with larger datasets (and we can fully parallelize everything at training time, and many of the tricks we've learned for Text can carry over to this setting as well!)

Since the architectures are now basically the same, can we train a single model on both modalities?
- Really it's just how the input is being embedded
	- Subword embeddings vs patch embeddings
- And what the objective function is

Now there have been many efforts to train *joint* models; single models that can operate on both images and text!

![[Pasted image 20240605180004.png]]
OpenAI's [[CLIP]] is one of the most popular family of models.
- Collected 400M image/text pairs from the web (using `alt` text)
Then, train an image encoder and a text encoder wit ha simple [[Contrastive Loss]]:
- Given a collection of images and text, predict which (image, text) pairs actually occurred in the dataset!
Uses a separate encoder for text and for images that both encode data into the same vector space.
![[Pasted image 20240605180108.png]]
Given an collection of (image, text) pairs, feed each through their corresponding image to get a text/image vectors.
- The goal is to make the dot product with your matching image/text caption as high as possible, and as low as possible with all other images/text captions in the batch.
	- ((So in the image above, T1 itself is a whole text encoding vector for the first text caption in the batch))

![[Pasted image 20240605180630.png]]
If you know what labels you're interested in (plane, car, dog, bird), you can generate encodings of all these words using the text encoder ("A picture of a plane"), then you can take your image you want to classify, encode it, and select the class whose text vector has the highest dot product with your image vector.

![[Pasted image 20240605180901.png]]
[[LAION-5B]] was a dataset from open-source researchers that aimed to create a useful dataset in this vein; Used CommonCrawl datasets, filtered webpages, downloaded the image-text pairs, and did content filtering (using CLIP, I believe!).
- There were a lot of copyright issues and CSAM

![[Pasted image 20240605181137.png]]
Looking at the Fuyu-8B model from [[Adept]] (Which I think... isn't that good)
- But it has a simple architecture to enable generation on top of image+text inputs.

![[Pasted image 20240605181211.png]]
- Given a picture of a persimmon, they split the image into patches, and then for each patch, they use a simple linear projection to get it into a vector representation.
	- They add a special token at the en of each "row" of the image (like an image newline character) that indicates that they're moving on to the next row. This is just to help the model.
- You get the image patches, encode them into a vector... once you get to the final image newline character, you start predicting the image caption in natural language.
	- In this way, you learn a model that takes images as input and can do question answering, captioning, etc.
	- Every part of the model is trained via the NTP loss on the language side.
- It's a very simple architecture, but this model does have vaguely competitive performance.


Note: Generally patch size is fixed to 16x16; high resolution images are usually downscaled to some fixed size and patchified.

There are problems with patches
- Important parts might be split across multiple patches


