April 8, 2009 -- [[Alex Krizhevsky]]
Paper: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
Blog: [CIFAR-10](https://paperswithcode.com/dataset/cifar-10)

The **CIFAR-10** dataset (Canadian Institute for Advanced Research, 10 classes) is a subset of the *Tiny Images* dataset and consists of ==60,000 32x32 color images==. The images are ==labelled with one of 10 mutually exclusive classes==: airplane, automobile (but not truck or pickup truck), bird, cat, deer, dog, frog, horse, ship, and truck (but not pickup truck). ==There are 6000 images per class with 5000 training and 1000 testing images per class==.

Abstract
> ==Groups at MIT and NYU have collected a dataset of millions of tiny colour images from the web==. It
> is, in principle, an excellent dataset for unsupervised training of deep generative models, but previous
> researchers who have tried this have found it dicult to learn a good set of lters from the images.
> We show how to train a multi-layer generative model that learns to extract meaningful features which
> resemble those found in the human visual cortex. Using a novel parallelization algorithm to distribute
> the work among multiple machines connected on a network, we show how training such a model can be
> done in reasonable time.
> A second problematic aspect of the tiny images dataset is that there are no reliable class labels
> which makes it hard to use for object recognition experiments. ==We created two sets of reliable labels==.
> The ==CIFAR-10== set has ==6000 examples of each of 10 classes== and the CIFAR-100 set has 600 examples of
> each of 100 non-overlapping classes. Using these labels, we show that object recognition is signicantly
> improved by pre-training a layer of features on a large set of unlabeled tiny images.

