November 10, 2016 -- [[Samy Bengio]] and others
Link: [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530)

Created Zotero paper using Zotero extension

There's not a lot interesting in this paper to me in 2024. They basically just say that they can fit a NN to even a randomly-labeled set of CIFAR-10 data (meaning the models are very flexible), though they can't generalize very well to correctly-labeled data when you do this, which is obvious. Increasing the corruption rate (percent of labels that are randomized) of the labels means that it takes longer to train the network, and that the generalization error increases, but we always can get 0 training error. Thy note that the usual tools that they use to understand predictive models don't seem to be fit to address the NN era, which is true.

Abstract
> Despite their massive size, successful deep artificial neural networks can exhibit a remarkably small difference between training and test performance. Conventional wisdom attributes small generalization error either to properties of the model family, or to the regularization techniques used during training.  
> Through extensive systematic experiments, we show how these ==traditional approaches fail to explain why large neural networks generalize well in practice==. Specifically, our ==experiments== establish that state-of-the-art convolutional networks for image classification trained with stochastic gradient methods ==easily fit a random labeling of the training data==. This phenomenon is qualitatively unaffected by explicit regularization, and occurs even if we replace the true images by completely unstructured random noise. We corroborate these experimental findings with a theoretical construction ==showing that simple depth two neural networks already have perfect finite sample expressivity as soon as the number of parameters exceeds the number of data points as it usually does in practice==.  
> We interpret our experimental findings by comparison with traditional models.
