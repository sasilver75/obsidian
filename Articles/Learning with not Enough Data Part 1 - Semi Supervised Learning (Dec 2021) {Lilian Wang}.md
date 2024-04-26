#article 
Link: https://lilianweng.github.io/posts/2021-12-05-semi-supervised/

------

When facing a *limited amount of labeled data for supervised learning tasks*, four approaches are commonly discussed.

1. Pre-training and fine-tuning
	- Use a pre-trained model, then finetune it on the downstream task with a small set of labeled samples.
2. [[Semi-Supervised Learning]]
	- ==Learn from the labelled and unlabeled samples together==!
3. [[Active Learning]]
	- Labeling is expensive, but we still want to collect more, given a cost budget. ==Can we learn to select the *most valuable* unlabeled samples to be labeled next, so that we can act smartly with our limited budget==?
4. Pre-training + dataset auto-generation
	- Given a capable pre=trained model, we can utilize it to auto-generate many more labeled samples.
		- ((Assumedly this is then used to finetune some model))


This part one blog post is on Semi-Supervised learning 🎉

# What is semi-supervised learning?
- ==Semi-supervised learning is about using *both* labeled *and* unlabeled data to train a model.==
- Interestingly, most existing literature on semi-supervised learning focuses on *vision tasks* 👁️📈

All the methods introduced in this post have a loss combining two parts:
$L = L_s + \mu(t)L_u$ 
- The ==supervised loss== $L_s$ is easy to get, given all the labeled examples.
- We will focus on how the ==unsupervised loss== $L_u$ is designed.
- A common choice of the ==weighting term== $\mu(t)$ is a ramp function that increases the importance of $L_u$ in time, where $t$ is the training step.

![[Pasted image 20240409193035.png]]

# Hypotheses
Several hypotheses have been discussed in literature to support certain design decisions in semi-supervised learning methods.

1. H1: Smoothness Assumptions
	- ==If two data samples are close in a high-density region of feature space, their labels should be the same or very similar.==
2. H2: Cluster Assumptions
	- The feature space has both dense regions and sparse regions. Densely grouped data points naturally form a cluster.
	- ==Samples in the same cluster are expected to have the same label (this is very similar to H1)==
3. H3: Low-density Separation Assumptions
	- ==The decision boundary between classes tends to be located in these sparse, low-density regions.== 
		- If this weren't true, high-density clusters would be cut into two classes -- this would invalidate H1/H2.
4. H4: Manifold Assumptions
	- ==The high-dimensional data tends to locate on a low-dimensional manifold.==
	- Even though real-world data might be observed in very high dimensions, they can actually be captured by a lower-dimensional manifold where certain attributes are captured, and similar points are grouped closely.
	- This enables us to learn a more efficient representation for us to discover and measure similarity between unlabeled data points. This is the foundation for representation learning.


---
An aside on the Manifold Hypothesis
- Imagine that you have a bunch of seeds fastened on a glass plane, resting horizontally on a table; we could say that these seeds live in a two-dimensional space, more or less.
- Now imagine that you take the plate and tilt it diagonally upwards, so that the surface of the glass isn't horizontal with respect to the ground.If you wanted to locate one of the seeds, you have two options:
	1. If you decide to ignore the glass, then each seed would appear to be floating in the three-dimensional space above the table
	2. But no matter how you rotate the plate in three dimensions, the seeds still live along the surface of a two-dimensional plane
---


# Consistency Regularization
- [[Consistency Regularization]], also known as Consistency Training, assumes that randomness within the neural network or data augmentation transforms should not modify model predictions, given the same input.
	- ((Alternative explanation: The core idea is to enforce consistency in the model's predictions for the same input, or for slightly perturbed versions of the input. The assumption behind this is that the model's predictions should be consistent or invariant to certain transformations/perturbations applied to the input data. These can be introduced through random noise, dropout, or data augmentation techniques like rotation/flipping/scaling. By minimizing the consistency regularization loss during training, the model is encouraged to learn representations that are invariant to the perturbations introduced, leading to improved robustness and generalization.))
- This idea has been adopted in several self-supervised learning methods:
	- [[SimCLR]]
	- [[BYOL]]
	- SimCSE

Different augmented versions of the same sample should result in the same representatoin
- ((A dog pic, a dog picture with noise, a dog picture rotated 90 degrees, etc. Different augmented versions of the same sample should result in the same representation.))

### Pi-Model
- Sajjadi et al. (2016) proposed an unsupervised learning loss to minimize the difference between two passes through the network with stochastic transformations (e.g. dropout, random max-pooling) for the same data point.
	- Laine and Aila (2017) later coined the name ==$\pi$-Model== for such a setup.

![[Pasted image 20240409195400.png]]
Above: $f'$ is the same NN with different stochastic augmentations or dropout masks applied.


### Temporal Ensembling
- The $\pi$-model requests the network to run two passes per sample, doubling the computation cost!
- To reduce this, ==Temporal Ensembling== (2017) maintains an exponential moving average (EMA) of the model prediction in time per training sample $\tilde{z}_i$ as the learning target, which is *only evaluated and updated once per epoch*.
![[Pasted image 20240409195934.png]]


### Mean teachers
- Temporal Ensembling keeps track of an EMA of label predictions for each training sample as a learning target.
	- However this label prediction only changes every *epoch*, making the approach clumsy when the dataset is large.
- Mean Teacher (2017) is proposed to overcome the slowness of target update by tracking the moving average of model weights instead of model outputs. 
![[Pasted image 20240409200126.png]]

The consistency regularization loss is the distance between predictions by the student and teacher, and the student-teacher gap should be minimized.
- The mean teacher is expected to provide more accurate predictions than the student!


### Noisy Samples as learning targets
- Several recent consistency training methods learn to minimize prediction difference between the original unlabeled sample and its corresponding augmented version -- it's quite similar to the $\pi$-model, but the consistency regularization loss is only applied to the unlabeled data.
- Adversarial Training (Goodfellow, 2014) applies adversarial noise onto the input and trains the model to be robust to such an adversarial attack.
- Virtual Adversarial Training (VAT, 2018) extends this idea to work in semi-supervised learning.
- Interpolation Consistency Training (ICT, 2019) enhances the dataset by adding more interpolations of data points, expecting the model predictions to be consistent with interpolations of corresponding labels.
- [[Mixup]] (2018) operation mixes two images by a simple weighted sum and combines it with [[Label Smoothing]].

Similar to VAT, ==Unsupervised Data Augmentation (UDA; 2020)== learns to predict the same output for an unlabeled example and an augmented one. UDA especially focuses on studying how the "quality" of noise can impact the semi-supervised learning performance with consistency training.
- Good augmentations should produce *VALID* and *DIVERSE* noise, and carry targeted inductive biases

For images, UDA adopts RandAugment, which uniformly samples augmentation operations available in PIL -- no learning or optimization, so it's much cheaper than AutoAugment.

For language, UDA combines [[Back-Translation]] and [[TF-IDF]]-based word replacement.
- Back-translation preserves the high level meaning, but may not retain certain words
- TF-IDF based word replacement drops uninformative words with low TF-IDF scores.

UDA was found to be complementary to transfer learning and representation learning.

UDA found that two techniques helped to improve results:
1. Low confidence masking
	- Mask out examples with low prediction confidence, if lower than some threshold $\tau$   
2. Sharpening prediction distribution
	- Use a low temperature $T$ in softmax to sharpen the predicted probability distribution
3. In-domain data filtration
	- In order to extract more in-domain data from a large out-of-domain dataset, train a classifier to predict "in-domain" labels, and then retain samples with high confidence predictions as in-domain candidates.



# Pseudo Labeling
- Pseudo Labeling (2013) assigns fake labels to unlabeled samples based on the maximum softmax probabilities predicted by the current model, and then trains the model on both labeled and unlabeled samples simultaneously, in a purely supervised setup.

# Label propagation
- Label Propagation (2019) is an idea to construct a similarity graph among samples based on feature embedding -- then the pseudo labels are "diffused" from known samples to unlabeled ones where the propagation weights are proportional to pairwise similarity scores in the graph -- similar to a kNN classifier.

# Self-Training
- Not a new concept (1965, 2000); an iterative algorithm, alternating between the following two steps until every unlabeled sample has a label assigned:
	1. Initially, builds a classifier on the labeled data
	2. Uses this classifier to predict labels for the unlabeled data, and converts the most confident ones into labeled samples

Xie et all (2020) applied self-training in deep learning and achieved great results, in a method they called **Noisy Student**.
- Noise is important for the student (who has noise applied) to perform better than the teacher (who has no noise applied).


# Reducing confirmation bias
- Confirmation bias is a problem with incorrect pseudo-labels provided by an imperfect teacher model; overfitting to wrong labels may not give us a better student model!
- Arazo et al (2019) proposed two techniques to reduce confirmation bias:
	1. Adopt MixUp with soft labels
	2. Further set a minimum number of labeled samples in each minibatch by oversampling the labeled samples.


# Pseudo Labeling with Consistency Regularization

It's possible to combine the above two approaches together, running semi-supervised learning with *both* pseudo labeling and consistency training!

## MixMatch (2019)
- A holistic approach to semi-supervised learning, utilizing unlabeled data by merging the following techniques:
	1. Consistency regularization: Encouraging the model to output the same predictions on perturbed unlabeled samples
	2. Entropy minimization: Encourage the model to output *confident predictions* on unlabeled data
	3. MixUp augmentation: Encourage the model to have *linear behavior between samples*

It's critical to have MixUp especially on the unlabeled data. Removing temperature sharpening on the pseudo label distribution hurts the performance quite a lot.

## ReMixMatch (2020)
- Improves MixMatch by introducing two new mechanisms:
	1. Distribution alignment
		- It encourages the marginal distribution p(y) to be close to the marginal distribution of the ground truth labels.
	2. Augmentation anchoring
		- Given an unlabeled sample, it first generates an "anchor" version with weak augmentation, and then averages $K$ strongly-augmented versions using CTAugment (Control Theory Augment). CTAugment only samples augmentations that keep the model predictions within the network tolerance.


## DivideMix (2020)
- Combines semi-supervised learning with Learning with Noisy Labels (LNL).
- Models the per-sample loss distribution via a GMM to dynamically divide the training data into a labeled set with clean examples, and an unlabeled set with noisy ones.
	- Clean samples are expected to get lower loss faster than noisy samples

... Skipping ...

## FixMatch (2020)
- Generates pseudo labels on unlabeled samples with weak augmentation and only keeps predictions with high confidence.


# Combined with Powerful Pre-Training
- It's a common paradigm to first pre-train a task-agnostic model on a large unsupervised data corpus via self-supervised learning and then fine-tune it on the downstream task with a small labeled dataset.
	- Research has shown that we can obtain extra gain if combining semi-supervised learning with pretraining.


...Skipping...

![[Pasted image 20240409213752.png]]


























