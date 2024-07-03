Link: https://lilianweng.github.io/posts/2021-12-05-semi-supervised/

Note that this article was written in 2021

---

When faced with a limited amount  of labeled data for supervised learning, four strategies are commonly discussed:
1. Pre-training + Fine-Tuning: Pretrain a task-agnostic model on a large, unsupervised training corpus, and then fine-tune it on the downstream task with a smaller set of labeled samples.
2. [[Semi-Supervised Learning]]: Learn from labelled and unlabeled samples together. A lot of research has happened on vision tasks within this approach.
3. [[Active Learning]]: Labeling is expensive, but we still want to collect more, given a cost budget -- can we learn to select the most valuable *unlabeled samples* to be collected next, so that we can act smartly with a limited budget?
4. Pre-training + dataset auto-generation: Given a capable pre-trained model, we can utilize it to auto-generate a lot more labeled samples; this has been especially popular within the language domain, utilizing the success of few-shot learning (([[Synthetic Data]])).


## What is Semi-Supervised Learning?
- [[Semi-Supervised Learning]] uses both labeled and unlabeled data to train a model.
- ==All methods introduced in this post have a loss combining two parts:== 
$\mathcal{L} = \mathcal{L}_s + \mu(t)\mathcal{L}_u$. 
- This combines a supervised loss $\mathcal{L}_s$ with an unsupervised loss $\mathcal{L}_u$. A common choice of weighting term $\mu(t)$ is a *ramp function* that increases the importance of $\mathcal{L}_u$ in time, where $t$ is the training step.

## Hypotheses
Several hypotheses have been discussed in literature to support certain design decisions in semi-supervised learning methods.
1. H1:  ==Smoothness Assumptions==: If two data samples are close in a *high-density region* of feature space, their labels should be the same or very similar.
2. H2: ==Cluster Assumptions==: The feature space has both *dense regions* and *sparse regions*. *Densely-grouped data points naturally form clusters, and clusters are expected to have the same label*. This is a small extension of H1 above.
3. H3: ==Low-density Separation Assumptions==: The decision boundary between classes tends to be located in sparse, low-density regions, because otherwise the decision boundary would cut a high-density cluster into two classes, corresponding to two clusters, which invalidates H1 and H2.
4. ==[Manifold Assumptions](https://stats.stackexchange.com/questions/66939/what-is-the-manifold-assumption-in-semi-supervised-learning)==: The high-dimensional data tends to locate on a low-dimensional manifold. Even though real-world data might be observed in very high dimensions (such as images of real-world objects/scenes), they can actually be captured by a lower-dimensional manifold (images of real-world objects/scenes are not drawn from a uniform distribution over all pixel combinations). This enables us to learn more efficient representations for us to discover and measure similarity between unlabeled data points. 

## Consistency Regularization
- [[Consistency Regularization]] (aka Consistency Training) assumes that the randomness within the NN (eg with Dropout) or data augmentation transforms *should not* modify model predictions, given the same input.
	- In otherwise, if $d$ is labeled as an $A$, then any augmented $\bar{d}$ should still be labeled as an $A$. ==Different augmentations of the same sample should result in sam representation/label.==
- This idea has been adopted in several methods, like [[SimCLR]], [[BYOL]], SimCSE, etc.

## Pi Model

![[Pasted image 20240527145912.png]]
The $\Pi$ model (Sajjadi et al, 2016 + Laine at Aila, 2017) proposed an unsupervised loss to minimize the difference between two passes through the network with stochastic transformations.

## Temporal Ensembling

![[Pasted image 20240527150153.png]]
But this requires two passes through the network per sample, doubling computation costs. To reduce costs, ***Temporal Ensembling*** (Laine and Aila, 2017) maintains an exponential moving average (EMA) of the model prediction in time per training sample $\tilde{z}_i$  as the learning target, which is evaluated and updated once per epoch.

## Mean Teachers
![[Pasted image 20240527150231.png]]
Because Temporal Ensembling's EMA of label predictions for each training samples is only updated every epoch, the approach is clumsy when the dataset is large. 
- Mean Teacher (2017) proposed to overcome the slowness of target updates by tracking the moving average of model *weights*, instead of model *outputs!*
- If the original model with weights $\theta$ is the *student model*, then the model with moving averaged weights $\theta'$ is the *mean teacher*, where $\theta' = \beta\theta' + (1-\beta)\theta$ 
- The consistency regularization loss is the distance between predictions of the student and teacher, and the student-teacher gap should be minimized.

## Noisy samples as learning targets
- Several recent consistency models learn to minimize prediction difference between original unlabeled samples and corresponding augmented version; similar to the $\Pi$ Model, but consistency regularization is *only* applied to the unlabeled data.
- ![[Pasted image 20240527150550.png]]
- Adversarial Training (Goodfellow et al. 2014) applies adversarial noise into the input and trains the model to be robust to such adversarial attacks.
- Virtual Adversarial Training (VAT; Miyata et al. 2018) extends the idea to work in semi-supervised learning.
- Interpolation Consistency Training (ICT; Veram et al. 2019) enhances the dataset by adding more interpolations of data points, and expects the model predictions to be consistent with interpolations of corresponding labels.
- [[MixUp]] operation mixes two images via a simple weighted sum, and combines it with label smoothing. 
![[Pasted image 20240527150759.png]]
Because the probability of two randomly-selected unlabeled samples belong to different classes is high, the interpolation by applying a mixup between two random unlabeled samples is likely to happen around the decision boundary. According to the low-density separation assumptions, the decision boundary tends to locate in the low-density regions.

- Unsupervised Data Augmentation (UDA; Xie et al. 2020) learns to predict the same output for an unlabeled example and the augmented one. Focuses on studying how the "quality" of noise can impact the semi-supervised learning performance with consistency training. It's crucial to use advanced data augmentation methods for producing meaningful and effective noisy samples; ==good data augmentation should produce valid (does not change the label) and diverse noise, and carry targeted inductive biases.==
	- For images, adopts RandAugment, which uniformly samples augmentation operations available in PIL.
	- For language, combines back-translation and TF-IDF-based word replacement.

## Pseudo Labeling
- Pseudo Labeling (Lee 2013) assigns fake labels to unlabeled samples based on maximum softmax probabilities predicted by the current model, and trains the model on both labeled and unlabeled samples simultaneously in a pure unsupervised step.
	- Why does this work? It's equivalent to *Entropy Regularization,* which minimizes the conditional entropy of class probabilities for unlabeled data to favor low-density separation between classes.
	- Training with pseudo labeling naturally comes as an iterative process; we refer to the model that produces pseudo labels as teacher, and the model that learns from pseudo labels as student.

## Label Propagation
- Label propagation (Iscen et al. 2019) is an idea to construct a similarity graph among samples based on feature embedding. Pseudo labels are "diffused" from known samples to unlabeled ones, where propagation weights are proportional to pairwise similarity scores in the graph.
![[Pasted image 20240527151551.png]]

## Self-Training
- An iterative algorithm alternating between the following two steps, until every unlabeled sample has a label assigned:
	- Initially builds a classifier on labeled data
	- Uses this classifiers to predict labels for the unlabeled data, and converts the *most confident* ones into labeled samples.
- Xie et al (2020) applied self-training in deep learning on the ImageNet classification task, using an EfficientNet model as a teacher to generate pseudo-labels for 300M unlabeled images, and then trained a larger EfficientNet as student to learn from both true labeled and pseudo-labeled images.
	- A critical element is to have *noise* during student model training, but have no noise for the teacher to produce pseudo labels. This is called ==Noisy Student==; the used stochastic depth dropout and RandAugment to noise the student.

## Reducing Confirmation Bias
- Confirmation bias is a problem with incorrect pseudo-labels provided by an imperfect teacher model; overfitting to wrong labels won't give us a better student model.
- Arazo et al. (2019) combats this by adopting Mixup with soft labels.
- Pham et al. (2021) adapts the teacher model constantly with the feedback of how well the student performs on the labeled dataset. The teacher and the student are trained in parallel, where the teacher learns to generate better pseudo labels, and the student learns from the pseudo labels.

# Pseudo-labeling with Consistency Regularization
- It's possible to use both approaches!

## MixMatch
- [[MixMatch]] (Berthelot 2019) is a holistic approach to semi-supervised learning, using unlabeled data by merging the following techniques:
	1. ==Consistency Regularization==: Encourages the model to output the same predictions on perturbed, unlabeled samples.
	2. ==Entropy Minimization==: Encourages the model to output confident predictions on unlabeled data.
	3. ==[[MixUp]] Augmentation==: Encourages the model to have linear behavior between samples.
	- Given a batch of labeled data X and unlabeled data U, we created augmented versions of them via MixMatch $\bar{X}$ and $\bar{U}$ , containing augmented samples and guessed labeled for unlabeled examples.
![[Pasted image 20240527152537.png]]

ReMixMatch (Berthelot et al. 2020) improves MixMatch by adding two new mechanisms
![[Pasted image 20240527152614.png]]
1. ==Distribution Alignment==
	- Encourages the marginal distribution $p(y)$ to be close to the marginal distribution of ground-truth labels.
2. ==Augmentation Anchoring==
	- Given an unlabeled sample, first generates an "anchor" version with weak augmentation, and then averages K *strongly* augmented versions using CTAugment.

DivideMix (Junnan Li et al. 2020) combines semi-supervised learning with Learning with noisy labels (LNL).
- This models the per-sample loss distribution via a GMM to dynamically divide the training data into a labeled set with clean examples, and an unlabeled set with noisy ones.
![[Pasted image 20240527153405.png]]

FixMatch (Sohn et al. 2020) generates pseudo labels on unlabeled samples with weak augmentation, and only keeps predictions with high confidence. Here, both weak augmentations and high confidence filtering helps produce high-quality trustworthy pseudo label targets. Then FixMatch learns to predict these pseudo labels given a heavily-augmented sample.
- Here, weak augmentations are standard flip-and-shift augmentations, and Strong augmentations are AutoAugment, Cutout, RandAugment, CTAugement.

![[Pasted image 20240527153635.png]]

## Combined with Powerful Pre-Training
- Research has shown that we can obtain extra gain if combining semi-supervised learning with pretraining.
- Authors on multiple papers found:
	- Effectiveness on pre-training diminishes with more labeled samples available for downstream tasks; pre-training is helpful in low-data regimes (20%) but neutral or harmful in the high-data regime.
	- Self-training helps in high data/strong augmentation regimes, even when pre-training hurts.
	- Self-training can bring in additive improvement on top of pre-training, even using the same data source.
	- Self-supervised pre-training (e.g. via SimCLR) hurts performance in high data regime, similar to how unsupervised pre-training does.
	- Joint-training supervised and unsupervised objectives helps resolve mismatch between pre-training and downstream task.
	- Noisy labels or un-targeted labeling is worse than targeted pseudo labeling.
	- Self-training is computationally more expensive than fine-tuning on a pre-trained model.



