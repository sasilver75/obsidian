Link: https://lilianweng.github.io/posts/2022-02-20-active-learning/

See part 1: [[Learning with not Enough Data  Part 1 - Semi-Supervised Learning (Dec 5, 2021) {Lillian Wang}]]

---

## What is Active Learning?
- Given an unlabeled dataset $U$ and a fixed amount of labeling cost $B$, [[Active Learning]] aims to select a subset of $B$ examples from $U$ to be labeled, such that they can result in maximized improvement in model performance.
	- This is an effective way of learning, especially when data labeling is difficult and costly (e.g. medical images).

![[Pasted image 20240527155414.png]]

## Acquisition Function
- The process of identifying the most valuable examples to label next is referred to as "sampling strategy" or "query strategy." 
	- ***The scoring function in the sampling process is named*** "==acquisition function==," denoted as $U(x)$. Data points with higher scores are expected to produce higher value for model training if they get labeled.

Here are some basic sampling strategies

### Uncertainty Sampling
- ==Selects examples for which the model produces most uncertain predictions==. Given a single model, uncertainty can be estimated by predicted probabilities. Another way is by Query-by-Committe (QBC) methods, which measure uncertainty based on a pool of opinions -- We can use Voter entropy, Consensus entropy, or KL divergence.
### Diversity Sampling
- ==Intends to find a collection of samples that can well-represent the entire data distribution==. Diversity is important because the model is expected to work well on any data in the wild, not just a narrow subset.
### Expected Model Change
- Refers to the impact that a sample brings into model training. The impact can be the influence on the model weights, or the improvement over the training loss.
### Hybrid Strategy
- Many methods above aren't mutually exclusive; hybrid sampling strategies value different attributes of data points, combining different sampling preferences into one.


## Deep Acquisition Function

- Model uncertainty is commonly categorized into two buckets:
	- *Aleatoric Uncertainty*: Introduced by noise in the data (sensor data, noise in measurement process) and can be input-dependent or input-independent. Generally considered irreducible.
	- *Epistemic Uncertainty*: Uncertainty within the model parameters; theoretically reducible given more data.

### Ensemble and Approximated Ensemble
- There is a long tradition in ML of using ensembles to improve model performance; diversity among models results in better results.
	- [[AdaBoost]] aggregates many weak learners to perform similar to even better than a single strong learner.
	- [[Bootstrap]]ping ensembles multiple trials of resampling to achieve more accurate estimation of metrics.
	- [[Random Forest]] and GBM are also good examples.
- To get better uncertainty estimation, it's intuitive to aggregate a collection of *independently trained* models.
- ==In active learning, a common approach is to use *dropout* to "simulate" a probabilistic Gaussian process. We thus ensemble multiple samples collected from the *same* model, but with different dropout masks applied during the forward pass.==
- Alternative cheaper options to naive ensembles:
	- Snapshot ensembles: Use a cyclic learning schedule to train an implicit ensemble, such that it converges to different local minima.
	- Diversity encouraging ensemble (DEE): Use a base network trained for a small number of epochs as initialization for $n$ different networks, each trained with dropout to encourage diversity
	- Split head approach: One base model has multiple heads, each corresponding to one classifier.

## Loss Prediction
- The loss objective guides model training. a Low loss value indicates that a model can make good and accurate predictions. Yoo and Kweon (2019) designed a loss prediction module to predict the loss value for unlabeled inputs, as an estimation of how good a model prediction is on the given data.
- Data samples are selected if the loss prediction module makes uncertain predictions (high value loss) for them. The loss prediction module is a simple MLP with dropout.
![[Pasted image 20240527163126.png]]
- A simple MSE loss for training this model to predict true loss is not a good choice, because the loss decreases in time as the model learns to behave better. A good learning objective should be independent of the scale changes of the target loss.


...
## Adversarial Setup


==SAM NOTE: I'm stopping at this midpoint.==
