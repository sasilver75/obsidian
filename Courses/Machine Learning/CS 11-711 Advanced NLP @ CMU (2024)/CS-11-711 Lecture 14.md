# Topic: Ensembling and Mixture of Experts
https://www.youtube.com/watch?v=MueCRSZ3RQ0&list=PL8PYTP1V4I8DZprnWryM4nR8IZl1ZXDjg&index=14

----

In 2024, many models exist! They have different architectures, and their training results in different P(Y|X) (conditional distributions). Can we harness multiple of them to eke out a few more percentage points of performance?

But wait, why are there so many models these days?
- Different architectures
- Different initializations
- Different pre-training/fine-tuning datasets
![[Pasted image 20240617145241.png]]

The easiest way to benefit from multiple modes is via [[Ensemble Learning|Ensembling]], where we combine predictions from multiple models.
![[Pasted image 20240617145520.png|300]]
- Why would we want to do this [[Ensemble Learning|Ensembling]]? 
	- Reduces the bias of a single model, offering a more robust prediction -- multiple models make somewhat *uncorrelated errors*, smoothing over idiosyncracies of the model.
	- Can incorporate the power of different inductive biases of different architectures.

==Linear Interpolation==
- Taking a weighted average of M model probabilities.
Mathematically, this looks like:
![[Pasted image 20240617150000.png]]
The second term is often set to a constant, independently of context -- but you can introduce a router component that outputs probabilities for models too (we'll see more on this in the MoE section).


==Log-Linear Interpolation==
- Weighted combination of the log probabilities, and then renormalizing so that we get an actual probabilistic output.
![[Pasted image 20240617150730.png]]
- We have an interpolation coefficient $\lambda_m$  (can be set constant or learned), and then on the right side we have the log probabilities. 
- We take the softmax too... why didn't we take the softmax in the linear interpolation example? Because our left term already was 0..1 and adding up to 1.
	- Here, we don't have that, so we use a softmax to make it into a probability distribution.


So which one should we pick, Linear or Log-Linear interpolation?
- ==Linear: "Logical OR"==
	- The interpolated model likes any choice that a model gives a higher probability.
	- Use models with models that capture different traits.
	- Necessary when any model can assign zero probability (eg models with different vocabularies; this is hard to do, think about whether you want to!)
- ==Log-Linear: "Logical AND"==
	- Only likes choices where all models agree
	- Useful when you want to have a model in the mix that helps restrict possible answers (eg a model averse to toxic language)
	- Your interpolation coefficients dont' need to be positive, they can be negative too! If you want to use a model as *negative evidence* that you want to remove
![[Pasted image 20240617151458.png|100]]

![[Pasted image 20240617153312.png|300]]


Another ensembling method that's been around for a long time is [[Bootstrap Aggregation|Bagging]], where we have a dataset, and we resample the dataset (with replacement) to get another dataset of (eg) equal size, and train on that (and do this like 10 times, train 10 different models, and ensemble those models).
- You can also get multiple models from different checkpoints of training a single model, and ensemble those.

These all just take advantage of seeing different data, data in a different order, etc. The idea is that errors will be uncorrelated across models.

### Efficient Methods for using Multiple Models
- The big problem with ensembling is the inference cost (requiring us to run multiple models at inference time).
	- N times the computation
	- N times the memory
- Is there any way that we can get the benefits of Ensembling without having to host multiple models?

Parameter Averaging (Utans 1996)
- A cheap way to get some good effects of ensembling; basically, we just average the parameters of multiple models together.
	- Need same architecture, parameter count

![[Pasted image 20240617154620.png]]


[[Task Vector]]s (Ilharco et al. 2022)
- Takes advantage of the fact that we're looking at different finetunes of the same base model.
- We have a base model, and the task vector is the difference between the baes model and finetuned model parameters.
- This allows us to do a number of interesting things:
	- We can subtract out "tasks" that we don't want; If we had a model trained on a lot of private text, the idea is that we could try to subtract out the task vector representing that knowledge.
	- Can take take two task vectors and combine them together, to try to get the model to learn from the combination of the two (this isn't exactly the same as averaging the parameters).
	- Also allows us to resolve conflicts from vectors of different tasks (TIES).
		- See that sometimes these model task vectors conflict; the process identifies the vectors that are pointing the most strongly in particular directions, resolves conflicts between them, and tries to have a vector that improves all tasks at the same time.
![[Pasted image 20240617155415.png]]


A popular toolkit called [[MergeKit]] makes it easy to do many of the things we've talked about here! Includes Linear, TIES.


Ensemble [[Distillation]]
- Problem: Parameter averaging only works for identical model architectures.
- Knowledge distillation trains a model (perhaps of a different architecture) to copy the ensemble!
	- Specifically, it tries to match the distribution over the predicted words (Soft target).
	- Why? We want the model to make the same mistakes as an ensemble.

---

## Sparse Mixture of Experts Models

Sparse Computation
- What happens when a scalar/tensor multiplication involves a scalar of zero?
	- The result is guaranteed to be zero -- there's no computation needed!
- This manifests itself in a bunch of places in models:
	- Single rows in a matrix multiplication (Can be optimized by a GPU automatically)
	- Larger tensors (Occurs often in sparse MoE models)
	- Whole models in an ensemble (Just don't neven need to use that model in the ensemble, if the probability of one of the models is 0.)

GPU-level Sparsity Support
- NVIDIA GPUs support various types of sparsity through the cuSPARSE
![[Pasted image 20240617162256.png]]

### Sparsely Gated Mixture of Experts Layer (Shazeer+ 2017)
- Select a subset of FFNNs to actually execute.
![[Pasted image 20240617162430.png]]
The gating method predicts some probability distribution over the experts, and then we keep the top k experts, and softmax over them to recreate a probability distribution.
- Different experts will be active for different parts of the batch.

Cascaded/Pipeline Systems
- In many cases, we hook the input of one system into the output of another system. An example of this is speech translation, where, given speech, we might do automatic speech recognition followed by machine translation.

Deliberation Networks
- Take in an output, and then iteratively refine it to make it better and better.
Diffusion Models
- Train models to generate with multiple steps. Basically we start with "nothing," and then gradually make it better and better.
	- Key difference with deliberation networks is that you can train diffusion moels from scratch by noising inputs.
	- Widely used in image generation, but not commonly used in text because autoregressive generation is pretty good.
- Self-Refine (Madaan et al 2023)
	- Feed in previous outputs to a LM in context