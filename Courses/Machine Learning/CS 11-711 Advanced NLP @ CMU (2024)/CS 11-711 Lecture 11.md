# Topic: Distillation, Quantization, Pruning
https://www.youtube.com/watch?v=s9yyH3RPhdM&list=PL8PYTP1V4I8DZprnWryM4nR8IZl1ZXDjg&index=11

---

Training big models is expensive!
But inference is actually *more expensive*, with costs far exceeding training costs when deploying models at any reasonable scale!

The answer is Model Compression! (Taking a trained model and reduce the size of it, pre-deployment))
1. [[Quantization]]: Reducing the number of bits, with the same architecture/parameter count
2. [[Pruning]]: Remove parts of the model (parameters) while retaining performance
3. [[Distillation]]: Training a smaller model to imitate the bigger model (might change architecture too)

Instead of taking a big model and making it smaller, why not just start with small model and train it from scratch?

Why is it possible to take a big model and throw pieces of it away without sacrificing accuracy?
- Our models are often overparametrized, meaning we have more parameters than training data (more than statistical learning would say we would need)
- Training NNs for most tasks requires optimizing a non-convex objective, where we aren't guaranteed to find the global optimum. If we have a bunch of parameters, having a lot of parameters lets us sidestep around both saddle points and bad local optima; we can take shortcuts around barriers in optimization space (see CMU Convex Optimization course)
	- But we don't need all these parameters for inference; it's a training time trick!

## Quantization
- Most obvious way is ==post-training quantization,== where the model is trained at whatever precision you want, and then we quantize the weights after training.

If we train a LLaMA 65B model with 4Byte/32Bit precision, that would take 260GB of GPU memory, which is more than most single GPUs will have;
- If we reduce the precision of weights in the model, we see a pretty massive decrease, to the extent that we can fit it onto consumer GPUs.
![[Pasted image 20240617095452.png|400]]

Refresher on Floating Point Numbers
- NNs represent weights as floating point numbers to have a broad range of values in the model.
- Floating points have
	- A sign bit (pos/neg)
	- A fractional bit, which specifies the specific values
	- Exponent, which scales how big or small the float is
![[Pasted image 20240617095618.png]]
![[Pasted image 20240617095716.png]]
==Float16== is pretty common, but for ML, we often have bery small or big values (underflow/overflow), so a very popular datatype designed for ML is balled ==[[bfloat16]]==; the idea is we just move some bits from the fractional to the exponential part, so we have a bigger range of values.

A way to get a smaller footprint in models is to quantize to integers
![[Pasted image 20240617095921.png]]
AbsMax quantization maps each number in a list of floats to a range of integers; eg if the maximum value in our dataset were 20, then it becomes 127, and everything else is proportional to that.

![[Pasted image 20240617100213.png]]
One bit per vector seems pretty attractive, right?

![[Pasted image 20240617100302.png]]
But by reducing precision, we might be reducing the range of precision we might be able to express 😟


==Model-Aware Quantization==: If you can study the statistics of your model, you can learn ways to represent values that actually matches the distribution of weights in the model!
- In BERT, most weights in each layer are concentrated around a mean value, and most weights are clustered around that; you can fit a gaussian distribution to the distribution of weights; only a few weights will be at the tails of this distribution.
![[Pasted image 20240617100442.png]]
The idea here is that we basically store the outliers *separately, in full precision* (because think of what they'd do to our quantization in the context of AbsMax; it throws off our range), and everything else gets quantized to lower precision.


A problem with that approach is that we define the outliers/min/max for each layer uniformly! 
![[Pasted image 20240617100651.png]]
Instead, this ==LLM.int8== strategy goes a step further, and, instead of quantizing each layer uniformly, they quantize each row or column of a vector separately, with the motivation that most of the parameters in transformers are for MatMuls; by doing this, they get a better quantization because they have a more precise range of values for each row/column of a matrix.
- This is a very popular quantization in NLP, apparently!
[[Model-Aware Quantization]]

There's an overhead to pay when doing this quantization, where, for each vector, you now have to map it to a list of numbers, and then later decode your ints back into floats when doing inference.
- This can still double your inference speed for large models (and allow you to have larger models)
![[Pasted image 20240617101305.png]]

----

If you're someone interested in algorithms, you might have very creative ideas for quantization, but the ability for quantization to make things faster is really limited by:
- Hardware
	- Some datatypes (eg Int3) aren't really supported by hardware; Int3 would just be treated like Int4
- The framework you're running models on (PyTorch)
	- PyTorch doesn't even have support for Int4!

Models that *do have this work* are writing custom hardware accelerators to make this work!

---

Okay... let's say that we *know* that we're going to do post-training quantization of our model -- can we train our models with that in mind?
![[Pasted image 20240617102018.png]]
Binarized NNs didn't work earlier, but they *can* work, if you train with binarized quantization in mind!


![[Pasted image 20240617103147.png]]
Above: Layer-by-Layer Quantization-Aware Distillation
- For doing quantization, you can also start with a model that's full precision, and then train each layer (one layer at a time) to replicate its counterpart in the full-precision space.
	- Then you do this for the second layer (now you have the hidden states from the second to last layer, and train your quantized layer to match those hidden states)
- The intuition here is that by doing layer-by-layer distillation, you're replicating not just the output (sparse), but the flow of data throughout the whole model, step by step.

![[Pasted image 20240617103343.png]]
In QLoRA, they use PEFT to train a highly-quantized 4-bit model, and they do a bunch of other fancy tricks.
- If you're going to use a quantization method today, this is probably it.

---
## Pruning
- Pruning is removing parameters from the model after training (rather than chipping away at every parameter in our model with quantization, here we're just lopping off some parameters)
![[Pasted image 20240617103425.png]]

Magnitude Pruning
- The most intuitive way to do this is that if we have a bunch of parameters, some of them are very close to zero, and so we assume they aren't doing anything, and we set them to zero.
- [[Magnitude-Based Pruning]]![[Pasted image 20240617103641.png]]
- In [[Machine Translation|MT]], people have seen that you can remove ~half the parameters in your model and still retain the same performance.
	- This goes back to the thing of over-parametrization, where it's helpful to have a lot of parameters to train your model, but not to do inference, so you can remove them.

![[Pasted image 20240617103732.png]]
Related: The [[Lottery Ticket Hypothesis]]; the idea is that there are subnetworks of the model that are actually better initialization... than the original model ((?))
- If you have a model with 100B params, there are subnetworks of the model with (eg) 1B parameters that are actually better than the full model. 
- Here, they prune the model then retrain it, and find that a model pruned to 20% of the original size and then retrained is actually more effective than the original model 🤯
So it's all about finding good initializations of these subnetworks.
- Generally though, we don't think about pruning as a method to improve performance.


![[Pasted image 20240617103926.png]]
A paper from CMU: Wanda
- Magnitude pruning presumes that we can just decide which params we want to throw away based on how big they are... but it doesn't consider that there are systematic differences in the size of the inputs that come in... (Explanation on whiteboard that I can't see)


![[Pasted image 20240617104222.png]]
We can make vectors sparse, but if your hardware doesn't take advantage of the sparsity of your matrices, then you're just doing the same amount of work, but with more zeroes in your matrix.
- Right now, hardware for ML doesn't support sparse computations *that well.* 

Therefore, a more immediately useful idea is something called ==[[Structured Pruning]],== where, instead of picking parameters willy nilly across the whole model, we remove entire components/layers, and therefore we're pruning the model in a way that is really going to make a difference in the overall runtime of the model. (This is in contrast to unstructured pruning)
![[Pasted image 20240617104352.png]]

If you're training a Transformer model, you usually have many heads of attention; in practice, most of these heads of attention can be removed without any negative impact on the performance of your model
![[Pasted image 20240617104655.png]]
Paper showed that we could remove half the attention heads and get negligible impact on performance.

Generalizing this, recent work has proposed controllign other parts of your model:
- This paper proposes having two levels of masks in your model
	- Coarse Mask: Turning off large components, like full self-attention layers, full FFNN layers (replaced with identity matrices)
	- Fine Masks: Individual attention heads, removing individual hidden state dimensions (eg from 512 to 200 dimensions).
- The idea is to give two different levels of granularity at which you can turn off components; these learn using some held out validation data.
![[Pasted image 20240617104850.png]]
You can get pretty far with this idea!

With methods like this, we learn a type of control over our model; a set of masks... that's pretty expensive in terms of training budget, requiring a lot of GPU memory. If you want to prune a LLaMA 70B model, you'll need almost as much compute as it took to train it!

Enter...
Pruning with Forward Passes
![[Pasted image 20240617105102.png]]
- Can we do pruning without computing gradients at all? If we can run the model on our computer, can we prune it?
- Idea:
	- Randomly mask out all the different modules in the network, creating 100s/1000s variants of the model with different masks turned off.
	- They measure the performance of these different perturbed models, and do a regression to learn how much each impacts model performance, and then use regression weights to figure which models to turn off without impacting performance too much.


----

## Distillation
- The core idea in [[Distillation]] is that we train one model (student) to replicate the behavior of another model (teacher).
![[Pasted image 20240617105857.png]]
- In distillation, you might even have a different architecture! 

[[Weak Supervision]] (1995)
- The idea that if you have unlabeled text/images/data, you can produce things that are *like labels* that you can use like labels, but aren't written by humans.,
- You can train on these as if they were labels, and get pretty good performance:
- ![[Pasted image 20240617110024.png]]
- ==Self-Training==: Initialize a model with a handful of examples, train a classifier using that small number of points, have that model make its own (bad) predictions on a bunch of unlabeled text, use those pseudo-labels to update the model again, and do this iteratively.
	- A pretty classic method at this point
- Pseudo-Labels are also used when you don't have the ability to annotate thousands of examples, but you have a basic rule ("If a movie review says 'awesome', it's positive") that you use to label data. You can use these rules to construct pseudolabels to train an actual model on.

![[Pasted image 20240617110302.png]]

![[Pasted image 20240617110613.png]]
- They combine these two objectives together, and it's really effective. This seems like the right way to do it for sequences of text.


![[Pasted image 20240617110628.png]]
- [[DistilBERT]] is basically just [[BERT|BERT]], with the size reduce in half, and the same performance.
	- They take every other layer of BERT, and initialize the layer from one of the initial BERT model... and effectively did soft-target distillation. 

![[Pasted image 20240617110654.png]]
You can take a model trained on supervised learning, repeatedly distill it on itself (training the model to match its own soft targets)... and it bizarrely works, consistently improving performance of a model.
- Intuition: The soft target objective, which is different than what you'd train on using supervised learning, conveys more information to your model.


![[Pasted image 20240617112054.png]]
In [[Self-Instruct]] they want to do [[Instruction-Tuning]]; they have a vanilla language model generate arbitrary instructions, then produce responses to those instructions... and then train the *same model (GPT-3)* on that dataset, to learn instruction-following.
- Interesting bit:
	- The quality of your labels is only as good as your teacher is; so they found that they had to first generate the output-first for classification tasks, whereas they could do the (expected) input-first generation for non-classification tasks. I think this special behavior for classification was a result of the model output imbalanced inputs 🤔 for classification problems (eg all positive amazon reviews).
	- They call this idea ==Task Asymmetry== in the paepr.



Here's a CMU paper (the PhD student talking)
![[Pasted image 20240617112851.png]]
Idea: Let's forget that distillation is anything but a (data?) generator... distillation is one way to get training data for your model, but there might be other ways to get data as well that we're leaving on the table.
- Can we combine retrieved data with data generated from an LLM? Can we put these together and do even better?
- We:
	- Ask user to specify their task in a prompt
	- Given a prompt, retrieve existing *datasets* that might be relevant for the prompt
	- We take the retrieved dataset (likely to be high quality, but might not match the task the user cares about). We complement this retrieved dataset with generated data from our LM (maybe not as high quality, but matches user intentions more closely re: task).
		- (We also tried retrieving pre-trained models as well in your domain)
	- We put all these things together, finetune the small model on generated/retrieved datasets.
	- We were able to create small models that often overcame GPT3 (Which was the LM we used as a teacher) by leveraging both distillation as well as taking advantage of existing datasets (and models?) that were available on the internet.


[[Synthetic Data]] generation (effectively the same thing as hard-target distillation) is one of the hottest topics in NLP right now:
![[Pasted image 20240617113231.png]]
He saw this paper recently and thought it was cool.
He says it's a very exciting direction re: making dataset generation something more mature, that can be managed as an engineering problem.
This is from a paper called [[DataDreamer]]




