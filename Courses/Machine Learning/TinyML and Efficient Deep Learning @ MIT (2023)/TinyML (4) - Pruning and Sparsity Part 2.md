https://youtu.be/3t9aGLLaCqs?si=Plo8qcBnzXPP_7gN

----

Recall: The definition of pruning is to remove some of the weights in our network to zero.
![[Pasted image 20240630145126.png]]
We can have different granularities
- Fine-grained gives maximum flexibility, but is hard to hardware accelerate.
- Coarse-grained channel-level pruning give highly structured pruning that's easier to hardware accelerate.

We also talked about different ways to select the synapses to prune.
![[Pasted image 20240630145219.png]]
- In [[Magnitude-Based Pruning]], we use the heuristic that weights with larger absolute values are probably more important than other weights, we we prune the weights with smallest magnitude.

We also talked about ways to select entire *==neurons==* to prune (really, just meaning all of the activations that "connect" to this neuron; this ultimately means some vector in the matrix, I think-- or even an entire channel, in convolutions)

![[Pasted image 20240630151214.png]]
If you remove a neuron, all neurons connected to the neuron are removed


# Pruning Ratio
So how do we determine our Pruning ratio? What should our per-layer sparsity target be?
![[Pasted image 20240630151342.png]]
Uniform shrinking vs choosing different levels of sparsity for different layers.

![[Pasted image 20240630151751.png]]
Some layers can be heavily pruned without much loss in accuracy, while others are sensitive to pruning.
We should be able to perform some sort of sensitivity analysis:
![[Pasted image 20240630152424.png|400]]
We can choose another few layers and plot them in different colors
![[Pasted image 20240630152456.png]]
And see that charts of different layers can be pruned more or less aggressively. It would be best to prune the L1 layer here, which is pretty pruning-insensitive.

...but is there a catch with this?
- We have to trade-off how much experiment/gpu-hours we want to use to do these evaluations.
- Recall that our goal is to balance between compute, accuracy, and model size. In this chart, we don't know the absolute size of these varying layers! So maybe the green layer L1 is a super tiny layer with only 10 parameters -- even if we prune 80% away, we only lose 8 parameters -- whereas maybe the blue L0 layer is *huge*, so pruning even 10% would mean a massive reduction in parameter count.

In Transformers though, it's common that each layer is pretty homogenous in size, with some common fixed hidden dimension.
- In contrast, for Convolutional NNs, however, the size of various hidden layers can differ drastically! So take into account not only the accuracy degradations from pruning a layer, but also the layer absolute size.

![[Pasted image 20240630153014.png]]
Here's a heuristic, engineer-based idea: We set some certain threshold, and select the pruning ratio by this threshold; for the Blue layer, for example, we prune away ~70%, and for the purple curve, we can prune away 80%, etc.
- There are more advanced pruning algorithms too that might be based not only on accuracy, but also on the size of layers.

![[Pasted image 20240630153127.png]]
Recall again that this ==might not consider the interaction between layers== -- we assumed that different layers' sensitivities are independent!

How can we go beyond simple heuristics?

Let's motivate automatic pruning:
![[Pasted image 20240630153214.png]]
![[Pasted image 20240630153340.png]]

At the time (2018), they developed AMC: AutoML for Model Compression
- They phrased pruning as an Actor-Critic reinforcement learning problem.
![[Pasted image 20240630153455.png]]
The lower the error of the model the better (Reward = -Error), but we also modify this this with the log of the FLOPs -- we want to make sure that we have a low amount of computing. But why Log FLOPs?
![[Pasted image 20240630153610.png]]
People empirically found that the number of operations and accuracy had a log relationship, which is why they formulate the reward like this.

We treat it as a game! Imagine we have a 3-layered NN
- We have a game, where we have three actions, where each action is to select (rather than "left" or "right"), we select from a continuous space, which is the pruning ratio. After we select the pruning ratio for all the layers, we get a reward, which is the accuracy. We can repeat this process to find the optimal pruning ratio.
- ((I don't really get how RL helps, because this isn't an iterative problem, is it? Hmm))

![[Pasted image 20240630154101.png]]
![[Pasted image 20240630154109.png]]
They found an interesting pattern about the pruning ratio of different layers in ResNet-50:
- Remember we're doing iterative pruning: WE prune a little, finetune it, prune again.
- We find that for some layers, the agent automatically prunes very aggressively, whereas some layers are pruned much less aggressively.
	- The aggressively pruned layers are the 3x3 convolutions.
		- Why aggressive for 3x3? There's a lot of parameters here!
	- The less aggressively pruned convolutions are 1x1 convolutions. If you prune these, you only prune a single weight.
(This goes back to our previous sensitivity analysis chart where we said the chart wasn't taking into account the absolute size of layers!)


----

# Fine-Tuning  Pruned Neural Networks
## How should we improve performance of pruned models?

Okay, once we've pruned our model, we've lost some performance. What can we do to recover some of that performance?
![[Pasted image 20240630170555.png]]

![[Pasted image 20240630170940.png]]
After we prune the NN, we should decrease the learning rate to ~1/10 or 1/100 of the original learning rate.
- Why do we decrease the learning rate? The model has *pretty much* already converged, so we want to use small learning 

![[Pasted image 20240630171042.png]]
If you want to target a 90% pruning ratio, we can either:
- Jump immediately there in one big step
- ==Iteratively prune==; each time, we prune to 20%, 40%, 60%, then to 90%... finetuning before pruning again.
![[Pasted image 20240630171356.png]]
If you just have a single model to prune, he recommends that you use this iterative pruning approach.

Another challenge we face when making a product... is bypass layer.
- The bypass layer (resnet?)... We can't do an add if the two matrices aren't the same size.
((Not clear what's talking about, but there==might be some considerations for pruning with residual connections==))


![[Pasted image 20240630171640.png]]
When we're finetuning during our iterative pruning loop, we want to add different [[Regularization]]s, too!!
- Penalize the non-zero parameters; we want to encourage as many parameters to be zero as possible.
- We want to encourage smaller parameters. If it's small, then it's likely that during the next pruning iteration, they'll be pruned.

So the most common is to use either [[L1 Regularization]] or [[L2 Regularization]]

-----

![[Pasted image 20240630172304.png]]
This paper explored certain cases (eg CIFAR, MNIST)... but how do we find this sparsity? We still have to train it to convergence, and then prune it.

![[Pasted image 20240630172453.png]]
 We train to convergence, and then prune.

----

![[Pasted image 20240630173002.png]]
How do we actually get speedup from our pruning?
- If we do unstructured, fine-grained pruning, we can sparse-out our network, but not get much speedup, because there's not system support for sparsity.

![[Pasted image 20240630173130.png]]


![[Pasted image 20240630173215.png]]

![[Pasted image 20240630173201.png]]









