https://www.youtube.com/watch?v=w5WiUcDJosM&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=5

An important subject in model compression is pruning and sparsity. We'll have two lectures on this subject.

---

![[Pasted image 20240628123021.png]]
The MLPerf competition measures how fast you can run a workload at a given degree of accuracy. In the closed division (you cannot change the NN, you have to use hardware innovation), there's also open division (openly try different ideas; you and prune, compress, quantize, as long as accuracy is well-maintained).
- See that in the Open Division, they're able to achieve incredible performance; 4.5x faster!

They achieved that using three major techniques
![[Pasted image 20240628123227.png]]
- [[Quantization-Aware Training]]
- [[Pruning]]
- [[Distillation]]

Today we'll introduce Pruning, which can reduce the parameter counts of NNs by more than 90%, reducing storage requirements and improving speed/efficiency of NNs.
- WE'll introduce the different granularities of pruning, and the criteria for each of them.

We want to reduce memory access as much as possible
![[Pasted image 20240628123416.png]]
As soon as we go to DRAM, it becomes two orders of magnitude more memory consuming! So it's really memory reference that's draining the battery of our phones! So we really want to reduce these memory accesses.
- If we can reduce the number of weights in a NN, we can reduce DRAM access, and save power!

What is Pruning?
- Basically turns a dense NN into a sparse NN

![[Pasted image 20240628123524.png]]

![[Pasted image 20240628123652.png]]
Removing here means setting it to zero; because zero multiplied by anything is still zero, we don't really need to compute on it, so we can save memory/computation.


Here's an experiment conducted with an old [[AlexNet]] model
![[Pasted image 20240628123814.png]]
We can see that as we prune away additional parameters, performance suffers (though see the Y axis, it's not so bad!) -- but we can remove ~50% of parameters with very little performance degradation!

Can we do better?
- If we both prune AND finetune, we can do better!
![[Pasted image 20240628123901.png]]
We can finetune the remaining weights to recover the accuracy, having almost no degradation of accuracy! We can remove ~85% of weights with ~no loss to accuracy!

So we Train it -> Prune it -> Retrain it
We can even do this process iteratively, pruning it a little, retraining, pruning it a little
![[Pasted image 20240628124058.png]]
This is pretty amazing! With almost no loss of accuracy, we can reduce the size of our model significantly!

![[Pasted image 20240628124542.png]]
Even small models can be significantly compressed! A 2015 experiment, I believe.


![[Pasted image 20240628124626.png]]
Even language models can be pruned! (The right one is the one that's incorrect)

![[Pasted image 20240628124732.png]]
Lots of people are using sparsity to accelerate LLMs in industry, but this research started back in the 1980s with a paper called *Optimal Brain Damage!*

![[Pasted image 20240628124822.png]]
In 2016, the start of attempts of hardware acceleration that supports sparse matrices.


Okay, so pruning is pretty cool- but what really is pruning, and how do we formualte it?

![[Pasted image 20240628124955.png]]
Previously, the weight is denoted as $W$, but after pruning we call it $W_p$
- We want to make sure that the number of 0s is smaller than some target pruning ratio; a target number of non-zeros
(This slide doesn't really say anything, lol)


==So how do we determine the pruning granularity that we use? In what *pattern* should we prune the network?==
![[Pasted image 20240628125109.png]]
Do we prune individual cells in our matrix, or prune groups of them?

==And what should our pruning criteria be?==

==And what should our pruning ratio be? What sort of target sparsity should we have, for each alyer?==

==How do we improve the performance of pruned models?==


# 2) Pruning Granularity
- Pruning can be performed at different granularities, from structured to non-structured. You can imagine that structured pruning will have better efficiency.
![[Pasted image 20240628125324.png]]
In is ==fine-grained, unstructured pruning==, the pruning looks completely random; we can have the most flexibility to prune our model, but it's hard to hardware accelerate, because it's irregular, which hardware doesn't like.

There's ==%% coarse-grained, structured pruning %%== too -- We might choose to prune away entire rows of our matrix.
- The benefit of this is that it can still be considered as a dense matrix, so we can use our usual hardware acceleration!
- But this is certainly a less flexible method of pruning -- we might be pruning "good" weights in our model too. 

Let's talk bout pruning in the context of convolutional layers
- The weights of our convolution have 4 dimensions
	- Input channnel dimension
	- Output channel dimensions
	- Kernel size ehight
	- Kernel size width
- These four dimensions give us more choices of what to prune, as opposed to the fully connected layer we see in the slide above (with just the input and output dimensions)
![[Pasted image 20240628125537.png|100]]
Above: The notation for describing our 4d matrix
![[Pasted image 20240628125524.png]]
- In fine-grained pruning, we can basically pick any of these dimensions; any element could be zero.
- We can make it have a little better regularity, enforcing some structure using some patterns (like Tetris)!
- We can perform vector-level pruning where we prune entire rows to be zero.
- We an even do Kernel-level pruning, where entire 3x3 kernels are either all zero or all non-zero.
- We can go more irregular, and prune entire channels too.
So there's a very broad spectrum of pruning regularity for convolutional operations.

![[Pasted image 20240628131721.png]]
If you look at the 1x4s that make it up, only two of them are non-zero
- For every 4 elements only 2 of them are non-zero. The sparsity ratio is 50%
- So we can condense it, so that each row, instead of having 8 elements, has 4 elements
- But we have to pay some overhead to store the INDEX of the non-zero activations in each row.
	- So for the first 1x4, it's 0 and 3
	- For the second 1x4 it's 1 and 2
	- So we can store these indices for the first row as 4 bits (2 bits for each 1x4 group)


![[Pasted image 20240628132042.png]]
We don't straightforwardly always get speedup from reducing the number of parameters!
Speedup also refers to the ease of parallellization and acceleration in hardware!
- This is why the pattern-based pruning we saw above is useful.


Remember our spectrum of irregular -> regular pruning techniques?
![[Pasted image 20240628132218.png]]

The most regular of them in Chanel-level pruning, where we're left easily with a very dense matrix.

![[Pasted image 20240628132637.png]]
Channel Pruning is very widely adopted in industry; at a mobile AI company, this is probably *the* technique used, because it's so easy for hardware to accelerate.
- At each layer, we just prune different channels to make it a less number of channels.
- We can either uniformly shrink our layers (selecting sparsity ratios), or wisely select different sparsity ratios for each layer.


# 3) Pruning Criterion (Which neurons should we prune?)

![[Pasted image 20240628132904.png]]
In this simple multilayer perceptron with three weights w0, w1, w2... we combine these with our inputs x0, x1, x2 and pass through an activation function

If we wanted to remove one weight here, which should we remove, intuitively?
- Probably the one here with the smallest magnitude, 0.1 (w_2)

==So this is our starting point -- selecting pruned neurons based on magnitude.==

![[Pasted image 20240628133024.png]]
This seems super simple, but it's been *the method* used in industry for the last five years! For industry use, we want a good balance of effectiveness and easy of use.

[[Scaling-Based Pruning]]
![[Pasted image 20240628134026.png]]
When we're doing learning, we want to encourage this scaling factor to be as close to zero as possible, using L2 or L1 regularization. 
![[Pasted image 20240628134832.png]]
Before pruning ,we have 5 channels -- two are pruned, so we're left only with those that have big scaling factors.
Luckily, the scaling factor is already there in the Batch Normalization layer; we can just re-use the scaling factor in the batch normalization layer to determine which channel to prune.

[[Second-Order-Based Pruning]]
![[Pasted image 20240628134909.png]]
Here, delta means changing from your original weight to zero
Taylor expansion showing first, second, third order term.
![[Pasted image 20240628135000.png]]



![[Pasted image 20240628135155.png]]
As soon as a neuron is pruned, all of the weights associated with a neuron are pruned too!
- So neuron pruning is a coarse-grained version of weight pruning.

Let's see an example of a convolutional NN, determining how we determine which neurons or activations to prune:
![[Pasted image 20240628135408.png]]
Our ReLU activation function will generate zeros for any inputs less than zero.
Our task it to determine, among these three RGB channels... which channel we should pruen away?
- A simple heuristic is just to examine the number of zeroes in the channel -- but we have different batches! So we want to look at this across batches.

![[Pasted image 20240628135514.png]]
We can define this ==APoZ== metric to determine which channel to prune.





