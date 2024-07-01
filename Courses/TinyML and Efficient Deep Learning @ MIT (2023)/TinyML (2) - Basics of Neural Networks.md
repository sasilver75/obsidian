https://www.youtube.com/watch?v=Q9bdjoVx_m4&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=3

Slides: https://hanlab.mit.edu/courses/2023-fall-65940

Last lecture we introduced some of the interesting problems that are being solved in AI (self-driving, image generation, etc.) as well as the massive amount of computing needed to power it, which motivated the need for efficient techniques.

This should be an advanced course, so we suppose that you have some knowledge about ML as well as computer architecture (C, C++ programming), and with some system knowledge (page table, cache locality) as well as deep learning concepts.

----

(Showing some more examples where {insert efficient technique} accelerates inference from 12 images/s to 842 images/s, etc. More just stuff motivating learning about acceleration techniques)

![[Pasted image 20240628014619.png]]


![[Pasted image 20240628015303.png]]

![[Pasted image 20240628015313.png]]
FFNNs are some of the most popular layer types; we have some amount of input nodes, and some amount of output nodes.
- We use a linear transformation to take a linear combination of input features, and then add a bias vector (optional, in many networks) to the result.
- Our weight matrix $W$ projects from the input to the output dimensionality.

What if we have more than one input? Maybe we have another dimension, which is the batch dimension!
![[Pasted image 20240628015501.png|400]]
See now that our input X is now a matrix with dimensionality (n, c_i)
==See that our weight matrix didn't change its dimensionality at all!== 

Let's look at a convolutional layer
![[Pasted image 20240628015727.png]]
We have a bit input feature map, and the output is only related to some subset of the input, for example.
A common one-dimensional signal (with a channel dimension too, often) is something like a speech signal!
- In the picture, it looks like a "two dimensional feature," but it seems like the speaker doesn't really count that as an additional dimension. In the speech analysis example, each "row" in that blue matrix is basically going to be a sample from a timestep.

Each output (green cell on the right) depends on some patch of the input.
![[Pasted image 20240628020008.png]]
We can add additional filters to produce additional output channels.

As we slide this filter along our input (eg by one)...
![[Pasted image 20240628020048.png]]
... we get an additional vector produced in out output. 

![[Pasted image 20240628020125.png]]
(Again)

![[Pasted image 20240628020145.png]]

What if we had something like an image as input, where we have two spatial dimensions and a channel dimension (eg RGB?)
- Now we're considering *2-D Convolutions, instead of 1-D convolutions* (though of course our tensor is a 3-dimensional tensor, but one is the channel dimension).

![[Pasted image 20240628020529.png]]
We convolve our filter tensors to produce a single output

![[Pasted image 20240628020541.png]]
similarly, if we have three filters, we have three output channels

As we move our filter by one, we get an additional element in our output.
![[Pasted image 20240628020601.png]]

![[Pasted image 20240628020625.png]]
We shift again

![[Pasted image 20240628020641.png]]
And again, but this time going to the next "row" -- see this effect on the output?

![[Pasted image 20240628020711.png]]
Eventually, we've completed the full 2-D convolution using a 3x3 filter.
- Our input tensor was 5x5, and our output tensor was 3x3, using 3 filters (so therefore we have three output channels).

![[Pasted image 20240628020852.png]]
Consider again the dimensionalities (use the notations in the bottom left)

Let's see why the input is 5x5, but the output is 3x3
We've lost two pixels, what happened here? And what's the general relationship between the input/out width/height?

$h_o = h_i - k_h + 1$
The output height is the input height minus the kernel height, plus one
In our example:
$x = 5 - 3 + 1$
$x = 3$

![[Pasted image 20240628021416.png]]
But if we want the output to be the same as the input, we can pad the input!
- [[Zero Padding]]: Just pad the boundaries with zero; this is the default in PyTorch
- [[Reflection Padding]]
- [[Replication Padding]]

With padding, the equation for the output height changes to:
$h_0 = h_i + 2p - k_h + 1$
(Where p is the number of padding cells (eg) on one side of your image, to the outward edge).


![[Pasted image 20240628021252.png]]
In order to understand the image, we have to see a global view, or at least a larger patch of the input. To understand what the lecturer is doing right now, you need to see his *whole body*.
So how do we calculate the [[Receptive Field]] when we have multiple convolutions in a chain, like in many CNNs?
How many pixels in an earlier pixel map will impact a given later pixel after multiple convolutions?
- In the image, if we consider the pixel maps as A,B,C,D ... and we're curious what the receptive field size is for the highlighted pixel in D with respect to B, we can see that it's a 5x5 receptive field. We can solve for it using the equations above.

If we want to enlarge the receptive field, we can increase the number of layers (making it slower), or increase the kernel size (which means more weights, too). Is there another way to enlarge the receptive field without incurring a large amount of computing?
- The method is to downsample the feature map by using downsampling (eg Strided convolutions)

![[Pasted image 20240628022259.png]]
- Above: 
	- On the bottom, with a 3x3 kernel, we need 3 convolutions to get us to a 7x7 receptive field.
	- On the top, with a stride of 2, we increase our receptive field to 7x7 in only two convolutions!


![[Pasted image 20240628022455.png]]
Previously, all of the output depended on all the inputs. [[Grouped Convolutions]]
(His explanation for this is a little jumbled)
Grouped convolution is the an effective way to reduce the number of weights.

![[Pasted image 20240628022714.png]]
Rather than just having two groups, we can have as many groups as the number of input channels! [[Grouped Convolutions|Depthwise Convolutions]]
- Each output channel can be only dependent on one input channel, even if we have 8 channels.
This is the foundation of the MobileNet family that we're going to introduce.


![[Pasted image 20240628023025.png]]
Another way to downsample other than strided convolutions, to increase the receptive size, is to use [[Pooling]].
- [[Max Pooling]], [[Average Pooling]]


![[Pasted image 20240628023330.png]]
After we pick a region (Batch, Layer, Instance, Group), we find the mean of that region and the variance/standard deviation of that region.
- We center the distribution to mean 0 and variance 1 by doing $\hat{x}_i = (1/\sigma)(x_i - \mu_i)$ 

But how do we pick a region over which we want to normalize? 
- [[Batch Normalization|BatchNorm]] selects from the batch dimension, plus the height/width dimension. For each channel, we have its own independent mean and standard deviation.
- [[Layer Normalization|LayerNorm]] selects a specific element from our batch, over the channel dimension and h/w. So each element in the batch has its own mean and standard deviation. 
	- e.g. Before you do the attention among different tokens (eg in a language model), you want to make sure you normalize them before doing attention between them.
- Instance Norm: Only one channel, only one element from the batch; across the hxw
- Group Norm: A subset of the channels for a single element

![[Pasted image 20240628024113.png]]
Some activation functions are difficult to quantize and are not hardware-friendly.

We'll have two lectures on Transformers coming up
- For each Transformer block, we have multi-headed attention and a fully-connected feed forward layer.  

Later in the lecture, we'll introduce other techniques like [[Multi-Query Attention]] (MQA) and [[Grouped Query Attention]] (GQA).

![[Pasted image 20240628024510.png]]

![[Pasted image 20240628024523.png]]
Later people designed VGG net which was much more homogenous in terms of the hidden size.

![[Pasted image 20240628024550.png]]
ResNet introduced Residual Connections to CNNs

![[Pasted image 20240628024710.png]]
He's goin crazy

![[Pasted image 20240628024722.png]]
The main difference in MobileNet is that we actually use a 3x3 depthwise convolution, a form of grouped convolutions where the number of groups equals the number of channels.

-----

# Talking about metrics, etc.


![[Pasted image 20240628114221.png]]
See that we have memory-related and compute-related metrics
Memory-related:
- Parameters
- Model size
- Total/Peak activations
Computation-Related:
- MAC
- FLOP, FLOPS
- OP, OPS


==Latency== measures the delay for a specific task. For a video segmentation model, 640ms or 63ms might refer to the amount of time to process a single frame.

==Throughput== measures the rate at which data is processed. Processing 6.1 videos/second, or 77.4 videos/second

Does higher throughput translate to lower latency?
Does lower latency translate to higher throughput?

![[Pasted image 20240628114418.png]]
It does in the case where we're talking about serial execution
But in the case where we can have multiple jobs in process, we can have longer latency but still good throughput.   
- So with more cuda cores or more GPUs can increase throughput for many problems
- ==Reducing latency is not easy==, relative to reducing throughput.

So how do we reduce Latency?

![[Pasted image 20240628121358.png]]

![[Pasted image 20240628121652.png]]
We want to try to avoid memory reference!
Computing is cheap -- memory is expensive (in terms of energy usage!)

![[Pasted image 20240628121902.png]]
So how do we calculate the number of parameters in a NN?
- For a linear layer, it's just the input dimensionality times the output dimensionality.
- For a convolution layer, it's the input channel size times output channel size times kernel height and width.
- For grouped convolutional layers, it's the same , but we divide by g

Model size measures the storage for the weights of the given neural network
- The most common units for model size are in MB (megabyte), KB (kilobyte) or bits

Model Size = \#Parameters * Bit Width

![[Pasted image 20240628122017.png]]


Next let's talk about the total and peak activations

![[Pasted image 20240628122035.png]]
From ResNet to MobileNet, the total activations decreased, but the peak activation increased! And sometimes your peak activations is what determines your peak memory usage. This is one of the problems of inbalanced activations across your network, especially for models meant to deploy on (eg) mobile devices.

![[Pasted image 20240628122149.png]]
Early layers in CNN we have big resolution... that gets pooled, and becomes smaller
In later layers, we have a lot more channels, making the weight become larger.


![[Pasted image 20240628122324.png]]
MAC Operations: Multiple-Accumulate Operation
- As we can see, the GEMM has more compute than matrix-vector multiplication




