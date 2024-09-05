2014
Introduced the ==Inception Module==, where each inception module consists of parallel convolution layers with different filter sizes (and max pooling layers). We apply these to the same input in parallel, and then concats them, combining low-level and medium-level kernels together.

![[Pasted image 20240701102334.png]]
![[Pasted image 20240701102345.png]]

The 1x1 Kernel ("Pointwise Convolution") just multiplied each pixel with a single fixed value; this is why they're often called pointwise convolutions. They basically just scale the input. 
![[Pasted image 20240701102438.png]]
Effectively combining the projections of different input channels, and mixing them together to go from a four channel input to a three channel output.
![[Pasted image 20240701103504.png]]
![[Pasted image 20240701103534.png]]
Very parameter efficient, and a good way to do dimensionality reduction to reduce the number of channels before doing spacial filtering with 3x3 and 5x5 convolution layers. This helps to cut down the number of weights in the network; using 1x1 convolution to decrease he number of channels, then use spacial filtering with 3x3/5x5 kernels, then use another 1x1 convolution to expand the number of channels back.