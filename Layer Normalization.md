---
aliases:
  - LayerNorm
---

![[Pasted image 20240628023330.png]]
After we pick a region (Batch, Layer, Instance, Group), we find the mean of that region and the variance/standard deviation of that region.
- We center the distribution to mean 0 and variance 1 by doing $\hat{x}_i = (1/\sigma)(x_i - \mu_i)$ 

But how do we pick a region over which we want to normalize? 
- [[Batch Normalization|BatchNorm]] selects from the batch dimension, plus the height/width dimension. For each channel, we have its own independent mean and standard deviation.
- [[Layer Normalization|LayerNorm]] selects a specific element from our batch, over the channel dimension and h/w. So each element in the batch has its own mean and standard deviation. 
	- e.g. Before you do the attention among different tokens (eg in a language model), you want to make sure you normalize them before doing attention between them.
- Instance Norm: Only one channel, only one element from the batch; across the hxw
- Group Norm: A subset of the channels for a single element