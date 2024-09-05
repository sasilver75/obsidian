---
aliases:
  - Channel-Wise Convolutions
  - Depthwise Convolutions
---
[[Grouped Convolutions]] are a variation of the standard convolution operation. In contrast to regular convolutions *group convolutions* divide the input channels into several groups! Each group performs its own convolution operations independently!
	- So if an input has 64 channels and the grouping parameter is set to 2, the input would be split into two groups of 32 channels each, and these groups would then be convolved independently. This approach reduces computational cost and can also increase model diversity by enforcing a kind of regularization, leading to potentially improved performance in some tasks.