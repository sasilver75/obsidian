---
aliases:
  - CNN
---
Variant: [[Region-Based Convolutional Neural Network]] ([[Region-Based Convolutional Neural Network|R-CNN]])

Vocabulary:
- Padding: Can be used (when considered with kernel size) to control the "height" and "width" of the tensor that results from a convolutional layer (h,w,c,bs). Without padding, the spatial dimensions of the output feature maps would shrink with each convolution layer, which *might* not be desirable. They also ensure that information at the "edges" of the image are not lost.
	- [[Same Padding]]
		- Used to ensure that the output feature map after convolution has the same spatial dimensions as the input. "Same" refers to the output size being the same as the input size, rather than the adding method itself (which often uses 0s in practice).
	- [[Zero Padding]]
		- Refers to the process of adding layers of zeros (0s) around the input matrix or image before performing convolution.
	- Valid Padding
- Pooling: Can be used to control the "depth" of the tensor that results from a convolutional layer (h,w,c,bs)
	- [[Max Pooling]]
		- The most common pooling operation.
		- For each sub-region, outputs the maximum value of the subregion of the feature map covered by the filter. This captures the presence of certain features in the regions of input... and makes the detection of features invariant to small shifts and distortions.
	- [[Average Pooling]]
		- Calculates the average value in the elements in the region of the feature map covered by the filter.
		- Ensures that every element contributes equally to the output, which can sometimes lead to better generalization in certain types of problems.
- Stride
