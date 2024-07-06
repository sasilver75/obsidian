---
aliases:
  - CNN
---
Variants: 
- [[Region-Based Convolutional Neural Network]] ([[Region-Based Convolutional Neural Network|R-CNN]])

CNNs implement a ==hierarchical feature extraction approach== in which ==features are extracted at progressively higher levels of representation.==
- They extract features by implementing *kernel convolutions* on an image, where we transform an input image $f$ into an output image $g$ by sliding a filter/kernel $k$ (that encodes some pattern) over the image and extracting the inner/dot product between the kernel and the overlapped pixels in $f$, resulting in a single pixel in the output image $g$.

Designing complex feature detectors (eg for an eye, or an ear) are difficult to create over raw pixels (to capture eyes in all their generalities). CNNs take inspiration from the visual cortex, which constructs representations from lower-order representations.

Typically, the number of channels in our CNN increase as we go deeper in the network, enabling the detection of a greater number of higher-level features. At the end of the network, we'll flatten our data and pass through some FFNN layers with some appropriate head.

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
		- For each sub-region, and, instead of doing a usual kernel convolution with inner product, we output the *maximum value* of the subregion of the feature map covered by the filter. So we just retain our strongest detections. Along with Stride, helps us to reduce our feature map resolutions.
	- [[Average Pooling]]
		- Calculates the average value in the elements in the region of the feature map covered by the filter.
		- Ensures that every element contributes equally to the output, which can sometimes lead to better generalization in certain types of problems.
- Stride
	- Just tells us how many pixels to jump each time we move the kernel, as we scan it over our image. By increasing stride, we can produce outputs that are smaller than the input image.


![[Pasted image 20240705231402.png]]
Examples of Kernels being applied to images, in image processing.
- Gaussian Blur spreads the values of each pixel to every other pixel in the neighborhood
- Sharpening does the opposite, sharpening differences between neighboring pixels.
- Edge detection picks out regions where there's a sharp contrast in intensity.

