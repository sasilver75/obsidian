![[Pasted image 20240628021252.png]]
In order to understand the image, we have to see a global view, or at least a larger patch of the input. To understand what the lecturer is doing right now, you need to see his *whole body*.
So how do we calculate the [[Receptive Field]] when we have multiple convolutions in a chain, like in many CNNs?
How many pixels in an earlier pixel map will impact a given later pixel after multiple convolutions?
- In the image, if we consider the pixel maps as A,B,C,D ... and we're curious what the receptive field size is for the highlighted pixel in D with respect to B, we can see that it's a 5x5 receptive field. We can solve for it using the equations above.

If we want to enlarge the receptive field, we can increase the number of layers (making it slower), or increase the kernel size (which means more weights, too). Is there another way to enlarge the receptive field without incurring a large amount of computing?
- The method is to downsample the feature map by using downsampling (eg Strided convolutions)