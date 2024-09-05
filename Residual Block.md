
See: [[Residual Network]], [[Residual Connection]]



![[Pasted image 20240701105108.png]]
In residual learning, the input passes through 1 or more Convolutional layers as usual, but at the end, the original input is added back to the final output.
Called Residual Blocks because they don't actually need to learn the final output feature maps in the traditional sense, but instead learn just the *residual features* that need to be *added* to the input to get the final feature map.
![[Pasted image 20240701105254.png]]
During backpropagation, the gradients can flow back through these residual residual connections to reach the earlier layers of the networks fast, without much vanishing (letting us create deeper networks). This allowed ResNet to train a 152-layer model, shattering existing records.

![[Pasted image 20240701105352.png]]


