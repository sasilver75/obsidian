2018
See previous 2017 work [[MobileNet]], which it improved on by introducing two additional innovations:
- Linear Bottlenecks
- Inverted Residuals


![[Pasted image 20240701110041.png]]
![[Pasted image 20240701110108.png]]
If you were to use ReLU here, you would zero out the negative values coming out the dimension reduction step, which would cause the network to lose valuable information.


![[Pasted image 20240701110208.png]]
![[Pasted image 20240701110218.png]]
Instead of connecting the layers with the highest number of channels (because intuitively, that's where the maximum amount of information lies, right?), instead the authors assert that the bottleneck layers capture the essential information in lower-dimensional subspaces. 
So authors add shortcuts between *these* bottleneck layers:
![[Pasted image 20240701110325.png]]


![[Pasted image 20240701110330.png]]

