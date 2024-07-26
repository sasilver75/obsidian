Partitions the model vertically into stages by layers, so that different devices can process in parallel different stages of the full model pipeline. 

![[Pasted image 20240415165213.png]]
![[Pasted image 20240415165243.png]]

Key: You want to try to keep all GPUs busy (have maximum utilization) -- so often you have complex interleaving of forward (blue) and backward passes (green):
![[Pasted image 20240415165600.png]]
Above: See that we've already starting the backward while we're still completing the forward pass on other GPUs.
While for Tensor Parallelism you had to rewrite the model code, here you also have to rewrite the optimization code. This makes using the code quite complex -- HF has a `nanotron` library that tries to make this as simple as possible.


![[Pasted image 20240725105447.png]]
From [[LLaMA 3.1]]