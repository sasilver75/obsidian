Sebastian Raschka
https://youtu.be/nf-r9gnum7o?si=CDR-q3O4ftDTcF5q&t=36

What is Pytorch?
1. A tensor library
	- A tensor is a mathematical object that we can manipulate with linear algebr
	- In our context, it's a data structure that we can use in different linear algebra operations. 
```python
import torch

# a rank-0 tensor (a scalar)
a = torch.tensor(1.)
a.shape
# torch.Size([])

# a rank-1 tensor (a vector)
a = torch.tensor([1., 2., 3.])
a.shape
# torch.Size([3])

# a rank-2 tensor (a matrix)
a = torch.tensor([[1., 2., 3.,],[2., 3., 4.,]])
a.shape
# torch.Size([2,3])

```
1. An automatic differentiation engine
2. 
3. A deep learning library

Why PyTorch?
![[Pasted image 20231203152352.png]]
- You can see that there's a huge growth of PyTorch, and a decline in Tensorflow.

![[Pasted image 20231203152755.png]]
- Above: We can interpret this colored image as a stack of matrices (a stack of rank-2 tensors), otherwise known as a rank-3 tensor.
![[Pasted image 20231203152837.png]]
- Above: We could then interpret a stack of multiple images of cats (each being a rank-3 tensor) as a rank-4 tensor!

This is just highlighting the aspect of what a tensor library is.
- They look pretty similar to array libraries (eg numpy). This is actually by design, so that it's easier to use. But if we already have `np.array`, why do we care about `torch.tensor`? The reason is that torch.tensor has GPU support!

```python
a = torch.tensor(...)
# Cuda is a software library for GPUs by Nvidia; If I had multiple GPUs, I'm specifying which GPU I'd like to move it to!
a.to('cuda:0')
```

- On top of the GPU support, it also has ==autodiff support== (autodifferentiation).
	- This is a concept for doing the heavy mathetmatical work for us automatically! We want to compute the derivatives/gradients of our functions.

![[Pasted image 20231203153147.png]]
- Above: This is an abstract drawing of a logistic regression. Sigmoid(wx+b).
![[Pasted image 20231203153232.png]]- Above: Here's a *computation graph* of the same concept.




















