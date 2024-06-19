https://www.learnpytorch.io/00_pytorch_fundamentals/
(This is a pre-PyTorch 2.0 tutorial, but has some 2.0 content at the end.)
This is from a self-taught dude who I don't think has super strong theory fundamentals, so it's more of a "how to use PyTorch" tutorial than anything else.

PyTorch is the most used deep learning framework, and helps take care of many things like GPU acceleration behind the scenes.

We'll cover:
- Intro to tensors
- Creating tensors
- Getting information from tensors
- Manipulating tensors
- Dealing with tensor shapes
- Indexing on tensors
- Mixing PyTorch tensors and NumPy
- Reproducibility
- Running tensors on GPU

```python
import torch

# A scalar is a single number, and in tensor-speak it's a zero-dimension tensor
scalar = torch.tensor(7)
scalar.ndim  # 0

# How do we retrieve the number from the scalar tensor? We use .item()
scalar.item()  # 7

# A vector is a single-dimsional tensor, but can contain many numbers, eg [3,2]
vector = torch.tensor([7,7])
vector.ndim  # 1 ; You can tell the # of dimensions by counting the # of square brackets on a side
vector.shape  # torch.Size([2])

# Let's see a matrix
matrix = torch.tensor([[7,8],[9,10]])
matrix.ndim  # 2
matrix.shape  # torch.Size([2,2])  ; "two by two"

# Let's see a tensor ((This isn't a good example imo because one of its dimensions is 1))
tensor = torch.tensor([[[1,2,3], [3,6,9], [2,4,5]]])
tensor.ndim  # 3
tensor.shape  # torch.Shape([1,3,3])
```

In PyTorch, dimensions go from "outer to inner", so there's 1 dimension of 3x3 in our tensor.
![[Pasted image 20240618224658.png|400]]

Machine learning models often start with large random tensors of numbers, and then adjusts these numbers as it works through data to better fit our training data.
Let's create a tensor of random numbers using `torch.rand()`, which numbers ~ standard normal $N(0,1)$.

```python
random_tensor = torch.rand(size=(3,4))  # I think this is randn in 2.0
random_tensor.dtype  # torch.float32

# Say we wanted random image tensor in the common image shape of [224,224,3], like an image with three channels?
random_image_tensor = torch.rand(size=(224,224,3))

# Sometimes we want to fill a tensor with just zeroes or ones; this happens a lot with masking
zeroes = torch.zeros(size=(3,4))  # a 3x4 tensor of 0.'s with datatype torch.float32
ones = torch.ones(size=(3,4))  # a 3x4 tensor of 1.'s with datatype torch.float32

# Sometimes you know that there's ar ange of numbers, like 1 to 10 or 0 to 100
# We can use torch.arange(start, end, step) to do so! (arange is short for arrayrange)
zero_to_ten_deprecated = torch.range(0,10)
# The correct way is below
zero_to_ten = torch.arange(0,10)
odds = torch.arange(1,11,2)

# Sometimes you might want a tensor with a certain shape as some other tensor that you have on-hand
# We can use the zeros_like(input) and ones_like(input) to create tensors filled with zeroes or ones in the same shape as the input argument.
ten_zeroes = torch.zeros_like(zero_to_ten)  # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# There are many different tensor datatypes available in PyTorch, with some specific for CPU, and others better for GPUs. Generally, when you see torch.cuda anywhere, it's a good indication that the tensor is meant for GPU usage.
# The most common type (And generally hte default) is torch.float32 / torch.float, but there's also torch.float16 or torch.half, as well as 64-bit floating points torch.float64 or torch.double.
# In addition, there are also 8-bit 116-bit, 32-bit, and 64-bit integers, and more!
# Lower-precision datatypes are generally faster to compute, but less accurate.
float_32_tensor = torch.tensor(
	[3.0, 6.0, 9.0],
	dtype=None  # Defaults the None, which is torch.float32 
	device=None  # Defaults to None, which uses the default tensor type ((?))
	requires_grad=False  # If True, operations performed on the tensor are recorded
)
float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device
# (torch.Size([3]), torch.float32, device(type='cpu'))

# Aside from shape mismatch issues, one of the most common PyTorch issues you'll come across are datatype and dedvice issues. For example, if one of your tensors is float32 and the other is float16.
# Another issue is when one tensor is on the CPU and the other is on the GPU

# Let's make a float16-precision tensor
float_16_tensor = torch.tensor([1,2,3], dtype=torch.float16)  # torch.half would also work here

# So how do we get soem information from our tensors? We've alrady see .shape, .dtype, and .device, but what else can we do?
some_Tensor = torch.rand(3,4) # a 3x4 matrix tensor of unit-normally-distributed float32 values, stored on cpu by default

# We can use addition, subtraction, element-wise multiplication, division, and matrix multiplication on our tensors (among others)
tensor = torch.tensor([1,2,3])
# If we add a scalar x to a tensor, its value is broadcast among the tensor elements, and every element becomes elt+x
tensor + 10 # tensor([11,12,13])  ; Note this doesn't modify the original tensor

# Multplication behaves the same way
tensor * 10  # tensor([10,20,30])

# And subtraction behaves the same way
tensor - 10  # tensor([-9, -8, -7])

# PyTorch has a bunch of built-in functions like torch.mul() and torch.add() to perform basic operations, if you want to use that, but it's more common to just use the built-in Python operator symbols.
torch.multiply(tensor, 10)  # tensor([10,20,30])

# We can do element-wise multiplications of two tensors using the * operator too
print(tensor * tensor)  # tensor([1,4,9]), after multiplying two [1,2,3] tensors
# (This would be useful for a dot product, followed by a sum, it seems to me.)

# Matrix multiplication is pretty important! You can do it functionally using the torch.matmul() function, or by using the python @ operator
# Note that the inner dimensions of the two matrices must match for them to by multipliable, and the reuslting matrix has the shape of the outer dimensions.
t = torch.tensor([1,2,3]) 
t @ t # 14  # (1*1 + 2*2 + 3*3) ; matmul
t * t  # [1, 4, 9] ; elemntwise-multiplication

# One of the most common errors in deep learning is dimension mismatch. Shapes need to be oriented correctly?
tensor_A = torch.tensor([
						 [1,2]
						 [3,4],
						 [5,6],
						 dtype=torch.float32
])
tensor_B = torch.tensor([
						 [7,10]
						 [8,11],
						 [9,12],
						 dtype=torch.float32
])

torch.matmnul(tensor_A, tensor_B)  # this will error, because we're tryin to do (3x2)@(3x2), and our inner dimensions of our matrices don't match!
# One of the ways to make hte inner dimensions match is to do the transpose operation, which "switches" the dimensions of a given tensor; we can just access the .T attribute to get this.
torch.matmul(tensor_A, tensor_B.T)

# We'll see this later, but the torch.nn.Linear() module implement a matrix multiplication between any input x, and a weights matrix A! (I think it can also include a bias, so it's y = x(A^T) + b)
# This is a lienar function like y = mx + b that you saw in high school.
linear = torch.nn.Linear(in_features=2, out_features=6)

# We can also find the min, max, mean, sum, etc. of a tensor (various aggregations)
x = torch.arange(0, 100, 10)  # tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

print(f"Minimum: {x.min()}")  # 0
print(f"Maximum: {x.max()}")  # 90
# print(f"Mean: {x.mean()}") # this will error, because these are int64s; must be floats
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype  # 45.0
print(f"Sum: {x.sum()}") # 450

# You could have also used any of torch.nax, torch.min, etc.

# We can find the index of where the minimum or maximum value occurs in a tensor using torch.argmax() and torch.argmin(), respectively! 
# This is useful in certain situations like when using softmax activation functions!
tensor = torch.arange(10, 100, 10)  # tensor([10, 20, 30, 40, 50, 60, 70, 80, 90])
print(tensor.argmax())  # 8
print(tensor.argmin())  # 0

# Changing tensor datatypes (eg so that they match and you can do operations on them) is a useful ability -- you can change them using torch.Tensor.type(ddtype=None)
tensor = torch.arange(10., 100., 10.)
tensor.dtype  # torch.float.32
tensor_float16 = tensor.type(torch.float16)  # This doesn't change the original tensor object

# Creating an int8 tensor
tensor_int8 = tensor.type(torch.int8)
tensor_int8

# Reshaping, stacking, squeezing, and unsqueezing
# Some comon methods to reshape or change the dimensions of our tensors without actually changing the values in them include:

# torch.reshape(input, shape) - Reshapes input to shape if compatible
# Tensor.view(shape) - Returns a view of the original tensor in a different shape, but shares the same data as the original tensor
# torch.stack(tensors, dim=0) - Concatenates a sequence of tensors along a specified dimension -- all tensors must be the same size (along all dimensions, it seems)
# torch.squeeze(input) - Squeezes input to remove all dimensions with value 1
# torch.unsqueeze(input, dim) - Return input with a dimension value of 1 added at the specified dimension
# torch.permut(input, dims) - Returns a view of input with its dimensions permuted (rearranged) to dims

# Let's try some out!
x = torch.arange(1., 8.)  # tensor([1., 2., 3., 4., 5., 6., 7.])
x.shape  # torch.Size([7])

# Let's add a dimension using torch.reshape()
x_reshaped = x.reshape(1,7)
x_reshaped, x_reshaped.shape  # (tensor([[1., 2., 3., 4., 5., 6., 7.]]), torch.Size([1, 7]))

# Let's change the view of our tensor with torch.view
# Note that torch.view merely creates a VIEW of the original tensor; it will always share its data with the original tensor, so if you change one of them, the other will change too.
z = x.view(1,7)
z, z.shape  # (tensor([[1., 2., 3., 4., 5., 6., 7.]]), torch.Size([1, 7]))  ; Note that this appears to be the same as the results from reshape, but reshape actually resahpes and returns a reference to a new object. 

# If we want to stack our new tensor on top of itself four times, we can do so with torch.stack!




```



# 0. PyTorch Fundamentals
- 


# 1. PyTorch Workflow Fundamentals


# 2. PyTorch Neural Network Classification


# 3. PyTorch Computer Vision



# 4. PyTorch Custom Datasets


# 5. PyTorch Going Modular


# 6. PyTorch Transfer Learning


# 7. PyTorch Experiment Tracking



# 8. PyTorch Paper Replicating


# 9. PyTorch Model Deployment



# 10. PyTorch 2.0


# 11. PyTorch Extra Resources


# 12. PyTorch Cheatsheet


# 13. The Three Most Common Errors in PyTorch

