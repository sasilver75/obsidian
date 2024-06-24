https://www.learnpytorch.io/00_pytorch_fundamentals/
(This is a pre-PyTorch 2.0 tutorial, but has some 2.0 content at the end.)
This is from a self-taught dude who I don't think has super strong theory fundamentals, so it's more of a "how to use PyTorch" tutorial than anything else.


# 0. PyTorch Fundamentals

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
x_stacked = torch.stack([x,x,x,x], dim=0)  # dim=0 means stack along the rows, i.e. vertically
x_stacked
tensor([[5., 2., 3., 4., 5., 6., 7.],
        [5., 2., 3., 4., 5., 6., 7.],
        [5., 2., 3., 4., 5., 6., 7.],
        [5., 2., 3., 4., 5., 6., 7.]])

# What about removing all single dimensions from a tensor? We can do this using torch.squeeze()
x_reshaped, x_reshaped.size  # tensor([[5., 2., 3., 4., 5., 6., 7.]]), torch.Size([1,7])
x_squeezed= x_reshaped.squeeze()
x_squeezed, x_squeezed.size  # tensor([5.,2.,3.,4.,5.,6.,7.]), torch.Size([7])

# And to do the reverse, we can use torch.unsqueeze to *add* a dimension valeu of 1 at a specific index
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
x_unsqueezed, x_unsqueezed.size # tensor([[5., 2., 3., 4., 5., 6., 7.]]), torch.Size([1,7])

# and we can rearrange the order of axes values using torch.permute(input, dims), where the input gets turned into a _view_ with the new dims
x_original = torch.rand(size=(224,224,3))
# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2,0,1) # Shifts axis 0->1, 1->2, 2->0
# NOTE that permuting only returns a _view_, and if you change values in the permuted tensor, the original tensor's values will change too, and vice versa.

# Indexing (selecting data from tensors)
# Sometimes you want to select specific data from tensors (eg the first column or second row); to do so, you can use indexing.
x = torch.arange(1,10).reshape(1,3,3)  # Clever way to reshape an arange into a tensor!
x, x.shape
(tensor([[[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]]),
 torch.Size([1, 3, 3]))

# Indexing values goes outer dimension -> inner dimension (check the square brackets)
# Let's index bracket by bracket
print(f"First square bracket:\n{x[0]}") 
print(f"Second square bracket: {x[0][0]}") 
print(f"Third square bracket: {x[0][0][0]}")
First square bracket:
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
Second square bracket: tensor([1, 2, 3])
Third square bracket: 1

# You can also use : to specify "All values in this dimension", and then use a comma to add another dimension
x[:, 0]  # Get the first column of all rows
tensor([[1,2,3]])  # See that it's still a 2-dim tensor

x[:,:, 1]  # Get all values of the 0th and 1st dimensions, but only index 1 of 2nd dim
tensor([[2, 5, 8]])

x[0,0,:]  # This would be the same as x[0][0]
tensor([1,2,3])  # Note the dimensionality of this one, compared to the previous two; it's because we specified single indices for two of the dimensions, leaving us with one.

# Since NumPy is such a popular Python numerical computing library, PyTorch has functionality to interact with it nicely!
# Two main methods you'll want to use for NumPy to PyTorch (and back again) are:
# torch.from_numpy(ndarray)
# torch.Tensor.numpy()
# ~~ Examples omitted because obvious ~~

# Reproducibility is something that's important in machine learning, like when you're using torch.rand to reate tensors with random floats
# We can use torch.manual_seed(seed), where seed is an integer like 42 that flavors the (pseudo)randomness
import random
RANDOM_SEED=42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3,4)

# You have to reset the seed every time a new rand() is called
# Without doing this, tensor_D would be _different_ to tensor_C!
torch.manual)seed(seed=RANDOM_SEED)
rnadom_tensor_D = torch.random(3,4)

random_tensor_C == random_tensor_D  # True! This was because we (re)set the random seed to the same number before each torch.rand call

# to get PyTorch to use your GPU, you can check:
torch.cuda.is_available()  # True ; nice!
device = "cuda" if torch.cuda.is_available() else "cpu"

# You can even count th numbe rof GPUs PyTorch has access to using torch.cuda.device_count()
torch.cuda.device_count()  # 1

# To put a tensor (or model) on the GPU, we can use to(device) on them. GPUs are far faster at numerical computing than CPUs!
tensor = torch.tensor([1,2,3])
tensor_on_gpu = tensor.to(device)  # recall device="cuda"
tensor_on_gpu  # tensor([1, 2, 3], device='cuda:0')  <-- This means it's stored on the -th GPU avialable.

# To move back to a CPU (eg if we want to interact iwth our tensors using NumPy, which doesn'et leverage hte GPU), we can use .cpu()
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()  # array([1,2,3])
```


# 1. PyTorch Workflow Fundamentals
We've going to cover a standard PyTorch workflow in this section:
![[Pasted image 20240619121604.png|400]]
We'll use this workflow to predict a simple straight line, but the workflow steps can be repeated/changed based on the problem we're working on.

We'll cover:
- Getting data ready
- Building a model
- Fitting model to data
- Making predictions and evaluating a model
- Saving and loading a model
- Putting it all together

Data in ML can be almost anything you can imagine -- a table of excel numbers, images of any kind, videos, audio files, protein structures, text, and more.
Neural networks are universal function approximators, and work well across domains.

Let's creat some data representing a straight line, using linear regression with some known parameters, and then use PyTorch to see if we can estimate these parameters using gradient descent
```python
weight = 0.7
bias = 0.3

# Create some data!
start = 0
end = 1
step = 0.02
# Creating a vector of input data [0, 0.02, 0.04, ... 0.98]
X = torch.arange(start, end, step).unsqueeze(dim=1)  # Turning from torch.Size([5]) to torch.Size([5,1])
y = weight * X + bias  # For each input x, multiply it by weight and add bias (usign broadcast for each)
# Now we have X and y, and we can train a model to predict this known y for any x

# Let's split our data into train and test sets (and when required, a validation set)
train_split = int(.8*len(X))
X_train, y_train = X[:train_split], y[:train_split]  # Length 40
X_test, y_test = X[train_split:], y[train_split:]  # Length 10

# Let's build on LinearRegressionModel in PyTorch
class LinearRegressionModel(nn.Module):
	def __init__(self):
		super().__init__()
		# nn.Parameter (https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)
		# Parameters store tensors that can be used with nn.Module
		# A single parameter that we'll learn
		self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
		self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), required_grad=True)

	def forward(self, x):
		# defines computation in our model
		# y = mx + b ; the linear regression formula!
		return self.weights * x + self.bias
```
![[Pasted image 20240619123321.png|300]]

We can create a model instance and check its parameters using `.parameters()`:
```python
# Because our nn.Parameter are randomly initialized, let's set a random seed
# I'm still unsnure how many times we have to do this, because both our weights and biases are randomly initialized
torch.manual_seed(42)

model_0 = LinearRegressionModel()

# Let's list its parameters
list(model_0.parameters())
[Parameter containing:
 tensor([0.3367], requires_grad=True),
 Parameter containing:
 tensor([0.1288], requires_grad=True)]

# We can also get the state (what the model contains) of the model using .state_dict()
model_0.state_dict()
OrderedDict([('weights', tensor([0.3367])), ('bias', tensor([0.1288]))])

# We want to start from random parameters and get the model to update them towards parameters that fit our data best (our known w and b values from the start)
# torch.inference mode is a new context manager analagous to no_grad. In older versions of PyTorch, you might see torch.no_grad() being used for inference, but this inference_mode context manager is preferred.
# I don't believe it does the model.eval() thing though which (eg) turns off dropout layers, etc.
with torch.inference_mode():
	y_preds = model_0(X_test)
	# Our predictions look pretty bad, using our randomly initialized weights! Let's train

# We know that the "real" values are weight=0.7, and bias=0.3
# We need to create a loss function, optimizer, train loop, and test loop

# Create our loss function
loss_fn = nn.L1Loss()  # MAE Loss is the same as L1 Loss
# Create the SGD optimizer, with a reference to our model parameters
optimizer = torch.optim.SGD(params=model_9.parameters(), lr=0.01)

# Now let's create an optimization loop in PyTorch! I think it's better to create these loops separately, but the tutorial chooses to do it in one big loop

torch.manual_seed(42)

epochs = 100
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
	# TRAIN LOOP
	model_0.train() # Puts the model in training mode

	# 1. Make a forward prediction on our data
	y_pred = model(X_train)

	# 2. Determine our loss with respect to the actual label
	loss = loss_fn(y_pred, y_train)

	# 3. Zero grad of the optimizer (Note the official tutorial loop does backward, step, zerograd; it's just important that you zerograd before doing another backward pass)
	optimizer.zero_grad()

	# 4. Backpropagate
	loss.backward()

	# 5. Step the optimizer
	optimizer.step()

	# TEST LOOP: # Enter test loop (I guess we're doign this every epoch?)
	model_0.eval()  # Put the model in eval mode

	# 6. Make a pass through your test dataset and report data
	# This seems kinda dumb because we calculate per-epoch loss every epoch but only report epoch loss every 10 epochs
	with torch.inference_mode():
		#1. Forward pass on test data
		test_pred = model(X_test)

		#2. Calculate test loss
		## "Predictions come in torch.float datatype, so comparison ned to be done with tensors the same type"
		test_loss = loss_fn(test_pred, y_test.type(torch.float)) 

		# Print results
		if epoch % 10 == 0:
			epoch_count.append(epoch)
			# Tesnor.detach() returns a new tensor, detached frmo the current graph; the result will never require grad
			train_loss_values.append(loss.detach().numpy())
			test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
```

If we plot our epoch loss (every 10 epochs, here), we see that it goes down over time.
![[Pasted image 20240619132854.png|300]]
It's normal to see better performance on your training data than on your test data, since your model is training/optimizing on the training data distribution, which will be slightly different from the test data distribution.

Let's examine our model's `.state_dict()` to see what parameters it's learned.
```python
model_0.state_dict()  # Omitting some print statements

The model learned the following values for weights and bias:
OrderedDict([('weights', tensor([0.5784])), ('bias', tensor([0.3513]))])
And the original values for weights and bias are:
weights: 0.7, bias: 0.3
```
So we can see that our model got pretty close to calculate the exact original values.
- If we change the epochs to be ever higher, we'll see better performance.

There are three things to remember when making predictions (with a PyTorch model):
1. Set the model in evaluation mode (`model.eval()`)
2. Make the prediction using the inference mode context manager (using `with torch.inference_mode(): ...`)
3. All predictions should be made with objects on the same device (eg data and model are both on GPU or CPU)

Let's talk about saving and loading models, now!
- `torch.save`: Saves a serialized object to disk using Python's `pickle` utility. 
- `torch.load`: Uses `pickle`'s unpickling features to deserialize and load pickled Python object files into memory.

```python
from pathlib import Path

# Note that it's common convetnion for PyTorch saved models or objects to end with .pt or .pth

# Create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODLE_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to {MODEL_SAVE_PATH}")
# We only save the state_dict() of the model, containing the model learned parameters
# The disadvantage of saving the WHOLE model is that the serialized data is bound to the specific classes/exact directy structure used when the model is saved... your code can break in various ways when used in other projects.
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

# ... Later ...

# We can load a saved PyTorch mdoel's state_dict() too!
loaded_model_0 = LinearRegressionModel()  # New instance (instantiated with random weights)
# Now, load the state_dict of our saved model, updating the new instance of our model)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# cool, now let's do inference with it!
loaded_model_0.eval()
with torch.inference_mode():
	loaded_model_preds = loaded_model_0(X_test)
```

Now let's create a model the same as before, but instead of creating the weights and bias parameters of our model manually using `nn.Parameter`, we'll use `nn.Linear` to do it for us!
![[Pasted image 20240619134252.png|300]]

```python
class LinearRegressionModelV2(nn.Model):
	def __init__(self):
		super().__init__()
		# User nn.Linear() for creating the model parameters!
		# (I guess this includes both a weight and bias term?)
		self.linear_layeer = nn.Linear(in_features=1, out_features=1)

	def forward(self, x:torch.Tensor) -> torch.Tensor:
		return self.linear_layer(x)

# Not always needed, but useful here for demonstration repeatability purposes
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1.state_dict()
OrderedDict([('linear_layer.weight', tensor([[0.7645]])),
              ('linear_layer.bias', tensor([0.8300]))])
# See that it created a weight and bias for us!

# Now let's put our model on the GPU
next(model_1.parameters()).device  # device(type='cpu')
model_1.to('cuda')
next(model_1.parameters()).device  # device(type='cuda', index=0)  ; the 0th GPU
```

```python
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

# Now we just need to train and evaluate our model!
# NOTE: Because our model is on the GPU, the data needs to be on  the GPU too!
torch.manual_seed(42)
epochs=1000

X_train = X_train.to('cuda')
X_test = X_test.to('cuda')
y_train = y_train.to('cuda')
y_test = y_test.to('cuda')

for epoch in range(epochs):
	# TRAINING LOOP
	model_1.train()  # Put the model intraining mode if it's not already
	preds = mdoel_1(X_train)
	loss = loss_fn(preds, y_train)
	optimizer.zero_grad()  # Zero out hte optimizer before backpropping
	loss.backward()  # Get gradients
	optimizer.step()  # step

	# TEST LOOP
	model_1.eval()  # Put the model in eval mode
	with torch.inference_mode():  # Don't accumulate gradients, etc
		preds = model_1(X_test)
		loss = loss_fn(preds, y_test)

	# It's dumb that we're calculating test statistics every epoch, but only doing anything with it every 10 epochs.
	if epoch % 100 == 0:
		print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

# now, what if we check our learned parametrs?
model_1.state_dict()
OrderedDict([('linear_layer.weight', tensor([[0.6968]], device='cuda:0')),
             ('linear_layer.bias', tensor([0.3025], device='cuda:0'))])
# These are very close to our w=.7 and b=.3! If we trained for longer, we'd get even better resutls.

# Let's save our model
MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME
# Now, let's save our model's state dict to the specific save path
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)  

# LAter, we can load it!
new_model = LinearRegressionV2()
new_model.load_state_Dict(torch.load(MODEL_SAVE_PATH)) # load it
new_model.to('cuda') # put that bitch on the GPU

```



# 2. PyTorch Neural Network Classification
We're going to cover the full workflow of doing a NN classification exercise, covering
- Architecture of a classification NN
- Getting binary classification data ready
- Building a PyTorch classification model
- Fitting the model to data
- Making predictions and evaluating a model
- Improving a model
- Introducing nonlinearities
- Replicating non-linear functions

Let's use the make_circles() method from Scikit-Learn to generate two circle with different-coloured dots
```python
from sklearn.datasets import make_circles

n_samples = 1000
# A large circle containing a small circle in 2D, and then a point from the edge of the circles (noised) and its label
# This is just a useful little toy function for creating classification datasets
X, y = make_circles(n_samples, noise=0.03, random_state=42)

import pandas as pd
circles = pd.DataFrame({
	"X1": X[:,0],
	"X2": X[:,1],
	"label": y
})
circles.head(10)
```
![[Pasted image 20240619144330.png|100]]
It looks like each pair of X features (X1, X2) have a label (y) of either 0 or 1.
If we plot them:
![[Pasted image 20240619144401.png|200]]

One of the most common errors in deep learning is shape mismatch errors -- always be asking yourself: "What shape are my inputs, and what shape do I want my outputs to be?"

```python
X.shape, y.shape
((1000, 2), (1000,))
Values for one sample of X: [0.75424625 0.23148074] and the same for y: 1
Shapes for one sample of X: (2,) and the same for y: ()
# This tells us the second dimension for X means it has two features (vector), whereas y has a single feature (scalar)

# We need to turn our NumPy data into PyTorch tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)  # torch.float == torch.float32

# Split our data into train and test set splits, using a handly sklearn helper utility
from skelearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=.2,
	random_state=42
)

# Let's build our model
import torch
from torch import nn

device = "cuda"

def CircleModel(nn.Module):
	def __init__(self):
		super().__init__()
		# We're going to have 5 hidden units. This is a hyperparameter of your model; the amount you choose depends on the model type and dataset you're working with
		self.layer_1 = nn.Linear(in_features=2, out_features=5)
		self.layer_2 = nn.Linear(in_features=5, out_features=1)

	def forward(self, x):
		return self.layer_2(self.layer_1(x))
		
model = CircleModelV0().to(device)
model_0
CircleModelV0(
  (layer_1): Linear(in_features=2, out_features=5, bias=True)  # See that we have bias terms too, cool!
  (layer_2): Linear(in_features=5, out_features=1, bias=True)
)

# Note that we could have replicated our Model using nn.Sequential
model_0 = nn.Sequential(  # Chains together *arg layer outputs to inputs
	nn.Linear(in_features=2, out_features=5),
	nn.Linear(in_features=5, out_features=1)
)
# nn.Sequential is fantastic for straight forward computations, but hte it ALWAYS runs in sequential order
# So if you want something else to happen (rather than just straight-forward sequential computation), you should define your own custom nn.Module subclass and your own forward method.

# Let's make some predictions!
untrained_preds = model_0(X_test.to('cuda'))  # Recall that our model on the GPU needs data on the GPU
# These are shitty predictions because it's from our randomly initialized model

# We've setup a loss and optimizer before, but different problems require different loss functions.
# For ours, we'll use Binary Cross Entropy as the loss function.
# Pytorch has two BCE implementations:
	# torch.nn.BCELoss(): Loss function measuring BCE between label and prediction
	# torch.nn.BCEWithLogitsLoss(): Same as above but it has a sigmoid layer (nn.Sigmoid) built in!
loss_fn = nn.BCEWithLogitsLoss() # BCE with sigmoid built-in
optimizer = torch.optim.SGD(parmas=model_0.parameters(), lr=0.1)

# Let's also create an EVALUATION METRIC, which is soemthing used to get another perpsective on how our model is training (that isn't our loss function). This mettric is what we ACTUALLY care about.) We'll use accuracy
def accuracy_fn(y_true, y_pred):  # Given the truue labels and predicted labels
	correct = torch.eq(y_true, y_pred).sum().item()
	acc = (correct / len(y_pred)) * 100
	return acc

# Now let's train our model!
y_logits = model_0(X_test.to(device))[:5]
# Since we haven' trained this model, these outputs are basically random.
# The RAW OUTPuts of our y = mx + b equation are often referred to as logits, but these numbers are hard to interpret. To get our modle's raw outputs (logits) into such a form, we can use the sigmoid acitivation function!
# Note that sigmoid is only for binary classification logits; for multiclass classification, we'll be using the softmax activation function later on
y_pred_probs = torch.sigmoid(y_logits)
# Now, the outputs are all positive and sum to 1
# Because we're doing classification, our ideal outputs are either 0 or 1. So these values can be viewed as a decision boundary!1 The closer to 0, the more the model thinks the sample belongs to class 0, and the closer to 1, the more the model thinks the sample belongs to class 1.
# So we can just say if y_pred_probs >= .5, yhat=1 , and if <= .5, yhat=0

# torch.round rounds element of the input to the nearest integer
y_preds = torch.round(y_pred_probs)
# Map the first 5 records of our test data to our gpu, do a forward pass, turn into probabilities, and round to 1/0.
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# Now our model preditions are of the same format as our labels, which is nice, since we can now compare them to our labels.
# Let's train now
torch.manual.seed(42)

epochs = 100

# Get our data on gpu
X_train, X_test = X_train.to('cuda'), X_test.to("cuda")
y_train, y_test = y_train.to('cuda'), y_test.to('cuda')

# Now let's do our epochs
for epoch in range(epochs):
	# Training Loop
	model.train()
	y_logits = model(X_train).squeeze()  # Squeeze to remove extra '1' dimension
	y_pred = torch.round(torch.sigmoid(y_logits))  # Trun logits into predictions of 0 or 1
	loss = loss_fn(y_pred, y_train)
	acc = accuracy_fn(y_train, y_pred)  # training accuracy
	optimizer.zero_grad()
	loss.backard()
	optimizer.step()
	
	# Evaluation Loop
	model_0.eval()
	with torch.inference_mode():
		test_logits = model_0(X_test).squeeze()
		test_preds = torch.round(torch.sigmoid(test_logits))

		# Test loss and acc
		test_loss = loss_fn(test_preds, y_test)
		test_acc = accuracy_fn(y_test, test_preds)

		# Every 10 epochs, print out what's happening (this is stupid to calc every epoch)
		if epoch % 10 == 0:
	        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# What do we notice about the perofmrance of our model?
# It looks like it completed the training epochs successfully, but the results don't seem to improve too much -- the accuracy barely moves above 50% on each data split...

# If we visualize our decisioin boundary, we'll see that it's just a straight line that bisects our circle!
```
![[Pasted image 20240619154047.png|200]]

It seems like we've drawing a straight line between these two classes, which explains the 50% accuracy.
- We're underfitting our data, with errors due to bias. We need to have a more flexible model by introducing nonlinearities.
- We can add as many nn.Linear layers as we want, but stacking linear transformations will never make a nonlinear transformation.

PyTorch has a bunch of ready-made nonlinear activation functions that do similar but different things -- one of the most common and best performing is ReLU, which is available at torch.nn.ReLU()

```python
class CircleModelV2(nn.Module)
	def __init__(self):
		super().__init__()
		self.linear_1 = nn.Linear(in_features=2, out_features=10)
		self.linear_2 = nn.Linear(in_features=10, out_features=10)
		self.linear_3 = nn.Linear(in_features=10, out_features=1)
		self.nonlinear = nn.ReLU()  # Note this function doesn't dictate order of execution of layers

	def forward(self, x):
		# Note that the ReLU acitvation functions is interpsersed between layers
		return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2().to('cuda')

# Setup loss and optimizer 
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)

# Fit the model
torch.manual_seed(42)
epochs = 1000

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    # 1. Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
    
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_3.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model_3(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
      # 2. Calcuate loss and accuracy
      test_loss = loss_fn(test_logits, y_test)
      test_acc = accuracy_fn(y_true=y_test,
                             y_pred=test_pred)

    # Print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

# Now we can evaluate our model
model_3.eval()
with torch.inference_mode():
	y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

# If we graph these...
```
![[Pasted image 20240619155227.png|200]]
See the improvement!

Let's try a multi-class classification problem. Scikit-Learn has a useful make_blobs() method, which will create however many classes we want!
```python
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUUM_FEATURES = 2
RANDOM_SEED = 42

# Create some multi-class data
# make_blobs: Generates oem isotropic Gaussian blobs for clustering
X_blob, y_blob = make_blobs(
	n_samples=1000, 
	n_features=NUM_FEATURES,  # 2
	centers=NUM_CLASSES  # 4
	cluster_std=1.5,  # Giving the clusters a little shakeup
	random_state=RANDOM_SEED
)

# Turn the numpy data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)  # 64-bit integer

# Split into train and test using handy train_test_split method from sklearn
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
	X_blob,
	y_blob,
	test_size=.2,
	random_state=RANDOM_SEED
)

# Plot data
plt.figure(figsize=(10,7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmpa=plt.cm.RdYlBu)
```
![[Pasted image 20240619160158.png|200]]
Cool, we've got some multi-class classification data ready to rip.

Let's build a multiclass model that takes some parametersL
```python
device = "cuda" if torch.cuda.is_available() else "cpu"

class BlobModel(nn.Module):
	def __init__(self, input_features, output_features, hidden_units=8):
		super().__init__()
		self.linear_layer_stack = nn.Sequential(
			nn.Linear(in_features=input_features, out_features=hidden_units),
			# nn.ReLU(), # Unclear: Does our model need nonlinearities?
			nn.Linear(in_features=hidden_units, out_features=hidden_units),
			# nn.ReLU(),
			nn.Linear(in_features=hidde_units, out_features=output_features)
		)

	def forward(self, x):
		return self.linear_layer_stack(x)

model_4 = BlobModel(NUM_FEATURES, NUM_CLASSES, 8).to(device)

# Let's create an optimizer nad loss function
# for multi-class clasification, we'll use nn.CrossEntropyLoss() as our loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.optim,.SGD(mode.parameters(), lr=.1)

# Let's make some untrained shitty predictions
model_4(X_blob_train.to(device))[:5]
tensor([[-1.2711, -0.6494, -1.4740, -0.7044],
        [ 0.2210, -1.5439,  0.0420,  1.1531],
        [ 2.8698,  0.9143,  3.3169,  1.4027],
        [ 1.9576,  0.3125,  2.2244,  1.1324],
        [ 0.5458, -1.2381,  0.4441,  1.1804]], device='cuda:0',
       grad_fn=<SliceBackward0>)
# Note that these aren't probabilities. These are logits.
# For multiclass classification, we can turn these into probabilities using teh softmax activation function!
# This is available using torch.softmax

y_logits = model_4(X_blob_test.to(device))
y_pred_probs = torch.softmax(y_logits, dim=1)
y_pred_probs
tensor([[0.1872, 0.2918, 0.1495, 0.3715],
        [0.2824, 0.0149, 0.2881, 0.4147],
        [0.3380, 0.0778, 0.4854, 0.0989],
        [0.2118, 0.3246, 0.1889, 0.2748],
        [0.1945, 0.0598, 0.1506, 0.5951]], device='cuda:0',
       grad_fn=<SliceBackward0>)
# See that these now are all positive and add to one? Nice.
# Now if we're just selecting a single one of them, maybe we select the class with the highest probability (eg using argmax)
print(y_pred_probs[0])
print(torch.argmax(y_pred_probs[0]))
tensor([0.1872, 0.2918, 0.1495, 0.3715], device='cuda:0',
       grad_fn=<SelectBackward0>)
tensor(3, device='cuda:0')  # See that we chose the class at index 3?


# Let's make a training loop!

# Fit the model
torch.manual_seed(42)
epochs = 100

X_train, X_test = X_blob_train.to(device), X_blob_test.to(device)
y_train, y_test = y_blob_train.to(device), y_blob_train.to(device)

for epoch in range(epochs):
	# Training
	model_4.train()  # Train mode
	y_logits = model_4(X_blob_train)  # Get Logits
	y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # Make class prediction by softmaxing + getting max probability
	
	loss = loss_fn(y_pred, y_blob_train)
	acc = accuracy_fn(y_blob_train, y_pred)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	# Evaluation
	model.eval()
	with torch.inference_mode():
		test_logits = model_4(X_blob_test)
		test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1) 
		test_loss = loss_fn(test_logits, y_blob_test)  # Calculate Cross Entropy loss on the logits
		test_acc = accuracy_fn(y_blob_test, test_pred)  # Calculate accuracy metric on the class predictions

	if epoch % 10 == 100:
		# (Report Loss and Accuracy)
```
![[Pasted image 20240619162934.png|200]]

Other classification metrics we could consideR:
- Accuracy: `torchmetrics.Precision()` or `sklearn.metrics.precision_score()`
- Precision: `torchmetrics.Recall()` or `sklearn.metrics.recall_score()`
- Recall: `torchmetrics.F1Score()` or `sklearn.metrics.f1_score()`
- F1-Score: `torchmetrics.F1Score()` or `sklearn.metrics.f1_score()`
- ConfusionMatrix: `torchmetrics.ConfusionMatrix` or `sklearn.metrics.plot_confusion_matrix()`
- Classification Report (includes many of the ones above): `sklearn.metrics.classification_report()`


# 3. PyTorch Computer Vision
Let's talk about some modules to be aware of
- `torchvision`: Contains datasets/models architectures/image transforms used for CV problems
- `torchvision.datasets`: Contains many datasets for image classification, object detection, image captioning, video classification, and more.
- `torchvision.models`: Contains well-performing CV model architectures
- `torchvision.transforms`: Contains commonly-used image transformations that can process/augment images 
- `torch.utils.data.Dataset`: Base dataset class for PyTorch
- `torc.utils.data.DataLoader`: Creates a Python iterable over a dataset (does batching, shuffling, etc.)

We're going to use FashionMNIST
```python
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Getting Training Dataset
train_data = datasets.FashionMNIST(
	root="data",  # Where to download data to
	train=True, # Get training data
	download=True,  # Download if it doesn't already exist on disk
	transform=ToTensor(),  # feature transform; Images come in PIL format; we want to turn to Torch tensors
	target_transform=None  # no label transform
)
test_data = datasets.FashionMNIST(
	root="data",
	train=False,  # Test data
	download=True,
	transform=ToTensor()
)

image,label = train_data[0]
image.shape
# torch.Size([1, 28, 28])  # It's a graysaclae 28x28 image with 1 color channel
# The order of this tensor is often referred to as CHW (Channnels, Height, Width)
	# There's debate on whether images should be represented as CHW or HWC (channels last)

len(train_data.data), len(train_data.targets), len(test_data.data), len(test_data.targets)
# (60000, 60000, 10000, 10000)

# But what classes do these labels represent?
class_names = train_data.classes
class_names
['T-shirt/top',
 'Trouser',
 'Pullover',
 'Dress',
 'Coat',
 'Sandal',
 'Shirt',
 'Sneaker',
 'Bag',
 'Ankle boot']

# so we have 10 classes

# Now that we have our dataset ready to go, let's prepare it with a torch.utils.data.DataLoader, which helps load the data into our model in batches.
from torch.utils.data import DataLoader

bs = 32

train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True)  # Shuffle data at every epoch
test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=False) # Don't necessarily need to shuffle the test data

# We can see what sort of batch will be yielded by our train_dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape, train_labels_batch.shape
# (torch.Size([32, 1, 28, 28]), torch.Size([32]))
# Above: [bs, channels, rows, cols]

# Let's build a baseline model consisting of two nn.Linear layers!
flatten_model = nn.Flatten()

x = train_features_batch[0]  # Just a single sample
output = flatten_model(x)  # Turns from [1,28,28] to [1,784]

class FashionMNISTModelV0(nn.Module):
	def __init__(self, input_shape, hidden_units, output_shape):
		super().__init__()
		self.layer_stack = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features=input_shape, out_features=hidden_units),  # I hate that he did this...
			nn.Linear(in_Features=hidden_units, out_features=output_shapes)
		)

	def forward(self, x):
		return self.layer_stack(x)

# Now that we've got a baseline model class, let's instantiate a model!

torch.manual_seed(42)

# Choice of 10 as a hidden unit size is arbitrary here
model_0 = FashionMNISTModelV0(input_shape=784, hidden_units=10, output_shape=len(class_names))  # One for evey class
model_0.to("cpu")  # Let's keep it on the CPU to begin with ((?))

# Let's set up our loss, optimizer, and evaluation metrics
loss_fn = nn.CrossEntropyLoss()  # Useful for multi-class classification
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# Let's also create a function to time our experiments (to see how long it takes on CPU vs GPU)
from timeit import default_timer as timer

def print_train_time(start: float, end:float, device:torch.device = None):
	total_time = end - start
	print(f"TRain time on {device}: {total_time}:.3f seconds")
	return total_time

# Let's make a train loop, now
from tqdm.auto import tqdm  # This is a library that helps us get a progress bar for our trianing

torch.manual_seed(42)
train_time_start_on_cpu = timer()  # Start timer

epochs = 3

for epoch in tqdm(range(epochs)):
	print(f"Epoch: {epoch}\n----)
	# Train Loop
	train_loss = 0
	for batch, (X,y) in enumerate(train_dataloader):
		model_0.train()  # Put the model in trianing model (eg dropout on, batchnorm on)
		y_pred = model_0(X)
		
		loss = loss_fn(y_pred, y)
		train_loss += loss  # Accumulate the loss across batches in an epoch

		optimizer.zero_grad()  # Zero the grads before doing a backward pass and determining gradients
		loss.backward()
		optimizer.step()

	train_loss /= len(train_dataloader)  # Turn the accumulated train loss into average loss per batch
		
	# Eval Loop
	test_loss, test_acc = 0, 0  # I hate that he's calling it test_acc right now even though the variable functions as a count for a while until dividng by batch
	model_0.eval()  # Put in eval mode (dropout layers off, etc)
	with torch.inference_mode():  # Don't accumulate gradients, etc
		for X, y in test_dataloader:
			test_pred = model(X)
			test_loss += loss_fn(test_pred, y)
			test_acc += accuracy_fn(y, test_pred.argmax(dim=1))
		test_loss /= len(test_dataloader)  # Turn into an average loss per batch
		test_acc /= len(test_dataloader)  # Turn into an average accuracy per batch

train_trian_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, 
                                           end=train_time_end_on_cpu,
                                           device=str(next(model_0.parameters()).device))

# Results
Train loss: 0.45503 | Test loss: 0.47664, Test acc: 83.43%
Train time on cpu: 32.349 seconds
```

Since w'ere going to be building a few models, it's nice to write some code to evaluate them all in similar ways
```python
torch.manual_seed(42)

def eval_model(model, data_loader, loss_fn, accuracy_fn):
	"""Returns a dictionary containing results of model predicting on data_loader"""
	loss, acc = 0, 0
	model.eval()
	with torch.inference_mode():
		for X, y in data_loader:
			preds = model(X)
			loss += loss_fn(preds, y)  # Loss cares about logits
			acc += accuracy_fn(y, preds.argmax(dim=1))  # Acc cares about predicted class, hence argmax. Not sure why we aren't using some softmax to turn our many logits into a probability distribution before selecting, though.
	loss /= len(data_loader)
	acc /= len(data_loader)

	return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

# Now we have a handy funciton to compare baseline model results on our test data to other models later on!
```

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

# Let's build another model that uses non-linear activation functions too!
class FashionMNISTMoelV1(nn.Module):
	def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
		super().__init__()
		self.layer_stack = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features=input_shape, out_features=hidden_units),
			nn.ReLU(),
			nn.Linear(in_features=hiden_units, out_features=output_shape),
			nn.ReLU()  # I've never seen ReLUs not sandwiched by linear layers
		)

	def forward(self, x: torch.Tensor):
		return self.layer_stack(x)		
```

Let's instantiate with the same settings we used before and then use our nice evaluation function, after we've trained it.

```python
torch.manual_seed(42)

model_1 = FashionMNISTModelV1(input_shape,784, hidden_units=10, output_shape=len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.optim.SGD(params=model_1.parameters(), lr=.1)

# Let's write a training loop, remembering to also put our features and labels on the same device as our model
def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device):
	train_loss, train_acc = 0, 0
	model.to(device)
	for batch, (X, y) in enumerate(data_loader):
		# Send the batch of data to the GPU
		X, y = X.to(device), y.to(device)

		y_pred = model(X)
		
		train_loss += loss_fn(y_pred, y)
		train_acc += accuracy_fn(y, y_pred.argmax(dim=1)) # Go from logits -> pred labels

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	# Calculate loss nad accuracy per BATCH and print out what's happening
	train_loss /= len(data_loader)
	train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader, model, loss_fn, accuracy_fn, device):
	test_loss, test_acc = 0, 0
	model.to(device)
	model.eval()
	with torch.inference_mode():
		for X, y in data_loader:
			X, y = X.to(device), y.to(device)  # Move data to device
			test_preds = model(X)
			test_loss += loss_fn(test_preds, y)
			test_acc += accuracy_fn(y, y_preds.argmax(dim=1))
		test_loss /= len(data_loader)
		test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
```

Now let's put these together
```python
epochs = 3
for epoch in tdqm(range(epochs)):
	print(f"Epoch: {epoch}\n-----")
	train_step(train_dataloader, model_1, loss_fn, optimizer, accuracy_fn)
	test_step(test_dataloader, model_1, loss_fn, accuracy_fn)

# Interesting result observed here is that it took longer to train on GPU than on CPU. One reason is that our dataset nad model are both so small that the benefits of using a GPU are outweighed by the time it actually takes to transfer the data there. There's a small bottleneck between copying data from the CPU mmemory to the GPU memory, so for smaller projects, CPU might even be optimal.

model_1_results = eval_model(model_1, test_dataloader, loss_fn, accuracy_fn)
{'model_name': 'FashionMNISTModelV1',
 'model_loss': 0.6850008964538574,
 'model_acc': 75.01996805111821}

# Interestingly in this case, adding nonlinearities to our model made it perform WORSE than the baseline! Sometimes the thing that you think should work actually doesn't work -- it requires experimentation. 
# It seems that our model is overfitting the data, meaning not generalizing to the test data well.
# We can use a larger dataset, or a less flexible model.

# Let's build a convolutional network; we'll use a TinyVGG
# following the structure of Input Layer -> [Convolutional layer -> activation layer -> pooling layer] -> output layer

class FashionMNISTModelV2(nn.Module):
	def __init__(input_shape: int, hidden_units: int, output_shape: int):
		super().__init__()
		self.block_1 = nn.Sequential(
			nn.Conv2d(
				in_channels=input_shape,
				out_channels=hidden_units,
				kernel_size=3, # How wide/tall is the filter we slide over the image
				stride=1, # default
				padding=1,  # options = "valid" (no padding) or "same" (output has same shape as input), or any int
			)
			nn.ReLU(),
			nn.Convd2d(hidden_units, hidden_units, 3, 1, 1)
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)  # Defaults stride is the same as kernel size
		)
		self.block_2 = nn.Sequential(
			nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
			nn.RelU(),
			nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
			nn.RelU(),
			nn.MaxPool2d(2)
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
		)

	def forward(self, x: torch.Tensor):
		x = self.block_1(x)
		x = self.block_2(x)
		x = self.classifier(x)
		return x

torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

# Let's test these layers out by creating some toy data just like the data used in CNN explainer
torch.manual_seed(42)

images = torch.randn(size=(32,3,64,64))  # [bs, channels, height, width]

# Let's create an example nn.Conv2d with various parameters (in_channels, out_channels, kernel_size, stride, padding)
torch.manual_seed(42)
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0)

# (Skipping explanation of what a convolution and pooling is)
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_2, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    test_step(data_loader=test_dataloader,
        model=model_2,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )
model_2_results = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
model_2_results
{'model_name': 'FashionMNISTModelV2',
 'model_loss': 0.3285697102546692,
 'model_acc': 88.37859424920129}
# Nice results!
```

# 4. PyTorch Custom Datasets
- Let's learn how to build a custom dataset, using data related to a specific problem that you're working on!
	- First, we need some data: Food101 is a popular CV benchmark with 1000 images of 101 different kinds of foods, totaling 101,000 images.
		- We'll start with just 3 classes (pizza, steak, sushi), with a random 10% of the 1,000 image per class.

```python
# Assume we have our PIL images
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Let's write a series of transforms that resizes our images from 512x512 to 64x64, flips our images randomly on the horizontal using transforms.RandomHorizontalF/lio(), and turn our images from a PIL image to a PyTorch tensor
data_transforms = transforms.Compose([  # A handy function to combine transforms into a pipeline
	transforms.Resize(size=(64,64)),
	transforms.RandomHorizontalFlip(p=.5),  # Probability 50% of a flip
	transforms.ToTensor()  # This also converts all pixel values from 0-255 to be between 0 and 1
])

# Let's load our image data into a Dataset capable of being used with PyTorch; we can use torchvision.datasets.ImageFolder, where we pass the file path of the target image directory as well as a series of transforms.
from torcvision import datasets
train_data = datasets.ImageFolder(root=train_Dir, transform=data_transform, targt_transforms=None)
test_data = datasets.ImageFolder(root, data_transform)

class_names = train_data.classes # ['pizza', 'steak', 'sushi']
len(train_data), len(test_data)  # (255, 75)

# So our images are in the form of a tensor with shape [3, 64, 64] and the labels are in teh form of an integer related to a specific class (as referenced by the class_to_idx attribute)
# Note that if we wanted to pot our images with matplotlib, we need to use HWC instead of the CHW format that our tensor is currently in -- it's just a matplotlib preference.

# Let's turn out Pytorch Datasets into DataLoaders! This makes them iterable, yielding batches and such.
# The num_workers argument defines how many subprocesses will be created to load your data; the uathor usually sets it to the number of CPUs on his machine, via Python's os.cpu_count()
from torch.utils.data import DataLoader
train_dataloader = Dataloader(
	dataset=train_data,
	batch_size=1, # samples per batch
	num_workers=1, # number of subprocess to use for dat aloading
	shuffle=True  # Wheter the shuffle data
)

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             num_workers=1, 
                             shuffle=False) # don't usually need to shuffle testing data

# ASIDE: What if a pre-built Datset creator like torchvision.datasets.ImageFolder() didn't exist?
# We can build our own custom ImageFolder class by subclassing Dataset
import os
import pathlib
import torch
from PIL import Image # to handle images
from torch.utils.data import Dataset # to subclsas
from torchvision import transforms
from typing import Tuple, Dict, List

# recall that both of these work on the ImageFolder instance
train_data.classes, train_data.class_to_index
# (['pizza', 'steak', 'sushi'], {'pizza': 0, 'steak': 1, 'sushi': 2})

# Let's create a helper function to get classnames
target_directory = train_dir
class_names_found = sorted([entry.name for entry in list(os.scandir(image_path/"train"))])
# Class names found: ['pizza', 'steak', 'sushi']

# Let's turn it into a fn
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
	# 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

find_classes(train_dir) # (['pizza', 'steak', 'sushi'], {'pizza': 0, 'steak': 1, 'sushi': 2})

#3 Now let's build our custom Datset class ot replciate ImageFolder!
from torch.utils.data import Dataset
class ImageFolderCustom(Dataset):
	def __init__(self, targ_dir: stt, transform=None) -> None:
		self.path = list(pathlib.Path(targ_dir).glob("*/*.jpg"))  # only getting jpgs
		self.transform = transform
		self.classes, self.classes_to_idx = find_classes(targ_dir)

	def load_image(self, index:int) -> Image.Image:
		# Given an index, find that index in self.Path and open as a PIL image
		image_path = self.paths[index]
		return Image.open(image_path)

	def __len__(self) -> int:
		return len(self.paths)

	def __getitem__(self, index: int) -> Tuuple[torch.Tensor, int]:
		img = self.load_image(index)
		class_name = self.paths[index].parent.name
		class_idx = self.class_to_idx[class_name]

		if self.transform:
			return self.transform(img), class_idx
		else:
			return img, class_idx

# Augment Training Data
train_transforms = transforms.Compose([
   transforms.REsize((64,64)),
   transforms.RandomHorizontalFLip(p=.5),
   transforms.ToTensor()
])

test_transforms = transforms.Compose([
  transforms.Resize((64,64)),
  transforms.ToTensor()
])

train_data_custom = ImageFolderCustomer(targ_dir = train_dir, transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)

# Seems like it worked! Let's turn these into dataloaders
from torch.utils.data import DataLoader

train_dataloader_custom = DataLoader(
	 dataset=train_data_custom,
	 batch_size=1,
	 num_workers=0,  # No explanation for this, lol
	 shuffle=True 
 )
 test_dataloader_custom = DataLoader(test_data_custom, batch_size=1, num_workers=0, shuffle=False) # Dont need to shuffle test

# Let's try out some dat augmentation -- transforms.RandAugment() and trasnforms.TrivialAugmentWide() generally perform bettert han hand-picked ones. Given a set of transforms, we randomly pick a number o them to perform at an image, and at a random magnitude betewen a given range (higher magnitude=more intense)

train_transforms = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.TrivialAugmentWide(num_magnitude_bins=31),  # How intense; 31 is default
   transforms.ToTensor()
])

test_transforms = transforms.Compose([
	  transforms.Resize((224,224)),
	  transforms.ToTensor()  # We don't usually perform data augmentation on the test set ((Though you do with test-time augmentation!))
])

# Let's construct a CV model to see if we can classify an image as pizza/steak/sushi
simple_transform = transforms.Compose([
   transforms.Resize((64,64)),
   transforms.ToTensor()
])

# Let's now load the data, turning each of our training and test folders into a `Dataset`, then `DataLoader`
train_data_simple = datasets.ImageFolder(root=train_dir, tranform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

# And now to dataloaders
bs=32
num_workers=os.cpu_count()
train_dataloader_simple = DataLoader(train_data_simple, batch_size=bs, shuffle=True, num_workers=num_workres)
test_dataloader_simple = DataLoader(train_data_simple, batch_size=bs, shuffle=False, num_workers=num_workres)

# Great, now that we have our dataloaders,let's build a model.
class TinyVGG(nn.Module):
	def __init__(self, input_shape: int, hidden_unitS: int, output_shape: int) -> None:
		super().__init__();
		self.conv_block_1 = nn.Sequential(
			nn.Conv2d(
				in_channels=input_shape,  #input to hidden
				out_channels=hidden_units
				kernel_size=3,
				stride=1,
				padding=1
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=hidden_units,  #hidden to hidden
				out_channels=hidden_units
				kernel_size=3,
				stride=1,
				padding=1
			),
			nn.ReLU()
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.conv_block_2 = nn.Sequential(
			nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
			nn.ReLU()
		)
		self.classifier = nn.Sequential(  # Used as the head at hte end of the network
			nn.Flatten(),
			nn.Linear(in_features=hidden_units*16*16, out_features=output_shape)
		)

	def forward(self, x.torch.Tensor):
		x = self.conv_block_1(x)
		x = self.conv_block_2(x)
		x = self.classifier(x)
		return x

torch.manual_seed(42)
model_0 = TinyVGG(
	input_shape=3,
	hidden_units=10,
	output_shape=len(train_data.classes)
).to(device)

img_batch, label_batch = next(iter(train_dataloader_simple))
single_img, single_label = img_batch[0].unsqueeze(dim=0), label_batch[0]

model_0.eval()
with torch.inference_mode()
	pred = model_0(single_image.to(device))

# We can use a library called torchinfo, which comes with a helpful summary() method that takes a PyTorch model as well as an input_shape, and returns what happens as the tensor moves from your model
from torchinfo import summry
summary(model_9, input_size=[1,3,64,64])
```
![[Pasted image 20240620154238.png|300]]
The output of torchinfo.summary() gives us a whole bunch of information about our model, nice!

Let's now make some training/test loops to train our model
```python
def train_step(
   model: torch.nn.Module,
   dataloader: torch.utils.data.Dataloder,
   loss_fn: torch.nn.Module,
   optimizer: torch.optim.Optimizer
):
	train_loss, train_acc = 0,0
	mode.train()
	for batch, (X, y) in enumerate(dataloader):
		pred = model(X)
		
		loss += loss_fn(pred, y).item()
		y_pred_class = torch.argmax(torch.softmax(dim=1), dim=1)  # Get the predicted classes by softmax+argmax
		train_acc += (y_pred_class == y).sum().item()/len(y_pred) # Count percentage of agreeing predictions in batch
	# Adjustm etrics to get average loss + accuracy per batch
	train_loss /= len(dataloader)
	train_acc /= len(dataloader)
	return train_loss, train_acc

# Now let's do a similar thing for a test_ste
def test_step(model, dataloader, loss_fn):  # Note we don't need optimizer
	test_loss, test_acc = 0, 0
	model.eval()
	with torch.inference_mode():
		for batch, (X,y) in enumerate(dataloader):
			preds = model(X)  # logits
			test_loss += loss_fn(preds, y).item()  # The loss fn should just return a scalar and not a tensor lol

			test_pred_labels = np.argmax(np.softmax(preds, dim=1), dim=1) # clases
			test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)  # percent accuracy in batch

	test_loss /= len(dataloader)
	test_acc /= len(dataloader)
	return test_loss, test_acc

# Now that we have both a train_step and test_step, let's make a train fn
def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs):
	results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

	for epoch in tqdm(range(epochs)):
		train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer)
		test_loss, test_acc = test_step(model, test_dataloader, loss_fn, optimizer)

		print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

		results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

	return results
```

Now we've got everything we neeed to train and evaluate our model!
```python
NUM_EPOCHS = 5

model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data.classes)).to(device)  # 3=RGB
loss_fn=nn.CrossEsntropyLoss()
optimizer = nn.optim.SGD(params=model_0.parameters(), lr=0.1)

model_0_results = train(model_0, train_dataloadeR_simple, test_dataloader_simple, optimizer, loss_fn, NUM_EPOCHS)
# Looks like our model performed pretty poorly (test_acc=.1979 at epoch 5) -- how can we improve it?

# Let's plot the model loss curves.
def plot_loss_curves(results: Dict[str, List[float]]):
	loss = results["train_loss"]
	test_loss = results["test_loss"]
	accuracy = resutls["train_acc"]
	test_accuracy = results["test_acc"]

	epochs = range(len(results["train_loss"]))
	
	# Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

```
![[Pasted image 20240620160702.png|300]]
Woah, looks like things are all over the place. It also looks like our trainign loss is much much lower than our test loss, meaning our model is overfitting/
![[Pasted image 20240620160800.png|300]]

We have much higher test loss than training loss -- there are many techniques to reduce overfitting, but a common one is introducing a regularization term to our loss function.
- Get more data/Data augmentation
- Simplify the model
- Early stopping
- Dropout layers
- Learning rate decay

If a model were underfitting, we could:
- Have a deeper/more flexible model with better capacity
- Tweak the learning rate
- Train for longer
- Use less regularization

Let's try out using TinyVGG with some Data Augmentation to create some additional data (+ robustness)
```python
train_transform_trivial_augment = transforms.Compose([
	transforms.Resize((64,64)),
	transforms.TrivialAugmentWide(),
	transforms.ToTensor()
])
test_transform = transforms.Compose([
	 transforms.REsize((64,64)),
	 transforms.toTensor()
])

# Now we can recreate our datasets using these transforms
train_data_augmented = datasets.ImageFolder(train_dir, transfor=train_transform_trivial_augment)
test_data_simple = datasets.ImageFolder(test_dir, tranform=test_transform)

bs=32
nw = os.cpu_count()
torch.manual_seed(42)

train_dataloader_augmented = DataLoader(train_data_augmented, bs=bs, shuffle=True, num_workres=nw)
test_dataloader = DataLoader(test_data_simple, bs=bs, shuffle=True, num_workres=nw)

torch.manual_seed(42)  # I still don't really understand exactly when we have to set this seed
model_1 = TinyVGG(input_shape=3, hidden_units=10,output_shape=len(train_data_augmented.classes)).to(device)  # Create hte model and move it to device

torch.manual_seed(42)
torch.cuda.manual_seed(42)  # FIRST TIME We'VE SEEN THIS, I THINK? When do we need to set teh cuda seed?
n_epochs=5

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.optim.SGD(params=model_1.parametrs(), lr=0.001)
model_1_results = train(model=model_1, 
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS)

# Still, it doesn't look like our model performed very well again. CAn we plot the loss curves of our Model?
# it looks like it's about as equally bad as the previous one, and were pretty sporadic. What else can we do to improve performance?

```


# 5. PyTorch Going Modular

This section answers the question: How do I turn my notebook into Python scripts?

Going modular involves turning notebook code (eg from a Jupyter Notebook) into a series of different Python scripts that offer similar functionality. 
We might break out notebook into:
- `data_setup.py`: File to prepare/download data is needed
- `engine.py`: File containing various training functions
- `model_builder.py` or `model.py`: File to create a PyTorch model
- `train.py`: A file to leverage all other files and train a target PyTorch model
- `utils.py`: A file dedicated to helpful utility functions


# 6. PyTorch Transfer Learning


# 7. PyTorch Experiment Tracking



# 8. PyTorch Paper Replicating


# 9. PyTorch Model Deployment



# 10. PyTorch 2.0


# 11. PyTorch Extra Resources


# 12. PyTorch Cheatsheet


# 13. The Three Most Common Errors in PyTorch

