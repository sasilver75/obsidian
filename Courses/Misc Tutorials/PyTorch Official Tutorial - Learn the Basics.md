https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

## 0. Quickstart

### Working with Data
Pytorch has two primitives to work with data:
- `torch.utils.data.Dataset`: Stores the samples and their corresponding labels
- `torch.utils.data.DataLoader`: Wraps an iterable around the Dataset (supports batching, sampling, shuffling, multiprocess data loading)

PyTorch offers domain-specific libraries like TorchText, TorchVision, and TorchAudio, all of which include datasets; we'll be using a TorchVision dataset FashionMNIST.

Every TorchVision `Dataset` constructor contains two arguments: `transform` and `target_transform` to modify the samplsa samples and labels respectively:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get training and test data
training_data = datasets.FashionMNIST(
	root="data",
	train=True,
	download=True,
	transform=ToTensor()
)

test_data = datasets.FashionMNIST(
	root="data",
	train=True,
	download=True,
	transform=ToTensor()
)
```

We can now pass our `Dataset` as an argument to `DataLoader`, which wraps an iterable over our dataset, and supports automatic *batching, sampling, shuffling, and multiprocess dataloading.*

```python
bs = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=bs)
test_dataloader = DataLoader(test_data, batch_size=bs)

# Now if we iterate, we'll have batch sizes of [64,1,28,28], and y vectors of [64]
for X, y in test_dataloader:
	...
```

Let's create a NN in PyTorch now, as a class that inherits from ``nn.Module`. 
- We can define the layers of the network in the `__init__` function, and specify how data will pass through the network in the `forward` function.

```python
# Get CPU/GPU/MPS device for training
device = (
	"cuda"
	if torch.cuda_is_available()
	else "mps"
	if torch.backends.mps.is_available()
	else "cpu"	  
)

# Define Mdoel
class NeuralNetwork(nn.Module):
	def __init__(self):
		# This is where we define the layers of our network
		super().__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, 512),  # Here, 28*28 is our flattened image, and 512 is our hidden dimension d, which is sort of a hyperparameter of our network.
			nn.ReLU(),
			nn.linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 10) # downscaling our dimensionality to our "answer" space
		)

	def forward(self, x):
		# Given an input, how should data pass through out network?
		x = self.flatten(x)
		# These are just logits, not yet a probability distribution
		logits = self.linar_relu_stack(x)
		return logits

model = NeuralNetwork().to(device)
print(model)

# result is
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

Now that we have a model architecture, we need a loss function and optimizer before we start training.
Note that we define these in a local outside of our NN class, because they're not intrinsic parts of the NN :).

```python
loss_fn = nn.CrossEntropyLoss()
# Interesting that we have to pass in our model.parameters()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

Now we can do a single training loop, where our model sequentially makes predictions on a batch and backpropagates the prediction error to adjust model parameters.

```python
def train(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)  # The total number of samples in the dataset 
	model.train()
	for batch, (X, y) in enumerate(dataloader):  # number of batches, (batch)
		# Interesting that we need to do this dtype/device conversion
		X, y = X.to(device), y.to(device)

		# Compute the prediction error
		pred = model(X)  # Crate predictions
		loss = loss_fn(pred, y)  # Get the Cross Entropy loss between prediction and model

		# Backpropagation
		loss.backward() # Computes dloss/dx for every parameter with requires_grad=True
		optimizer.step()   # Updates the value of x using the gradient x.grad
		optimizer.zero_grad()  # Zero out the x.grad gradients for all parameters x in the optimizer; it's important to call this before loss.backward.

		# every 100 batches, print out some statistics
		if batch % 100 == 0:
			loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

We can then also check the model's performance against the test dataset to ensure that it's learning

```python
def test(dataloader, model, loss_fn):
	size = len(dataloader.dataset)  # number of elements in dataset
	num_batches = len(dataloader)  # number of batches
	model.eval()  # A switch for some specific layers/parts of model that behave different during training and inference, like Dropout layers, BatchNorm layers, etc. It's common to use this with torch.no_grad() as well.
	test_loss, correct = 0, 0  # Initialize counters
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)  # Convert to appropriate data/device type
			pred = model(X)  # Make predictions for each record in X
			test_loss += loss_fn(pred, y).item() # .item() just returns the value of this tensor as a standard Pytohon number; only works for tensors with one element.
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # argmax returns the index of the maximum value over the 1st dimenison, here. 
	test_loss /= num_batches  # To get the average loss per batch
	correct /= size  # to get the percentage correct (accuracy)
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```
This training procedure above only runs for a single epoch; we'd like to continue it over several iterations (epochs)!

```python
epoch = 5
for t in range(epochs):
	print(f"Epoch {t+1\n------})
	train(train_dataloader, model, loss_fn, optimizer)
	test(test_dataloader, model, loss_fn)
print("Done!")
```

Now that we've got a model, we might want to save it by serializing its internal state dictionary (containing the model parameters)
Torch makes this easy:
```python
torch.save(model.state_dict(), "model.pth")  # Path is second argument
print("Saved PyTorch Model State to model.pth")
```

Later, we can load them:
```python
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))
```

And we can use it to make predictions!
```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()  # Basically putting it in "inference" mode (non-training)
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
	x = x.to(device)
	pred = model(x)
	predicted, actual = classes[pred[0].argmax(0)], classes[y]
	print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

## 1. Tensors
Tensors are a specialized data structure similar in many ways to arrays and matrices; in PyTorch, we use tensors to encode the input and outputs of a model, as well as the model's parameters.
- They're similar to NumPy's ndarrays, but tensors can run on GPUs or other hardware accelerators, and are also optimized for automatic differentiation.

```python
import torch
import numpy as np

# We can create tensors directly from data, and the data type is automatically inferred
data = [[1,2], [3,4]]
x_data = torch.tensor(data)

# We can also create a tensor directly from a NumPy array (and vice veras)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# We can also create them from other tensors! The new tensor will retain the properties (shape, datatype) of the argument tensor, unless explicitly overridden
x_ones = torch.ones_like(x_data)  # A tensor filled w the scalar value 1, same size as input
x_rand = torch.rand_like(x_data, dtype=torch.float) # Filled by unit normal distribution, with a datatype override

# We can create it from a tuple of tensor dimensions as well:
shape = (2,3,)  # Two rows, three columns
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
```

Tensor attributes describe their shape, datatype, and the device on which they're stored.
```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

Operations on Tensors
- There are over 100 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing).
- Each of these operations can be run on the GPU
- By default, tensors are created on the CPU! ==We need to explicitly move tensors to the GPU using the `.to` method.==
	- Keep in mind that copying large tensors across devices can be expensive in terms of time and memory!

```python
# Moving our tensor to GPU if available
if torch.cuda.is_available():
	tensor = tensor.to("cuda")
```

Let's try out some operations!
```python
# Slicing and Indexing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0  # This turns the second column into all zeroes, using tensor broadcasting
print(tensor)

# Joining Tensors: We can use torch.cat to concatenate a sequence of tensors along a given dimension. torch.stack is another similar tensor joining operator.
t1 = torch.cat([tensor, tensor, tensor], dim=1)  # dim=1 here is columns, so we're concatting them left-to-right, making a pretty wide 4x12 matrix from 3 4x4 matrices.
print(t1)
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])

# Arithmetic operations!
# The @ symbol is used for matrix multiplications in Python, but you can also use .matmul
# (Recall we have a 4x4 tensor called tensor, here, annoyingly)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = tensor.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)  # Not clear to me what out does here

# This computes the element-wise product. z1, z3, z3 will have the same value.
z1 = tensor * tensor  # The sum of this swould be the dot product :)
z2 = tensor.mul(tensor)

# If you have a single-element tensor (eg from aggregating all values of a tensor into one value), you can convert it to a Python numerical using .item():
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
12.0 <class 'float'>

# In-Place operations that store the result into the operand tensor are called "in-place" operations, and are always denoted by a _ suffix
tensor.add_(5)
print(tensor)
tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

Tensors on the CPU and NumPY arrays can share their underlying memory locations; changing one will change the other!

```python
t = torch.ones(5)
n = t.numpy
print(t)
print(n)
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]

# Now, a change in the tensor reflects in the NumPy array:
t.add_(1)
print(t)
print(n)
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]

# We can convert a NumPy array back to a tensor, and again changes in the NP array will reflect in the tensor:
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]

```


## 2. Datasets and DataLoaders
- Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code to be decoupled from our model training code for better readability and modularity.
- PyTorch has two data primitives that can help you use pre-loaded datasets, as well as your own:
	- `torch.utils.data.Dataset`: Stores the samples and corresponding labels
	- `torch.utils.data.DataLoader`: Wraps an iterable around `Dataset` that enables easy access to samples; also supports things like batching, shuffling, etc.

Loading a Dataset (here, Fashion-MNIST from TorchVision, a collection of 60k train+10k test 28x28 grayscale images from 10 classes)
```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets  # TorchVision-specific datasets; e.g. FashionMNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
	root="data",  # the path where the training/test data is stored
	train=True,  # specifies training or test dataset
	download=True,  # downloads the data from the internet if not available already at root
	transform=ToTensor()  # feature transformations (label transformations available in a target_transform argument)
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Let's visualize some samples in our training data:
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3  #3x3 graph
for i in range(1, cols * rows + 1):  # for each of our 9 cells
    sample_idx = torch.randint(len(training_data), size=(1,)).item()  # sample a random idex
    img, label = training_data[sample_idx]  # get the item at that index
    figure.add_subplot(rows, cols, i)  # add a subplot for that image
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")  # stick in the image
plt.show()
```

We can create a custom Dataset class for our files; custom Dataset classes must implement:
- `__init__`
- `__len__`
- `__getitem__`

Here's an example for where some FashionMNIST images are stored in a directory `img_dir`, and their labels are then separately stored in a CSV file `annotations_file`
```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
	# Custom Dataset classes need to implement __init__, __len__, and __getitem__
	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
		self.img_labels = pd.read_csv(annotations_file)
		slef.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		# The getitem dunder method is usually a function that returns the value at specified idx
		# here, it returns the (x,y) at the specified index
		img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
		image = read_image(img_path)  # Get the image at index idx
		label = self.img_labels.iloc[idx, 1]  # Get the label at index idx
		if self.transform:  # If the Dataset has an x transform, apply it
			image = self.transform(image)
		if self.target_transform: # If the Dataset has a y transform, apply it
			label = self.target_transform(label)
		return image, label
	
```

Now let's prepare our data for training using the `DataLoaders` classes!
- When training a model, we typically want to pass samples in "minibatches," *reshuffle the data at every epoch* to reduce model overfitting, and use Python's `multiprocessing` to speed up data retrieval.

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)  # Shuffle means it will be reshuffled at every epoch!

# Iterate through the DataLoader; each iteration now returns a batch of train_features and train_labels (of bs=64 features and labels). Because we specified shuffled=True, after we iterate over all batches, the data is shuffled.
train_features = train_labels = next(iter(train_dataloader)) # grab first batch
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])

img = train_features[0].squeeze()  # Returns a tensor with all dimensions of size 1 removed
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()  # An image of the first x
```

## 3. Transforms
- Data doesn't always come in its final processed form that's appropriate for ML algorithms; we often use transforms to perform some manipulation of the data, making it suitable for training.
- All TorchVision datasets have two parameters (`transform`, `target_transforms`) to modify the features and labels.
	- The `torchvision.transforms` module offers several commonly-used transforms out of the box.

Our FashionMNIST features are in PIL format, and the labels are integers; for training, we want the features as normalized tensors, and the labels as one-hot encoder tensors; to make these transformations, we'll use `ToTensor` and `Lambda`

```python
import torch
from torchvision import datasets
from torchvision.transsforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
	root="data",
	train=True,
	download=True,
	transform=ToTensor(),
	target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
# Above: We create a tensor filled with zeroes, and then use the in-place (_) scatter_ method, which writes values from the source tensor into the destinator tensor at the specified index, along the specified dimension. So this is just creating a tensor of zeroes with with the i'th index set to 1, where i=class.
```
Above: 
- Our `ToTensor()` transform converts a PIL image or NumPy ndarray into a FloatTensor and scales the pixel intensity values into the range \[0., 1.\].
- Our `Lambda` transform lets us apply any user-defined function; here, we define a function to turn the integer into a one-hot encoded vector.

## 4. Build Model
- NNs are comprised of layers/modules that perform operations on data. The `torch.nn` namespace provides all the building blocks needed to build your NN.
	- Every *module* in Pytorch subclasses the nn.Module
		- In PyTorch, we use modules to represent NNs. Modules are are tightly integrated with PyTorch's autograd computational, and are building blocks of stateful computation.
	- A NN is itself a module that consists of other modules (layers) -- this nested structured allows for building and managing complex architectures easily.

Let's build one to classify images in FashionMNIST
```python
import os
import torch
from torch import nn
frmo torch.utils.data import DataLoader
from torchvision import datsets, transform

device = "cuda" if torch.cuda_is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class NeuralNetwork(nn.Module):  # Note inheritance from nn.Module
	def __init__(self):
		# This is where we define our layers
		super().__init__()
		self.flatten = nn.Flatten()  # Not quite a layer, more of a function, right?
		# nn Sequential is a handy convenience tool; accepts any number of modules and its forward() method accepts any input and forwards it to the first module it contins, then chains outputs to inputs sequentially for each subsequent module.
		self.linear_relu_stack = nn.Sequential(  
			nn.Linear(28*28, 512),
			nn.ReLU(),
			nn.Linar(512, 512),
			nn.ReLU(),
			nn.LInear(512, 10)  
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits

# We cerate an instance of our NeuralNetwork class, and move it to our device.
model = NeuralNetwork.to(device)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits) # Interesting that we're creating a softmax layer on the fly here and then invoking it on our logits
y_pred = pred_probab.argmax(1)  # Then selecting the index (?) of the ... highest class prob
```

Many layers inside a NN are *parametrized*, having weights and biases associated with them that are optimized during training.
Subclassing `nn.Module` automatically tracks all fields detailed inside your model object, and makes all parameters accessible using either `parameters()` or `named_parameters()` methods.
```python
for name, param in model.named_parameters():
	print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

  

Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0273,  0.0296, -0.0084,  ..., -0.0142,  0.0093,  0.0135],
        [-0.0188, -0.0354,  0.0187,  ..., -0.0106, -0.0001,  0.0115]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0155, -0.0327], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0116,  0.0293, -0.0280,  ...,  0.0334, -0.0078,  0.0298],
        [ 0.0095,  0.0038,  0.0009,  ..., -0.0365, -0.0011, -0.0221]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([ 0.0148, -0.0256], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[-0.0147, -0.0229,  0.0180,  ..., -0.0013,  0.0177,  0.0070],
        [-0.0202, -0.0417, -0.0279,  ..., -0.0441,  0.0185, -0.0268]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([ 0.0070, -0.0411], device='cuda:0', grad_fn=<SliceBackward0>)

```

## 5. Automatic Differentiation
- When training NNs, the most frequently-used algorithm is back-propagation.
	- In this algorithm, parameters (model weights) are adjusted are adjusted according to the gradient of the loss function with respect to a given parameter.

PyTorch has a built-in differentiation engine called `torch.autograd`. It supports automatic computation of gradients for any computational graph!

```python
import torch

x = torch.ones(5)
y = torch.zeros(3)

# Defining a simple one-layer NN with parameters w, b
w = torch.randn(5, 3, requires_grad = True)  # MatMul Parameters
b = torch.randn(3, requires_grad = True) # Bias Parameters

z = torch.matmul(x, w) + b

loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)
```

![[Pasted image 20240618194711.png]]
In this network, w and b are parameters that we need to optimize -- we need to be able to compute the gradients of the loss function with respect to these variables -- in order to do this, we set the `requires_grad` property of these tensors to be true.

A function that we apply to tensors to construct a computational graph is in fact an object of class `Function`; this object knows how to compute the function in the *forward* direction, and also compute its derivative during the *backward* propagation step.
- A reference to the backward propagation function is stored in the `grad_fn` property of a tensor.

To optimize weights of parameters, we need to compute the derivatives of our loss function with respect to them.
To compute these, we call `loss.backward()`, and then retrieve the values from `w.grad` and `b.grad`:

```python
loss.backward()
print(w.grad)
print(b.grad)
```

We can only obtain the `grad` properties for the leaf nodes of the computational graph which have `requires_grad` set to true -- for all other nodes in the graph, gradients will not be available ((??))
We can only perform gradient calculations using `.backward` *once* on a given graph, for performance reasons!
- If we want to do several backward calls, we need to pass retain_graph=True to the backward call.

```python
z = torch.matmul(x, w)+b
print(z.requires_grad)
True

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
False

# Alternatively, we could use the detach() method on the tensor
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
False
```

==Why would you want to disable gradient tracking?==
- To mark some parameters in your NN as *frozen parameters* 🧊🥶
- To speed up computations when you're only doing forward passes, because computations on tensors that don't track gradients would be more efficient.

## 6. Optimization Loop
- Now that we have a model and some data, it's time to train, validate, and test our model by optimizing our parameters on our data.
- We load the code from the previous sections on Datasets and DataLoaders and Build Model

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
	root="data",
	train=True,
	download=True,
	transform=ToTensor()  # Transform our PIL images to 28x28 Tensors
)

test_data = datasets.FashionMNIST(
	root="data",
	train=False,
	download=True,
	transform=ToTensor() # Transform to tensor
)
# Create DataLoaders, which are iterables yielding bs=64 X,y tuples
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.lienar_relu_stack = nn.Sequential(
			nn.linear(28*28, 512),
			nn.ReLU(),
			nn.linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 10)
		)
	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits

model = NeuralNetworks()
```

Hyperparameters are adjustable parameters that let us control the model optimization process -- different hyperparameter values can impact model training and convergence rates, and they're usually optimized in an outer-loop.
Examples include number of epochs, batch size, learning rate, and more.

```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

Once we set our hyperparameters, we can then train and optimize our model with an optimization loop, with each iteration of the loop being called an epoch, consisting of two main parts:
1. The Training Loop: Iterate over the training dataset and update parameters
2. The Validation/Test Loop: Iterate over the validation dataset to check if model performance is improving on non-training data

Loss Function
- When presented with soem training data, our untrained network isn't likely to give the right answer; loss functions measure the degree of dissimilarity of obtained results to the target value, and it is the loss function that we want to minimize during training.
- Common examples include MSE, NLL, or CrossEntropy.

We can pass our model's outputs to nn.CrossEntropyLoss, which will normalize the logits and compute the prediction error.
```python
loss_fn = nn.CrossEntropyLoss()
```

Optimizers are also important parts of training, and control the process of adjusting model parameters in each step, given a loss. In our example, we'll use Stochastic Gradient Descent (SGD), but there are many others like ADAM and RMSProp that work better for different kinds of models and data.
- All optimization logic is encapsulated in the `optimizerr` object.

We initialize the optimizer by registering the model's parameters that need to be traind, and passing in our LR hyperaparameter:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```
Inside the training loop, optimization happens in three steps:
1. Call `optimizer.zero_grad()` to reset the gradients of model parameters. By default, gradients add up, so to prevent double-counting (or worse), we should explicitly zero them at each iteration.
2. Backpropagate the prediction loss with a call to `loss.backward()`.
3. Once we have our gradients, we call `optimizer.step()` to adjust the parameters by the gradients collected in the backward pass.

Full implementation:
- Define a `train_loop` that loops over our optimization code, and `test_loop` that evaluates the performance against our test data:
```python
def train_loop(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset) # number of records in train dataset
	model.train()  # Tells our model that we're training the model, informing layers like Dropout and BatchNorm that should behave different during training v evaluation. Opposite of .eval()

	# For each batch in our dataset
	for batch, (X, y) in enumerate(dataloadeR):
		pred = model(X)  # Make predictions on features
		loss = loss_fn(pred, y)  # Compute loss between predictions and labels

		loss.backward()  # Determine gradients with respect to 
		optimizer.step()  # Step our optimizer 
		optimizer.zero_grad()  # Zero out the gradients

		# Report trainign loss every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
	# Note that we don't need our optimizer for this loop, cf train_loop
	model.eval() # Set hte model to evaluation mode (eg turning off normalization, dropout layers)
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss, correct = 0, 0  # We're just going to accumulate our test loss and hit counts

	# Oftne used with model.eval() is torch.no_grad, which just tells us not to accumulate gradients on 
	with torch.no.grad():
		for X, y in dataloader:
			preds = model(X)
			test_loss += loss_fn(pred, y).item()  # accumulate cross entropy loss
			correct += (pred.argmax(1)==y).type(torch.float).sum().item()  # accumulate hits

	test_loss /= num_batches  # Get average loss per batch
	correct /=  # Get accuracy over dataset
	    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Now that we have these, we'd put them together simply:
epochs = 10
for t in range(epochs):
	print(f"Starting epoch {t}\n -----")
	train_loop(train_dataloader, model, loss_fn, optimizer)
	test_loop(test_dataloader, model, loss_fn)
print("Done!")
```