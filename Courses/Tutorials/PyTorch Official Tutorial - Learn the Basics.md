
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

	def __getitem__(self):
		# The getitem dunder method is usually a function that returns the value at specified idx
		img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
		image = read_image(img_path)
		label = self.img_labels.iloc[idx, 1]
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transf
	
```

## 3. Transformers


## 4. Build Model


## 5. Automatic Differentiation


## 6. Optimization Loop


## 7. Save, Load, and Use Model