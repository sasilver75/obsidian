#article 
Link: https://vickiboykis.com/2024/02/28/gguf-the-long-way-around/

Takeaways:
Eh, it's okay. Basically talks about the problems with GGML (using a list instead of a dict for model hyperparameters, not including the model architecture) which led to GGUF. The history about why Pickle wasn't secure, and how that led to the development of the Safetensors library @ HuggingFace is also interesting.

---

How we use LLM artifacts:
1. As an API endpoint for proprietary models hosted by OpenAI/Anthropic/major cloud providers
2. As model artifacts downloaded from HuggingFace's model hub and/or trained/fine-tuned using HuggingFace libraries, and hosted on local storage.
3. As model artifacts available in some format optimized for local inference, typically ==GGUF==, and accessed via applications like `llama.cpp` or `ollama`
4. As [[ONNX]], a format that optimizes sharing between backend frameworks.

For a side project, she's using `llama.cpp`, a `C/C++`-based inference engine targeting M-series GPUs on Apple Silicon.

[[GGUF]], or "[[GGUF|GPT-Generated Unified Format]]" is the file format used to serve models on `llama.cpp` and other local runners like `Llamafile`, `Ollama`, and `GPT4All`

To understand how it works, we need to look into ML models and the types of artifacts they produce:

# What is a machine learning model
- A model is a file (or collection of files) that contain the model architecture, weights, and biases of the model generated from a training loop.

- In the transformer, we have many moving parts:
	- For the input, we use training data corpuses aggregated from human-generated natural language content.
	- For the algorithm, we:
		- Convert that data into embeddings
		- Positionally encode the embeddings to provide information about where words are in relation to eachother in the sequence.
		- Create multi-headed self-attention for each word in relation to each other word in the sequence, based on an initialized combination of weights.
		- Normalize layers via softmax
		- Run the resulting matrix through a FFNN
		- Project the output into the correct vector space for the desired task
		- Calculate loss and then update model parameters
	- The output:
		- Generally, for chat completions, the model returns the statistical likelihood that any given word completes a phrase. It does this again for every word in the phrase, because of its autoregressive nature.

# Starting with a simple model
- Let's take a step back in complexity and consider linear regression, which are very similar to neural networks.
- Let's say we're producing an artisinal hazlenut spread for statisticians, and want to predict how many jars of Nutella we'll produce on a given day.

When training our model, we might use a training, validation, and test dataset (80% training/validation, 20% testing, perhaps).

We define some "architecture"
![[Pasted image 20240429155122.png]]

And then our task during training is to continuously take in data, make predictions using our model and its randomly-initialized parameters, calculate the "loss", and hope to find (eg) the smallest sum of squared differences (eg). We optimize this through grdient descent, where we start with random weights, make predictions, evaluate our loss function, and make updates to our parameters that minimze the loss.

If we defined our model in pytorch

```python
... {Omitted}
```

```python
model = LinearRegression()
print(model.state_dict())
```
This `state_Dict` attribute holds the information about each layer, and the parameters in each layer -- the weights and biases. At its heart, it's a Python dictionary!

In this case, the implementation for LinearRegression returns an ordered dict with each layer of the network, and the values of those layers. Each of the values is a `Tensor`

```python
OrderedDict([('linear.weight', tensor([[0.5408]])), ('linear.bias', tensor([-0.8195]))]) 

for param_tensor in model.state_dict(): 
print(param_tensor, "\t", model.state_dict()[param_tensor].size()) 

linear.weight torch.Size([1, 1])
linear.bias torch.Size([1])
```

For our tiny model, it's just a small OrderedDict of tuples, but you can imagine that in a 7B model, this can take up to 14GB in GPU.

We run our forward and backward passes for the model, in each step doing a forward pass to perform the calculation, a backward pass to update the parameters of the model, and then adding all that information to the model parameters.

Once we've completed the loops, we've trained the model artifact; now we have an in-memory object representing hte weights, biases, and metadata of the model, stored in within our instance of the LinearRegression module.
# Serializing our objects
- Now we've got stateful Python objects in-memory that convey the state of our model. But what happens when we want to persist the very large model that we spent 24+ hours training, and use it again?

> This might be useful if you have a compute cluster containing GPUs or other accelerators that you can use to run models.
> While these models are training, it might be useful to save snapshots of their progress in such a way that they can be reloaded and resumed, if hardware fails or the jobs are pre-empted. Once the models are trained, the researcher will want to load them again (potentially a final snapshot) in order to run evaluations on them again.

What do we mean by serialization? It's the process of writing objects/classes from our programming runtime to a file. Deserialization is the process of converting data to a programming language object in memory.

We serialize our data into a bytestream that we can write to a file.

Since many transformer-style models use PyTorch these days, our artifacts use PyTorch's `save` implementation for saving files to a disc.

# What is a file?
- Again, let's abstract away the GPU for simplicity and assume we're performing all of these computations in CPU.
- Python objects in memory. This memory is allocated in a special private heap at the beginning of their lifecycle. The private heap is managed by the Python memory manager, with specialized heaps for different object types.
- When we initialize our PyTorch model object, the operating system allocates memory through lower-level C functions (eg `malloc`) via default memory allocators.

# How does PyTorch write objects to files?
- We can also more explicitly see the types of these objects in memory.
- Among all the other objects created by PyTorch and Python system libraries, we can see our `Linear` object, which has a `state_Dict`; we want ot serialize thsi object into a bytestream to write it to disk.

- Python serializes objects to disk using Python's pickle framework and wrapping the pickle `load` and `dump` methods.
	- Pickle traverses the object's inheritance hierarchy and converts each object encountered into streamable artifacts.
	- It does this recursively for nested representations (eg that `Linear` inherits from `nn.Module`) and converts these representations to byte representations that can be written to file.

eg
```python
import pickle

X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)

with open('tensors.pkl', w'b') as f:
	pickle.dump(X, f)

```


When we inspect the pickled object with `pickletools`, we get an idea of how the data is organized.

...

The ==main issue== with `pickle` as a file format is that ==it not only bundles executable code, but that there are *no checks on the code being read, and without schema guarantees, you can pass something to Pickle that's malicious!*==

> The insecurity isn't because Pickles contain code, it's because they create objects by calling constructors that are named in the Pickle!
> Any callable can be used in place of your class name to construct objects.
> Malicious pickles will use other Python callables as the "constructors"
> For example, instead of executing "models.MyObject(17)," a dangerous pickle might execute "os.system('rm -rf /')".
> The unpickler can't tell the difference between models.MyObject and os.system! Both are names it can resolve, producing something it can call. The unpickler executes either of them as directed by the pickle.

## How Pickle works
- As Transformer-based models picked up after the release of Trasnformer paper in 2017, so did the use of the `transformer` library, which delegates the load call to a Pytorch's `load `methods, which uses pickle. So security became an issue!
- As ML use with PyTorch exploded, and security issues came to a head. In 2021, Trail of Bits released a post on the [insecurity of Pickle files](https://github.com/trailofbits/fickling).
- Engineers at [[HuggingFace]] started developing a library known as `safetensors` as an alternative to Pickle; it was developed to be efficient but also safer and more ergonomic than pickle.
	- `safetensors` is not bound to Python as closely as Pickle: With pickle, you can only read/write in Python, but [[Safetensors]] is compatible across languages.
	- `safetensors` limits language execution functionality available on serialization/deserialization.
	- Because the backend of `safetensors` is written in Rust, it enforces type safety more rigorously, and optimized specifically with tensors in mind in a way that Pickle was not.
- A safety audit from Trail of Bits and [[EleutherAI]] was conducted and found satisfactory, so HuggingFace added it as the default format for the models on the Hub going forward.

## How `safetensors` works
- ...
- We start to notice from looking at statedicts, pickle files, and safetensor files that we need to store:
	1. A large collection of vectors
	2. metadata about those vectors
	3. hyperparameters

We then need to be able to instantiate model objects that we can hydrate (fill) with that data and run model operations on.

## Checkpoint files

- So far we've started to look at simple `state_dict` files and dsingle `safetensors` files.
-  But if you're training a long-running model, it's likely that you'll have more than just weights and biases to save!
	- The opimizer's state_dict, including buffers and parameters that are updated as the model trains
	- The epoch you left off on
	- The latest recorded training loss
	- External `torch.nn.Embedding` layers
	- more
- This is also saved as a Dictionary and pickled, then unpickled when you ned it; all of this is saved to a dictionary, the `optimizer_state_dict`, distinct from the `model_state_dict`.

![[Pasted image 20240429163455.png]]

# GGML
- As work to migrate from pickle to safetensors was ongoing for generalized model finetuning and inference, Apple Silicon continued to get better!
- Georgi Gerganov's project to make OpenAI's [[Whisper]] model run locally with Whisper.cpp was a success, catalyzing later projects.
- The combination of the release of [[LLaMA 2]], along with compression techniques like [[Low-Rank Adaptation|LoRA]], performant-out-of-the-box LLMs were now workable locally for the hobby community.

- Based on the success of `whisper.cpp`, [[Georgei Gerganov]] created [[llama.cpp]], a package for working with LLaMA model eights, which were originally in pickle format, in GGML format instead, for local inference.

- [[GGML]] was initially both a library and a complementary format created specifically for on-edge inference for Whisper. You can also perform fine-tuning with it, but generally it's used to read models trained on PyTorch in GPU Linux-based environments and converted to GGML to run on Apple Silicon.

The resulting GGML file compresses all of these into one and contains:
1. A magic number with an optional version number
2. ==Model-specific hyperparameters==, including metadata about the model, such as the number of layers, the number of heads, etc.
3. A filetype that describes the type of the majority of the tensors, for GGML files, the quanitzation version is encoded in the ftype divided by 1000.
4. An ==embedded vocabulary==, which is a list of strings with length prepended
5. A ==list of tensors== with their length-prepended name, type, and tensor data.

There are several elements that make GGML more efficient for local inference htan checkpoint files:
- Makes use of 16-bit FP representations of model weights (Generally, torch initializes FP datatypes in 32-bit by default). These half-precision weights use 50% less memory at compute and inference time without significant los in model accuracy.
- Uses C, which offers more efficient memory allocation (And faster execution) than Python.
- GGML was built optimized for Apple Silicon

Unfortunately, in its move to efficiency, GGML contained a number of breaking changes that created issues for user
- Since both data and metadata and hyperparameters were written into the same tfile, if a model added hyperparameters, it would break backward compatability that the new file couldn't pick up.
- No model architecture metadata is present in the file, and each architecture required its own conversion script

All of this led to brittle performance and the creation of GGUF!

# Finally, GGUF
- [[GGUF]] has the same type of layout as GGML, with metadata and tensor data in a single file, but in addition, the file is designed to be backwards-compatible.
- The key difference is that previously, instead of a list of values for the hyperparameters, the new file format uses a key-value look table, which accommodates shifting values.

![[Pasted image 20240429165555.png|450]]
A GGUF file