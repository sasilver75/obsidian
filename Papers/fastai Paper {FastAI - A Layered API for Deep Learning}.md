---
aliases:
  - FastAI Paper
tags:
  - paper
---

Link: https://arxiv.org/pdf/2002.04688.pdf
Date: February 2020
Authors: Jeremy Howard, Sylvain Gugger

![[Pasted image 20231123171807.png]]

--------

# Abstract
- `fastai` is a deep learning library that provides users with high-level components that can quickly be combined to provide state-of-the-art results, and also provides low-level components that can be mixed and matched to build new approaches!
	- It aims to do *both* of these things without compromises in
		- ease of use
		- flexibility
		- performance
	- This is possible compared to a layered architecture, which expresses common common underlying patterns of both deep learning and data processing techniques in terms of *decoupled abstractions!*

----

# Introduction
- `fastai` is organized around two main design goals:
	- Be approachable and rapidly productive
	- Be deeply hackable and configurable
- A high-level API powers ready-to-use (but still configurable) functions to train models in various applications (with sensible defaults), and it's built on top of  a hierarchy of lower-level APIs that provide composable building blocks.
	- So you shouldn't have to learn how to use the lowest level stuff in order to rewrite *part* of the high-level API or add particular behavior.

![[Pasted image 20231123223841.png]]

- The ***high-level API*** is most likely to be useful to either beginners or to practitioners who are mainly interested in applying pre-existing deep learning methods -- it has concise APIs over four main application areas:
	- Vision (CV)
	- Text (NLP)
	- Tabular 
	- Collaborative Filtering (RecSys)
- These APIs provide intelligent defaults and behaviors based on all available information.
	- For example, your `Learner` , when provided with an architecture, optimizer, and data, will automatically choose an appropriate loss function where possible.
	- Another example; A training set should generally be ==shuffled==, while a validation set should not. The `DataLoaders` class automatically constructs these validation/training data loaders with these details already handled!
- Other defaults come from the developer's experience and thoughts on best practices, and extend to incorporate state-of-the-art research wherever possible:
	- `fastai` provides transfer-learning-optimized ==batch-normalization==, training, ==layer freezing==, and ==discriminative learning rates==.

* The ***mid-level API*** provides the core deep learning and data processing methods for each of these applications, while the ***low-level APIs*** provide a library of optimized primitives and functional and object-oriented foundations, allowing you to customize the mid-level.

---------
# Applications

##### Vision
Here's an example of how to fine-tune an ImageNet model on the Oxford IIT Pets dataset, achieving close-to-SOA accuracy within a few minutes:
```python
from fastai.vision.all import *
path = untar_data(URLs.PETS)
dls = ImageDataLoaders.from_name_re(  # regex
	path=path,  # path to dataset
	bs=64,  # batch size
	fnames=get_image_files(path/"images"),
	pat = r`/([^/]+)_\d+.jpg$`,  # Regex to capture label from filename
	item_tfms=RandomResizedCrop(450, min_scale=.75)  # Item transforms; will take a 450x450 crop of the image in a random location per-epoch
	batch_tfms=[*aug_transforms(size=224, max_warp=0.), Normalize.from_stats(*imagenet_stats))]  # Batch transforms
)
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
```

Note on the `import *`: It's true that most programmers don't prefer this style, but REPL programmers (like those in a Jupyter notebook) generally prefer the symbols that they need to be directly available to them. The library was carefully designed so that importing in this way only imports the symbols that are *actually* likely to be useful to the user, in order to avoid cluttering the namespace/causing collisions/shadowing other symbols.

```python
dls = ImageDataLoaders.from_name_re(...)
```
Above:
- This `DataLoaders` abstraction represents a combination of training and validation data, and will be described more in later section.
	- You can create these DataLoaders in a number of ways
		- Flexibly, using the Data Block API,
		- Built for specific predefined applications using specific subclasses (like here, with `ImageDataLoaders` , where we create it using a regular expression labeler.

- An interesting thing that you might have noticed in our definition of the DataLoader that we created was that we separated the *item transforms* and the *batch transforms.
	- Item transforms are applied to individual images, using the CPU
	- Batch transforms are applied to a ==mini-batch==, on the GPU if available.
		- `fastai` supports data ugmentation using the GPU, but images need to be the same sized before being batched!
			- The `aug_transforms()` used above selects a set of data augmentations that work well across a variety of vision sets and problems (and it can be customized using parameters to the function!)
				- This is a good example of a "helper" function; it's not strictly necessary, but by providing a single function that curates the best practices and makes the most common types of customization available, users have fewer pieces to learn to get good results!

```python
learn = cnn_learner(dls, resnet34, metrics=error_rate)
```
Above:
- This fourth line creates a `Learner`, which provides an abstraction combining an optimizer, a model, and the data to train it.
	- Each application (text, vision, etc) has a customized function(s) that creates a `Learner`(s), which will automatically handle the details that it can for the user.
- In this case, it:
	- Downloads an ImageNet-pretrained model with the resnet34 architecture (if it isn't already available on your machine in cache)
	- Removes the classification head of the model and replaces it with a head appropriate for this particular dataset
	- Sets the appropriate defaults for the optimizer, weight decay, learning rate, and so forth (assuming the user didn't overwrite them)

```python
learn.fit_one_cycle(4)
```
- Above: 
	- This fits hte model. In this case, it's using the ==1cycle== policy, which is a recent best-practice for training (it anneals both the learning rates and the momentums), and isn't widely available in most deep learning libraries by default.


Here's another example of us doing image segmentation on the CamVid dataset:
```python
from fastai.vision.all import *

path = untar_data(URLs.CAMVID)
dls = SegmentationDataloaders.from_label_func(
	path=path,
	bs=8,  # batch size
	fnames=get_image_filenames(path/"images"),
	label_func=lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
	codes=np.loadtxt(path/'codes.txt', dtype=str),
	batch_tfms=[*aug_transforms(size=(360,480)), Normalize.from_stats(*imagenet_stats)]
)
learn = unet_learner(dls, resnet34, metrics=acc_segment)
learn.fit_one_cycle(8, pct_start=.9)
```
Above:
- These lines of code above for image segmentation are very similar to the previous image classification, except for the lines necessary to tell `fastai` about the differences in the processing of the input data.


#### Text

- In modern NLP, perhaps the most important approach to building models is through fine-tuning already pre-trained language models.

```python
from fastai.text.all import *

# Download/uncompress the IMDB_SAMPLE data
path = untar_data(URLs.IMDB_SAMPLE)
# Read the uncompressed data into a dataframe, and then tokenize it
df_tok, count = tokenize_df(pd.read_csv(path/'texts.csv'), ['text'])
dls_lm = TextDataLoaders.from_df(
	df_tok,
	path=path,
	vocab=make_vocab(count),
	text_col='text',
	is_lm=True  # 👀
)
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=Perplexity()])
learn.fit_one_cycle(1, 2e-2, moms=(.8, .7, .8))
```

If we wanted to fine-tune this model for *classification*, it would follow the same basic steps:
```python
dls_clas = TextDataLoadres.from_df(
	df_tok,
	path=path,
	vocab=make_vocab(count),
	text_col='text',
	label_col'label'
)
learn = text_classifier_learner(dls_clas, AWD_LSTM, metrics=accuracy)
learn.fit_one_cycle(1, 2e-2, moms=(.8,.7,.8))
```

- The biggest challenge with creating text applications is often the *processing* of the input data!
- `fastai` provides a flexible processing pipeline with predefined rules for best practices (such as handling capitalization by adding tokens).
	- For instance, there's a compromise between lower-casing all text and losing information, versus keeping the original capitalization and ending up with too many tokens in your vocabulary!
		- `fastai` handles this by adding a special single token that represents that the *next symbol* should be treated as *uppercase* , and then converts the text itself to lowercase. This keeps the token space small and retains the information.
		- `fastai` actually uses a number of these special tokens.
			- For example, any sequence of more than three repeated characters is replaced with a special repetition token, along with a number of repetitions and then the repeated character! Handy 😺.
- The ==tokenization== process is flexible, and can support many different ==organizers== -- the default used is ==Spacy==. 
- A ==SentencePiece== tokenizer is also provided by the library. Subword tokenization, such as that provided by SentencePiece, has been used in many recent NLP breakthroughs.
- Numericalization and vocabulary creation often requires many tedious lines of code, but in `fastai` this is handled transparently and automatically.
- `fastai`'s text models are based on AWD-LSTM! The community has provided external connected to the popular HuggingFace Transformers library -- the training of the models proceeds in the same way for the vision examples with defaults appropriate for these models having already been selected.


##### Tabular

-  Tabular models have not been widely used in deep learning; ==Gradient-boosted machines== (GBMs) and similar methods are more commonly used in both industry and research settings.
	- Still, there have been example of competitions won using deep learning.
	- Deep learning models are particularly useful for datasets with high cardinality categorical variables, because they provide ==embeddings== that can be used for even non-deep learning models!
- The `pandas` library already provides excellent support for processing tabular data sets, and `fastai` doesn't attempt to replace it -- instead, it adds some additional functionality to DataFrames through various pre-processing functions, such as automatically adding features that are useful for modeling with date data!

The code to create and train a model for this tabular data will look familiar:

```python
from fastai2.tabular.all import *
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
dls = TabularDataLoaders.from_df(
	df,
	path,
	procs=[Categorify, FillMissing, Normalize],
	cat_names=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'], # categorical cols
	cont_names=['age', 'fnlwgt', 'education-num'], # continuous cols
	y_names='salary',  # target col
	valid_idx=list(range(1024, 1260)),  # List of indices to use for the validation set, defaults to a random split
	bs=64  # batch size
)
# Create a learner using tabular_learner
learn = tabular_learner(dls, layers=[200,100], metrics=accuracy)
learn.fit_one_cycle(3)  # 3 epochs of training
```
- `fastai` also integrates with NVIDIA's `cuDF` library, providing e2e GPU optimized data processing and model training.


#### Collaborative Filtering

- Collaborative filtering is normally modeled using a probabilistic matrix factorization approach -- in practice, however, a dataset generally has much more than (just) a userID and productID, but instead has many characteristics of a user, product, time_period, and so forth -- those should be use!
	- Therefore `fastai` tries to close the gap between collaborative filtering and tabular modelling. A collaborative filtering model in `fastai` can be simply seen as a tabular model with high-cardinality categorical variables (eg userID, productID).
	- (A classifc matrix factorisation model is also provided in fastai)

```python
from fastai2.collab import *
# The "Movielens" dataset of ratings
ratings = pd.read_csv(untar_data(URLs.ML_SAMPLE)/'ratings'.csv)
dls = CollabDataLoaders.from_df(ratings, bs=64, seed=42)
# Note here that we set our boundaries a little outside the 0-5 range?
learn = collab_learner(dls, n_factors=50, y_range[0, 5.5])
learn.fit_one_cycle(3)
```


##### Deployment
- `fastai` is mostly focused on model raining, but once this is done you can easily export the model and serve it in production using the `Learner.export` method, which will serialize both the model as well as the input pipeline (just the item transforms, not the training data) to be able to apply the same to the new data.
- The library provides `Learner.predict` and `Learner.get_preds` to evaluate the model on an item or a new *inference DataLoader*! 

-------------

# High-level API design considerations

#### High-level API foundations
- All the `fastai` applications share some basic components; one such component is the visualization API, which uses a small number of methods, the main ones being `show_batch` (for showing input data) and `show_results` (for showing model results).
	- Different types of model and datasets are able to use this consistent API because of `fastai`'s type dispatch system, a lower-level component we'll discuss later.
- The recommended way of training models is using a variant of the ==1cycle== policy, which uses a warm-up and annealing for the learning rate, while doing the opposite with the momentum parameter
![[Pasted image 20231124000527.png]]
- The ==learning rate== is the most important hyper-parameter to tune (and often the only one, since the library sets proper defaults for others).
- Other libraries often provide help for grid search or AutoML to guess the best value, but the `fastai` library implements the *==learning rate finder==*, which much-more-quickly provides the parameter, after a mock training.

The command `learn.lr_find()` will return a graph like this:
![[Pasted image 20231124000632.png]]
- Above:
	- The learning rate finder will do a mock training with an exponentially growing learning rate, over 100 iterations. A good value is then the minimum value on the graph divided by 10.

Another important high-level API component shared across all applications (CV, NLP, etc.) is the ***Data Block API***. 
This is an expressive, flexible API for data loading. It's the first attempt that the authors are aware of to systematically define all of the steps necessary to prepare data for a deep learning model, and give users a mix-and-match recipe book for combining these pieces (which we refer to as data blocks).

The steps that are defined in the Data Block API are:
- Getting the source items
- Splitting the items into the training set and one or more validation sets
- Labelling the items
- Processing the items (such as normalization)
- Optionally collating the items into batches

Here's an example:
```python
# Create our DataBlock
mnist = DataBlock(
	blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),  # From images, predict categories
	get_items=get_image_files,  # fn
	splitter=GrandparentSplitter(),  # obj
	get_y=parent_label  # fn
)

# Create a DataLoaders from our DataBlock by including the data
dls = mnist.databunch(untar_data(URLs.MNIST_TINY), batch_tfms=Normalize)

```

(In fastai v1, they used a "fluent" instead of a "functional" API for this, meaning hte statements to execute these steps would be chained one after another. The order mattered and people often messed it up.)

Here's a segmentation example of using DataBlocks:
```python
coco_source = untar_data(URLs.COCO_TINY)
images, lbl_bbox = get_annotations(coco_sources/'train.json')
lbl = dict(zip(images, lbl_bbox))

# Create our DataBlock for segmentation
# See explanation below code block for why there are three elements in the blocks argument
coco = DataBlock(
	blocks=(ImageBlock, BBoxBlock, BBoxLblBlock),  # TransformBlocks
	get_items=get_image_files,
	splitter=RandomSplitter(),
	getters=[noop, lambda o: lbl[o.name][0], lambda o: lbl[o.name][1]],  # noop is a fastai function that does nothing
	n_inp=1,  # number of inputs (the number of blocks that should be considered the input, with the reset forming the target)
)

# Create our Dataloaders from our DataBlock
# It seems like the transforms are added at the DataLoaders level of abstraction, wheres the DataBlocks are more about how to get/split the data? Not sure why that drawn line is obvious yet. And how the TransformBlocks in the datablock relate to the transforms that are here.
dls = coco.databunch(
	coco_source,
	item_tfms=Resize(128),
	batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
)

```
Above:
- In this case, the target are a tuple of two things: A list of bounding boxes *and* a list of labels (there's a cat *here*, and there's a dog *there*, and another dog over *here*). *THIS is why there are three blocks!*
	- We have to pass that n_inp thing; that specifies how many of the blocks should be considered the input, with the rest forming the target.

We could also build the modeling earlier using the DataBlock API:
```python
# Create the dataframe from the downloaded/uncompressed data
df = pd.read_csv(untar_data(URLs.IMDB_SAMPLE)/'texts.csv')

# Tokenize the dataframe
df_tok, count = tokenize_df(df, 'text')

# Create the DataBlock for lm (language model?)
imdb_lm = DataBlock(
	blocks=TextBlock(make_vocab(count),is_lm=True),
	get_x=attrgetter('text'),
	splitter=RandomSplitter()
)

# Create the DataLoaders by adding the dataset
dls = imdb_lm.databunch(df_tok, bs=64, seq_len=72)
```

Users find that the `DataBlock` API provides a good balance of conciseness and expressivity. 

##### Consistency across domains
- As the application examples have shown, the `fastai` library allows training a variety of kinds of application models, with a variety of kinds of datasets, all while using a very consistent API (less to learn!).

----------
# Mid-level APIs

- Many libraries, including `fastaiv1`, provided a high-level API to users, and a low-level API used internally for that functionality, but nothing in-between. This means that it's actually harder to create additional high-level functionality, because the low-level API becomes increasingly complicated and cluttered, and that users have to understand a large surface area of the low-level API in order to adapt it.


##### Learner
- As already noted, a library can provide more appropriate defaults and user-friendly behavior by ensuring that classes have all the information they need to make appropriate choices!
- The `DataLoaders` class brings together all the information necessary for creating the data required for modeling.
- The `Learner` class brings together all the information necessary for training a model based on the data! 
	- A PyTorch model
	- An optimizer (optional; appropriate default can be automatically selected)
	- A loss function (optional; appropriate default can be automatically selected)
	- A `DataLoaders` object
* The `Learner` is also responsible (along with `Optimizer`) for handling `fastai`'s *transfer learning* functionality!
* When creating a `Learner`, the user can also pass a *splitter*, which is a function that describes how to split the layers of a model into PyTorch parameter groups, which can then be frozen, trained with different learning rates, or more generally handled differently by an optimizer.
	* (Note this splitter this is different from the splitter used in a DataBlock/DataLoaders, which determines how to split up the data into testing and validation set(s))
- One area that we've found to be particularly sensitive in transfer learning is the handling of *batch-normalization* layers -- we learned that we should never freeze batch-normalization layers, and never turn off the updating of their moving average statistics -- therefore, by default, the `Learner` will *bypass* batch-normalization layers when a user asks to freeze some parameter groups. Users often report that this one minor tweak dramatically improves their model accuracy.

##### Two-Way Callbacks
- Customization points within the basic `fastai` training loop are called *two-way callbacks*. The `Learner` class's novel 2-way callback system allows gradients, data, losses, control flow, and anything else to be read and changed at *any point* during training!
	1. A callback should be available at every single point that code can be run during training, so that a user can be able to customize every single detail of the training method;
	2. Every callback should be able to access every piece of information available at that stage in the training loop, including hyper-parameters, losses, gradients, input and target data, and so forth;
	3. Every callback should be able to modify all of these pieces of information, at any time before they're used, and to be able to skip a batch, epoch, training or validation section, or cancel the whole training loop entirely!
		- This is why they're called 2-way callbacks, because information not only lows *from the training loop to the callback*, but *from the callback to the training loop* as well.


Here's the code for training a single batch `b` in `fastai`:
![[Pasted image 20231125001758.png]]

This shows how every step in the process is associated with some callback (the calls to self.cb(...)) and shows how exceptions are used as a flexible flow control mechanism for them.


##### Generic Optimizer
- `fastai` provides a new generic optimizer foundation that lets recent optimization techniques to be implemented in only a few handful of lines of code, by refactoring out the common functionality of modern optimizers into two basic pieces:
	1. *stats*: track and aggregate statistics like gradient moving averages
	2. *steppers*: combine stats and hyper-parameters to update weights using some function

- This has allowed us to implement every optimizer that we've attempted in fastai, without needing to extend/change this foundation. Cool!

For example, to support an optimizer that does decoupled weight decay (also known as AdamW):

```python
steppers = [weight_decay] if decouple_wd else [12_reg]
```

That's it! :) On the other hand, doing it in PyTorch requires creating an entirely new class, with over 50 lines of code. The benefit for research comes about because it makes it easy to rapidly implement new paper as they come out, recognize similarities and differences across techniques, and try out variants and combinations of these underlying differences, many of which have not yet been published.

##### Generalized metric API
- Nearly all ML and DL libraries provide some support for *metrics*. These are generally defined as simple functions  that takes the mean (or some other custom reduction function) across some measurement which is logged during training.
- In order to provide a flexible foundation to support whatever metrics you want, `fastai` provides a `Metric` abstract class which defines three methods: 
	1. `reset`
		- Called at the start of training
	2. `accumulate`
		- Called after each batch
	3. `value` (a property)
		- Called to calculate the final check

Here's an example of something called the *dice coefficient*: 
```python
class Dice(Metric):
	def __init__(self, axis=1): self.axis = axis
	def reset(self): self.inter, self.union = 0, 0
	def accumulate(self, learn):
		pred, targ = flatten_check(learn.pred.argmax(self.axis), learn.y)
		self.inter += (pre*targ).float().sum().item()
		self.union += (pre+targ).float().sum().item()

	@property
	def value(self): return 2. * self.inter/self.union if self.union > 0 else None
```
Note: The scikit learn (`sklearn`) library already provides a wide variety of useful metrics, so instead of reinventing them, `fastai` provides a simple wrapper function `skm_to_fastai`, which allows them to be used in `fastai`.  

##### fastai.data.external
- Many libraries have recently started integrating access to external datasets directly into their APIs.
- `fastai` builds on this trend by collecting a number of datasets (hosted by the AWS Public Dataset Program) in a single place, and makes them available through the `fastai.data.external` module.
- `fastai` will automatically download, extract, and cache these datasets when they're first used!

##### funcs_kwargs and DataLoader
- Once a user has their data available, they need to get it into a form that can be fed to a PyTorch model. The most common class used to feed models directly is the `DataLoader` class in PyTorch.
	- This class provides fast and reliable multi-threaded data-processing execution, with several points allowing customization.
- However the PyTorch `DataLoader` class isn't flexible enough to conveniently do everything that `fastai` wanted, so they implemented their own `DataLoader` class *on top* of the one that PyTorch uses, so that it has a more expressive front-end for the user.
- `DataLoader` provides 15 extension points via customizable methods, which can be replaced by the user as required. These customizable methods represent the ==15 stages of data loading== that we've identified, which fit into three broad stages:
	1. sample creation
	2. item creation
	3. batch creation
- The Python decorator `@funcs_kwargs` makes this possible -- it creates a class in which any method can be replaced by passing a new function to the constructor, or by replacing it through subclassing, so you can customize any part of your DataLoader class.
- `fastai` provides a DataLoader called `TfmdDL` ("transformed dataloader"), which subclasses `DataLoader`. In `TfmdDL`, the callbacks and customization points execute *Pipelines* of *Transforms*. Both mechanisms will be covered later.
	- A *Transform* is simply a Python function (which can include its inverse function (another function that "undoes" the transform)).
	- Transforms can be composed using the `Pipeline` class, which then allows the entire function composition to be inverted as well. These two directions (forwards, inverse) of the functions are referred to as the `Transform`'s `encodes` and `decodes` methods.


##### fastai.data.core
- When users who need to create a new kind of block for the Data Blocks API, or if they need a level of customization that the Data Blocks API itself doesn't support, they can use the mid-level components that the data block API is built on - there's a small number of simple classes.
- `TfmdLists` ("transformed list") *lazily applies a transform pipeline to a collection*, whilst providing a standard Python collection interface. This is an important foundational functionality for deep learning (the ability to index into a collection of filenames, and on-demand read some image file and apply processing to it)
	- It even provides subset functionality, allowing you to define subsets/slice of the data, like those representing the training and validation lets.

Another important class at this layer is called `Datasets` , which *applies multiple transform pipelines in parallel to a single collection*. Provides a standard Python collection interface. This is the class used by the data blocks API to return a tuple of image tensor(X), label for that image(y), both being derived from the same input filename.

##### Layers and Architectures

- PyTorch provides a basic "sequential" layer object, which cal be combined in sequence to form a component of the network. This represents a simple composition of functions, where each layer's output is the next layer's input!
	- However, many components in *modern* network architectures (eg ResNet blocks, skip connections) are not compatible with simple sequential layers.
	- The normal work-around for this in PyTorch is to write a custom forward function, effectively relying on the full-flexibility of Python to escape the limits of composing these sequence layers.
		- The downside of this is that the model is no longer amenable to easy analysis and modification (eg removing the last few layers in order to do transfer learning), because you're writing custom Python code.

Therefore, `fastai` attempts to provide the basic foundations to allow *modern* NN architectures to be built, by a small number of stacking predefined blocks:

The first piece of this system is the `SequentialEx` layer
- This has the same basic API as PyTorch's `nm.Sequential` , but with one key difference:
	- The original input value to the function is available to *every layer in the block*
	- Therefore, the user can, for instance, include a layer which adds the current value of the sequential block to the original input value of the sequential block (as is done in a ResNet)

To take full advantage of this capability, `fastai` also provides a `MergeLayer` class. This allows the user to pass *any function*, which will in turn be provided with both the layer block input value, and the current value of the sequential block.
- For instance, if you pass the simple add function, then `MergeLayer` provides the functionality of an *identity connection* in a standard ResNet block.
- If the user passes a concatenation function, then it provides the basic functionality of a *concatenating connection* in a DenseNet block.

In this way, `fastai` provides primitives which allow representing modern network architecture out of predefined building blocks, without falling back to totally unstructured Python code in the forward function.

`fastai` also provides a general-purpose class for combining these layers into a wide range of modern *convolutional* neural network architectures.
- These are often built on the underlying foundations from ResNet, and therefore this class is called `XResNet`. By providing parameters tot his class, the user can customize it to create architectures that include squeeze and excitation blocks, grouped convolutions like in ResNext, depth-wise convolutions such as in the Xception architecture, widening factors such as in Wide ResNets, *self-attention* and symmetric self-attention functionality, custom activation functions, and more!

------

# Low-Level APIs

- The layered approach of the `fastai` library has a very specific meaning at the lower levels of its stack. Rather than treating *Python itself* as the base layer of the computation, which the middle layer relies on, those middle layers still rely on a set of basic abstractions provided by the lower layer.

The low-level of the `fastai` stack provides a set of abstractions for:
- Pipelines of transforms
	- Partially-reversible composed functions mapped and dispatched over elements of tuples
- Type-dispatch based on the needs of data processing pipelines
- Attaching semantics to tensor objects, and ensuring that these semantics are maintained throughout a `Pipeline`
- GPU-optimized computer vision optimizations.
- Convenience functionality, like a decorator to make patching existing objects easier, and a general collection class with a NumPy-like API.

#### PyTorch Foundations
- The main foundation for `fastai` is the `PyTorch` library. PyTorch provides a GPU-optimized tensor class, a library of uesful model layers, classes for optimizing models, and a flexible programming model that integrates these elements.
- `fastai` uses building blocks from all parts of the PyTorch library, including directly patching its tensor class, entirely replacing its library of optimizers, providing simplified mechanisms for using its hooks, and so forth.
- `fastai` builds on other libraries too:
	- `Python Imaging Library (PIL)` is used and extended for CPU image processing
	- `pandas` is used for reading and processing tabular data
	- `Scikit-Learn` is used for most of its metrics
	- `Matplotlib` is used for plotting.

##### Transforms and Pipelines
- One key motivation is the need to often be able to UNDO some subset of transformations that are applied to create the data used to do modeling.
	- Ex:  The strings that represent categories cannot be used in models directly, and are turned into integers using some *vocabulary*.
	- Ex: Pixel values for images are generally normalized.
	- Neither of these can be directly visualized, and so at inference time we need to apply the INVERSE of these functions to get data that's understandable.
- `fastai` introduces a `Transform` class, which provides callable objects, along with a *decode* method.
	- this decode method is designed to *invert* the function provided by a transform; it needs to be implemented manually by the user.
	- By having both the *encode* and *decode* methods in a single place, the user ends up with a single object which they can compose into pipelines, serialize, and so forth.
- In addition, Sometimes transforms need to be able to opt out of processing altogether, depending on context:
	- In test-time augmentation, data augmentation methods (rotating, fuzzifying) should NOT be applied to the validation set in the way that they were applied to the training set!
		- Therefore, `fastai` provides the current subset index to transforms, allowing them to modify their behavior based on subset (eg for training vs validation).

- Transforms in deep learning pipelines often require state, which can be dependent on the input data
	- Eg: normalization statistics could be based on a sample of data batches
	- Eg: A categorization transform could get its vocabulary directly from the dependent variable
	- Eg: An NLP numericalization transform could get its vocabulary from the tokens used in the input corpus
- Therefore, `fastai` transforms and pipelines support a *setup* method, which can be used to create this state when setting up a `Pipeline` . When pipelines are set up, all previous transforms in the pipeline run first, so that the transform being set up receives the same structure of data that it will when being called.
- This is closely connected to the implementation of `TfmdList` ...

##### Type dispatch
- The `fastai` type dispatch system is like the `functools.singledispatch` system provided in the Python standard library while supporting multiple dispatch over two parameters.
	- Dispatch over two parameters is necessary for any situation where the user wants to be able to customize behavior based on both the input and target of a model. For instance, `fastai` uses this for the `show_batch` and `show_results` methods; these methods automatically provide an appropriate visualization of the input, target, and results of a model, which requires responding to the types of *both* parameters.
- It also provides a more expressive and yet concise syntax for registering additional dispatched functions or methods, taking advantage of Python's recently introduced type annotations syntax! Here's an example of creating two different methods which dispatch based on parameter types:
```python
def f_td_test(x: numbers.Integral, y): return x+1

@typedispatch
def f_td_test(x: int, y: float): return x+y
```
Here, f_td_test has a generic implementation for x of numeric types and all ys, and then there's a specialized implementation when x is an int, and y is a float.

##### Object-oriented semantic tensors
- By using `fastai`'s transform pipeline functionality, which depends heavily on types, the mid and hihg-level APIs can provide a lot of power, conciseness, and expressivity for users.
- But this doesn't work well with PyTorch's basic tensor type, becasue it doesn't have any subclasses which can be used for type dispatch, and subclassing PyTorch tensors is challenging.
- Therefore, fastai provides a new tensor base class which can be easily instantiated and subclassed. `fastai` also *patches* PyTorch's tensor class to attempt to maintain subclass information through operations whenever possible (this isn't always perfectly done).


##### GPU-accelerated augmentation
- The `fastai` library provides most data augmentation in CV on the GPU at the batch level. 
- Historically, the processing pipeline in CV has always been to open the images and apply data augmentation on the CPU, using a library like PIL or OpenCV, and then batch the results before transferring them to the GPU and using them to train the model.
- On modern GPUs, however, architectures like a standard ResNet-50 are often CPU-bound; Therefore `fastai` implements most common functions on the GPU.
- More data augmentations are random affine transformations (rotation, zoom, translation, etc) functions on a coordinates map (perspective warping), or easy functions applied to the pixels themselves (contrast or brightness changes), all of which can be easily parallelized and applied to a batch of images.
	- In `fastai` we combine all affine and coordinate transforms in one step to apply only one interpolation, which results in a smoother results.
	- Most other vision libraries don't do this and lose a lot of detail of the original image when applying several transformations in a row.

#### Convenience Functionality
- There's some other additions to make Python easier to use, including a NumPy-like API for lists called `L`, and some decorators to make delegation or patching easier.
	- Delegation is used when one function will call another and send it a bunch of keyword arguments with defaults. To avoid repeating those, they're often grouped into \*\*kwargs; the problem with this is that disappear from the signature of the function that delegates, and you can't use the tools from modern IDEs to get tab-completion for those delegated arguments or see them in its signature.
	- To solve this, `fastai` provides a decorator called @delegates that will analyze the signature of the delegated function to change the signature of the original function.
For example, the initialization for Learner has 11 keyword-arguments, so any function that creates a Learner uses this decorator to avoid mentioning them all. As an example the function `tabular_learner` (which creates a Learner) is defined like this:
```python
@deletgates(Learner.__init__)
def tabular_learner(dls, layers, emb_szs=None, config=None, **kwargs)
```
But when you look at its signature, you'll see the 11 additional arguments of `Learner.__init__` with their defaults!


* Monkey-patching is an important functionality of the Python language when you want to add functionality to existing objects. `fastai` provides a `@patch` decorator, using Python's type-annotation system to make this easier.
* Here's us adding the `write()` method to the pathlib.Path class:
```python
@patch
def write(self: Path, txt, encoding='utf8'):
	self.parent.mkdir(parents=True, exist_ok=True)
	with self.open('w', encoding=encoding) as f:
		f.write(txt)
```


Lastly, inspired by the NumPy library, `fastai` provides a collection type called `L` that supports fancy indexing and has a lot of methods that allow users to write simple, expressive code:
```python
d = dict(a=1, b=-5, e=9).items()
L(d).itemgot(1).map(abs).filter(gt(4)).sum()
```
Above: This takes a list of pairs, selects the second item of each pari, takes its absolute value, filters items greater than 4, and adds them up.

#### nbdev
- In order to assist developing this library, we built a programming environment called `nbdev` , which allows programmers to create complete Python packages, including tests and a rich documentation system, all in Jupyter Notebooks!
- `nbdev` is a system for *exploratory programming*, adding critically important tools for software development:
	- Python modules are automatically created, following best practices like automatically defining __all__ with exported functions, classes, and variables
	- Navigate and edit code in a standard text editor or IDE, and export any changes automatically back into your notebooks.
	- Automatically create searchable, hyperlinked documentation from your code
	- Pip installers
	- Testing (defined in notebooks, run in parallel)
	- Continuous integration
	- Version control conflict handling









