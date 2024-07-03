https://docs.fast.ai/quick_start.html

```python
# These are the highest-level "Applications"-layer modules in fastai
from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *
```

It's nice that `fastai` applications all use the *same basic steps and code!*
1. Create appropriate `DataLoaders`
2. Create a `Learner` ((Combines the data with a model, along with some other configuration))
3. Call a *fit* method ((There are multiple))
4. Make predictions or view results

### Computer Vision Classification

The following code does these things:
1. Downloads a dataset called the `Oxford-IIIT Pet Dataset`, which contains 7,349 images of cats and dogs from 37 different breeds.
2. A *pretrained model* that's already been trained on 1.3 million images, using a competition-winning model is downloaded from the internet ((its weight are downloaded))
3. The pretrained model gets *fine-tuned* using the latest advances in ==transfer learning== to create a model that's *specially customized* for our task of recognizing dogs and cats!


```python
# The untar_data function Downloads the url using FastDownload.get
# It downloads and extracts the data (by default to ~/.fastai) and then returns the path to the extracted data.
# Note that URLs is a global constant for dataset and model URLs
path = untar_data(URLs.PETS)/'images'

# A function to be used as a "labeling func" for our dataset
def is_cat(x):
	return x[0].isupper()

# Create the DataLoader, which points to a dataset and gives information about how to extract labels, batches, etc. from it
# Note use of ImageDataLoaders
dls = ImageDataLoaders.from_name_func(
	path,
	get_image_files(path),
	valid_pct=.2,  # 20% of data used as a validation set
	seed=42,  # random seet 
	label_func=is_cat,  # labeling func
	item_tfms=Resize(224),  # Transformation fn (224x224's image?)
)

# Create our Learner using `vision_learner(...)`
# Pass the DataLoader, a Learner, and some other config
learn = vision_learner(dls, resnet34, metrics=error_rate)

# Fine tune the (pretrained) model on our specific dataset
learn.fine_tune(1)

...

# Do some inference with our model using the predict(..) method
img = PILImage.create('images/cat.jpg')
img
# Shows cat

is_cat, _, probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}")
# :.6f rounds it to 6 decimal places. There's a lot of float formatting options though!
print(f"Probability it's a cat: {probs[1].item():.6f}")

```

### Computer vision segmentation

Here's how we can change a segmentation model with `fastai` using a subset of the *Camvid* dataset (which is  Cambridge-driving Labeled Video Database, a collection of driving videos (pictures?) with labels):

```python
# Download the dataset and return the path to it
path = untar_data(URLs.CAMVID_TINY)

# Crate our DataLoader
# Note use of SegmentationDataLoaders
dls = SegmentationDAtaLoaders.from_label_func(
	path,  # Path to downloaded dataset?
	bs=8,  # Batch size
	fnames=get_image_files(path/"images"), # assumedly filenames?
	label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
	codes = np.loadtxt(path/'codes.txt', dtype=str)  
)

# Create a Learner using `unet_learner`
learn = unet_learner(dls, resnet34)
# Fine-tune the dataset 
learn.fine_tune(8)

# We can now even visualize how well it achieved its task, by asking the model to color0code each pixel of an image!
# We use the learn.show_results function ((Curious if this works with other types of learners)):
learn.show_results(max_n=6, figsize=(7,8))
# (Shows some images that are color-coded)

# Or we can plot the k instances that contributed the MOST to the validation loss (the ones we performed the worst on) using the SegmentationInterpretation class:
interp = SegmentationInterpretation.from_learner(learn)
interp.plot_top_losses(k=2)
# Shows k pictures that we didn't do very well on

```

------

### Natural Language Processing

Here is all of the code that's necessary to train a model that can classify the *sentiment* of a movie review better than ANYTHING that existed in the world, even five years ago!

```python

# Create our DataLoader
# See that we're using a TextDataLoader, with the from_folder constructor, rather than the from_name_func or from_label_func constructors we used earlier.
dls = TextDataLoaders.from_folder(
	untar_data(URLs.IMDB),
	valid="test"
)

# Create our Learner using text_classifier_learner (data+model+config)
learn = text_classifier_learner(
	dls, AWD_LSTM, drop_mult=.5, metrics=accuracy
)

# Fine tune the (pretrained) learner on our data
learn.fine_tune(2, 1e-2)

# Now that we have a trained learner, we can use our predict(...) method on it to evaluate some text!
learn.predict("I really liked that movie!")
# ('pos', tensor(1), tensor([0.0041, 0.959]))

```

---------
### Tabular Data

Building models from plain *tabular* data is done using the same basic steps as the previous models. 
Here's the code to train a model to predict whether a person is a high-income earner, based on their socioeconomic background!

```python
# Download the Adult Sample dataset and get the path to it
path = untar_data(URLs.ADULT_SAMPLE)

# Create a DataLoader using TabularDataLoaders.from_csv(...)
dls = TabularDataLoadres.from_csv(
	path/'adult.csv',
	path=path,
	y_names="salary",  # I think this is the name of the y column
	cat_names=["workclass", "education", "marital-status", "relationship", "race"],  # These are the categorical predictors
	cont_names = ["age", "fnlwgt", "education-num"],  # These are the continuous predictors
	procs=[Categorify, FillMissing, Normalize]  # Transformations?
)

# Create a Learner using tabular_learner(..)
# This learner does NOT include a pretrained model, because your tabular data universe is meaningfully different from someone else's tabular data
learn = tabular_learner(dls, metrics=accuracy)

# Train the model! (Training statistics are output)
learn.fit_one_cycle(2)

```

----
### Recommendation Systmes

Here's how to train a model that will predict movies that people might like, based on their previous viewing habits, using the *MovieLens* dataset.

```python
# Download the datset and get the url
path = untar_data(URLs.ML_SAMPLE)

# Create the DataLoader for collaborative filtering using the from.csv(...) constructor
dls = CollabDataLoaders.from_csv(path/'ratings.csv')

# Create the Learner
# I believe in this case the actual Y values we want to predict are in the 1-5 range, but there's a reason why we actually want to set the y ranges as a little below/above.
learn = collab_learner(dls, y_range=(.5, 5.5))

# "Fine tune" the model (interesting that we can use the term fine_tune here, because this isn't a pretrained model)
learn.fine_tune(6)

# We can use the same show_results call from earlier to see a view examples of user and movie IDs, actual ratings, and predictions
learn.show_results()
```
![[Pasted image 20231120232303.png]]








