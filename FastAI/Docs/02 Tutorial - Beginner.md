https://docs.fast.ai/tutorial.vision.html

----
### Computer Vision

```python
from fastai.vision.all import *
```

For this task, we'll use the `Oxford-IIIT Pet Dataset` that contains images of cats and dogs of 37 different breeds. 😺 🐶

```python
# Download and decompress the dataset with one line of code
# It will only do this download once, and return the location of the decompressed archive. We can check what's inside with the .ls() method
path = untar_data(URLs.PETS)
path.ls()
# (#2) [Path('/home/jhoward/.fastai/data/oxford-iiit-pet/images'),Path('/home/jhoward/.fastai/data/oxford-iiit-pet/annotations')]

# We can see that it came with both an images and annotations folder
# Let's ignore the annotations folder for now, and focus on the images one...
# get_image_files is a fastai function that helps us grab all of the IMAGE FILES (recursively) in a folder:
files = get_image_files(path/"images")  # See overloaded magic / usage
len(files)
# 7390

```

We want to label our data for this cats vs dogs problem, so we need to know which filenames are of dog pictures and which are of cat pictures!
It *just so happens* in this dataset that an easy way to distinguish is that the name of the file begins with a capital for the Cats and a lowercase for dogs.

```python
files[0], files[1]
# (Path('/home/jhoward/.fastai/data/oxford-iiit-pet/images/basset_hound_181.jpg'), Path('/home/jhoward/.fastai/data/oxford-iiit-pet/images/beagle_128.jpg'))

# We can then define an easy function to determine the label:
def label_func(f):
	# Is the first chracter of the filename capitalized?
	return f[0].is_upper()
```

Now to actually get the data ready for modeling, we need to put it into a `DataLoader` object!

Given that we have a function that labels our data using the file names, we can use the `ImageDataLoaders.from_name_func`! There are other factory methods that *could* be more useful for another problem, so check them out in `vision.data` .

```python
dls = ImageDataLoaders.frmo_name_func(
	path, # Path to dataset (images + annotations)
	files,  # The image files
	label_func,  # fn that etermines correct label (based on filename)
	item_tfms=Resize(224)  # item transformations
)

# We can then check if everything looks okay with the show_batch method (TRue is for cat, False is for dog)
dls.show_batch()
# Shows a bunch of cats and dogs

# Create a Learner (an object combining data + model for training)
learn = vision_learner(dls, resnet34, metrics=error_rate)

# Fine tune the pretrained model on our unique data
learn.fine_tune(1)

# To make predictions, use the predict(...) method (passing a file)
learn.predict(files[0])
# ('False', TensorImage(0), TensorImage([9.9998e-01, 2.0999e-05]))
# Above: These are:
	# The decoded prediction (False for dog, here)
	# The index of the predicted class
	# The tensor of probabilities of all classes in the order of their indexed labels
```

Note: The `learner.predict(...)` method accepts any of (in this case):
1. A filename
2. a PIL image  (Note `PIL` = "Python imaging library", now known as the Pillow library, a library for opening/manipulating/saving many different image file formats. fastai's `Image` class is based on Pillow's `Image` class.)
3. A tensor, directly


#### Now that we've done dog vs cat, let's try to classify dog breeds! :) 

In order to label our data with the breed name, we'll use a regular expressoin to extract it from the filefname! Looking back at ta filename, we have:
```python
files[0].name
# great_pyrenees_173.jpg

# Regex pattern to extract the ground (.*) that precedes _#.jpg
pat = r'^(.*)_\d+.jpg)  # A lot of *s followed by a _ , number, .jpg

# Create an ImageDataLoader using the from_name_re constructor
dls = ImageDataLoaders.from_name_re(
	path,  # Path to dataset
	files,  # Image files
	pat,  # Regex to get image labels
	item_tfms=Resize(224)  # Image transforms
)

# Like before, we can use show_batch to look at our data
dls.show_batch()
# Cats and dogs

# Since classifying the exasct breed of cats or dogs amongst 37 different breeds is a harder problem, we'll slightly change the definition of our DataLoaders to use DATA AUGMENTATION!

dls = ImageDataLoaders.from_name_nre(
	path,
	files, # path to files
	pat,  # regex to find the response variable (the breed)
	item_tfms=Resize(224),
	batch_tfms=aug_transforms(size=224)  # New! BATCH Transforms
)
# Above: this aug_transforms is a function that provides a collection of data augmentation transforms with defaults that the fastai crew found performed well on many datasets. You can customize these transforms by passing arguments. ((Guessing this does things like fuzz, invert the images, etc?))

# Now we can craete our Learner
learn = vision_learner(dls, resnet34, metrics=error_rate)

# We've been using the default learning rate before, but if we want to find the best one possible, we can use the LEARNING RATE FINDER using learner.lr_find():
learn.lr_find()
# SuggestedLRs(lr_min=0.010000000149011612, lr_steep=0.0063095735386013985)
# Plot of the graph of the learning rate versus...loss?
# It gives us two suggestions:
	# Minimum divided by 10
	# Steepest gradient

# Let's use 3e-3 here, and do some more epochs to fine-tune our model
# This (re)trains our model on the dataset in the DataLoader passed to the Learner. So somewhere in the learner are the results of the latest predictions?
learn.fine_tune(2, 3e-3)

# Now we can look at some of the predictions with:
learn.show_results()
# Shows a bunch of images with the ground truth + predicted categories (red if they're wrong).

# We can also create an INTERPRETATION OBJECT, which can show us where the model made the WORST predictions:
interp = Interpretation.from_learner(learn)

interp.plot_top_losses(9, figsize(15,10))
# Shows the 9 trickiest images


```

#### Single-label classification using the Data Block API
- So we've been just creating our DataLoaders directly using secondary constructor functions, but we can also create them by using the slightly lower-level `DataBlock` API! 
- A `Datablock` is built by giving the fastai library a bunch of information:
	1. The types used, through an argument called `blocks`
		-  Here we have images and categories, so we pass `ImageBlock` and `CategoryBlock` 
	2. How to ***get*** the raw items
		- Here, our function `get_image_files`
	3. How to ***label*** those items
		- Here, with the same regular expression as before
	4. How to **split** those items, here with a random splitter
	5. Any transformations
		1. The `itemtfms` and `batch_tfms` like before

```python

pets = Datablock(
	blocks=(ImageBlock, CategoryBlock),  # Mapping from image to ctg.
	get_items=get_image_files,  # How to get image files
	splitter=RandomSplitter(),  # How to split data into train/test
	get_y=using_attr(RegexLabeller(r'(.*)_\d+.jpg)'), 'name'),  # How to get the label of each item
	item_tfms=Resize(460),  # per-item transforms
	batch_tfms=aug_transforms(size=224)  # per-batch transforms
)

# At this point, our `pets` Datablock object is EMPTY! It only contains the _functions_ that will help us gather the data! We have to call the .dataloaders(...) method to get a DataLoaders by passing hte source of the data!

dls = pets.dataloaders(untar_data(URLs.PETS)/"images")

# And now that we have a dataloader, we can view a batch of our data:
dls.show_batch(max_n=9)

```

#### Multi-Label Classification
- For this task, we'll use the Pascal Dataset, containing images with different kinds of objects/persons. It was originally a dataset for object detection, meaning that the task is not only to detect if there is an instance of one class of an image, but to also draw a ==bounding box== around it! Here, we'll just try to predict all the classes in a given image.
- ==Multi-label classification== differs from before, because an image could have a person *and* a horse inside of it, for instance! (Or have none of the categories)

```python
# Download and decompress the dataset
path = untar_data(URLs.PASCAL_2007)
path.ls()
# (#9) [Path('/home/jhoward/.fastai/data/pascal_2007/valid.json'),Path('/home/jhoward/.fastai/data/pascal_2007/test.json'),Path('/home/jhoward/.fastai/data/pascal_2007/test'),Path('/home/jhoward/.fastai/data/pascal_2007/train.json'),Path('/home/jhoward/.fastai/data/pascal_2007/test.csv'),Path('/home/jhoward/.fastai/data/pascal_2007/models'),Path('/home/jhoward/.fastai/data/pascal_2007/segmentation'),Path('/home/jhoward/.fastai/data/pascal_2007/train.csv'),Path('/home/jhoward/.fastai/data/pascal_2007/train')]

# Above: Can see that we've got a lot of files. The information about the LABELS of each image is in the file named train.csv -- we can load it using pandas:

# Create a pandas dataframe from the csv training dataset
df = pd.read_csv(path/'train.csv')
df.head()
# The first few rows of the dataframe
```
![[Pasted image 20231121004323.png]]

##### Multi-label classification using the high-level API:
* (In the table above): For each filename, we get the different labels (separated by a space) and then the last column tells us whether it's in the validation set or not.
* To ingest this into `DataLoaders` quickly, we have a factory method, `from_df`, that can specify
	* the underlying path where all the images are
	* an additional folder to add between the base path and the filenames (here `train`)
	* The `valid_col` to consider for the validation set (if we don't specify this, we take a random subset)
	* A `label_delim` to split the labels, and, as before, `item_tfms` and `batch_tfms`

Note that we don't have to specify the `fn_col` and `label_col` because *they default to the first and second column, respectively*. 

```python
dls = ImageDataLoaders.from_df(
	df,  # the dataframe of data
	path, # the path to the directory with the dataset
	folder="train",  # An additional folder to add betwixt base path  and the filenames. This one is a folder containing the actual images
	valid_col="is_valid",  # The validation set col (bool col)
	label_delim=' ', # The delimter used for the labels (assumed as column 2)
)

dls.show_batch()
# Shows some images
```

Training a model is as easy as before; we can use the same functions and the `fastai` library will *automatically detect* that we are in a multi-label problem, thus picking the right loss function.

The only difference is in the *metric* that we pass:
- `error_rate` ==will not work== for a multi-label problem, but we can instead use `accuracy_thresh` and `F1ScoreMulti` . We can also change the default name for a metric, for instance -- we might want to see **F1 scores** with `macro` and `samples` averaging.

```python
f1_macro = F1ScoreMulti(thresh=.5, average='macro')
f1_macro.name = 'F1(macro)'
f1_samples = F1ScoreMulti(thresh.5, average='samples')
f1_samples.name = 'F1(samples)'

learn = vision_learner(
	dls,  # our dataloader
	resnet50,  # our chosen model
	metrics=[partial(accuracy_multi, thresh=.5), f1_macro, f1_samples]  # an iterable of metrics to be reported at each epoch in training
)

# As before, we can use learn.lr_find(...) to pick a good learning rate:
learn.lr_find()
# SuggestedLRs(lr_min=0.025118863582611083, lr_steep=0.03981071710586548)

# So let's use 3e-2, or .03.
learn.fine_tune(2, 3e-2)
# Training output

# Like before, let's look at the results
learn.show_results()
# This shows a 3x3 matrix of the actual predicted labels nad the ground truth label.

# Or get the predictions for a given image
learn.predict(path/'train/000005.jpg')
# ((#2) ['chair','diningtable'], TensorImage([False, False, False, False, False, False, False, False,  True, False,
          True, False, False, False, False, False, False, False, False, False# TensorImage([1.6750e-03, 5.3663e-03, 1.6378e-03, 2.2269e-03, 5.8645e-02, 6.3422e-03, 5.6991e-03, 1.3682e-02, 8.6864e-01, 9.7093e-04, 6.4747e-01, 4.1217e-03, 1.2410e-03, 2.9412e-03, 4.7769e-01, 9.9664e-02, 4.5190e-04, 6.3532e-02, 6.4487e-03, 1.6339e-01]))

# As for the single classification predictions, we get three things: 
1. The decoded, readable classification
2. One-hot encoded targets (True for all predicted classes, meaning those that get a probability > .5)
3. The prediction of the model on each class (going from 0 to 1)


# Like with before, we can check where the trained model _did its worst_ on the training data by creatinga n Interpretation object
interp = Interpretation.from_learner(learn)
interp.plot_top_losses(9)
# Shows a table with columns target label, predicted label, probabilities (of each cat), loss function

```

### Multi-label classification using the DataBlock API
- So we just did an example using the DataLoader API; this is another one using the slightly-lower-level DataBlock API!

```python
df.head()
# Table with columns fname (filename), labels (true labels), is_valid (is in the validation set)

# Create our DataBlock

pascal = DataBlock(
	blocks=[ImageBlock, MultiCategoryBlock],  # From images, we're predicting (multiple) categories
	splitter=ColSplitter('is_valid'),  # Object to determine how to split data between the training and validation sets
	get_x=ColReader('fname', pref=str(path/'train')+os.path.sep), # How to get X
	get_y=ColReader('labels', label_delim=' '),  # How to get y (targets)
	item_tfms=Resize(460),  # Item transformations [resize]
	batch_tfms=aug_transformers(size=224) # Batch transformations
)
```
Above: This block is slightly different than before!
- Before, we passed a get_items function that actually gets the images. Here, we instead pass a get_x function that preprocesses rows of our dataframe in order to get images.

Our `DataBlock` is still just a *blueprint*, or a "plan" for how to load data from a dataset! 📝
We need to actually pass it the source of our data in order to get an actual `DataLoader`! ✨

```python
# Create our DataLoader by calling the .dataloaders(...) method on our DataBlock. Interesting that we're passing a dataframe here, whereas before we passed whatever is returned from untar_data.
dls = pascal.dataloaders(df)

# What would a (part of a) batch look like?
dls.show_batch(max_n=9)
# Shows some pictures


```

### Segmentation
- ==Segmentation== is a problem where we have to *predict a category for each pixel of the image*! 
- We'll use the Camvid dataset, a dataset of screenshots from car cams. Each pixel of the image has a label such as road, car, or pedestrian.

```python
# Download the data with the untar_data function
path = untar_data(URLs.CAMVID_TINY)
path.ls()
# (#3)[Path('/home/jhoward/.fastai/data/camvid_tiny/codes.txt'),Path('/home/jhoward/.fastai/data/camvid_tiny/images'),Path('/home/jhoward/.fastai/data/camvid_tiny/labels')]

# The `images` folder contains the images, and the corresponding segmentation masks of labels are in the `labels` folder. The `codes` file contains the corresponding integer to class (the masks have an int value for each pixel)

codes = np.loadtxt(path/'codes.txt', dtype=str)
codes
# array(["Animal", "Archway", "Bicyclist", "Bridge", ...])
```

#### Segmentation using the high-level API
- As before, we can use the `get_image_files` function to help us grab the image filenames:

(Recall: We've fetched and uncompressed the CAMVID driving images, with path)
```python
# Using our `path` variable that's a Path to the dataset directory, let's pull out one of the image paths
fnames = get_image_files(path/"images")
fnames[0] 
#Path('/home/jhoward/.fastai/data/camvid_tiny/images/0006R0_f02910.png')

# Let's look at the label folder in the `path` directory:
# recall: The labels folder contain "images" that are "Segmentation Masks", basically where every pixel has a category
(path/"labels").ls()[0]
#Path('/home/jhoward/.fastai/data/camvid_tiny/labels/0016E5_08137_P.png')

# It seems like these segmentation masks have the same base name as the images, but they have an extra _P at the end (0006R0_f02910 vs 0006R0_f02910_P), so we can define a label function!

def label_func(fn):  # What's this fn? I think it's "file name"
	return path/"labels"/f"{fn.stem}_P{fn.suffix}"

# We can then gather our data using SegmentationDataLoaders:
dls = SegmentationDataLoaders.from_label_func(
	path,  # Path to dataset
	bs=8,  # ... batches?
	fnames = fnames  # Path to images
	label_func = # Func return path to targets (segmentation masks)
	codes = codes  # Path to integer-to-class
)

# A traditional CNN doesn't work for segmentation, so we have to use a special kind of model called a UNet, so we use unet_learner to define our Learner:
learn = unet_learner(dls, resnet34)
# Train for 6 epochs
learn.fine_tune(6)

# We can get an idea of the predicted results with show_results, which shows the input image (no transforms), predicted label, and true label (In this segmentation case, it just shows two segmentation masks; predicted and acutal)
learn.show_results(max_n=6, figsiz=(7,8))


# We can also sort the model's errors on the validation set using the SEgmentationInterpretation class, and then plot the instances with the `k` highest contributions to the validation loss!
interp = SegmentationInterpretation.from_learner(learn)
interp.plot_top_losses(k=3)
# Shows input, target, and predicted+loss for the worst ones

```

##### Segmentation using the Data Block API
- We can also use the dat block API to get our data into DataLoaders! 

```python
camvid = DataBlock(
	blocks=(ImageBlock, MaskBlock(codes)),  # From images, predicting a mask using these codes
	get_items=get_image_files,  # Get X
	get_y=label_func  # Get y
	splitter=RandomSplitter(),  # train/validation split
	batch_tfms=aug_transforms(size(120,160))  # A collection of predefined, well-performing transforms to be applied per-batch using the GPU
)

# now create the DataLoaders object by combining the DataBlock with the actual dataset, plus addition info
dls = camvid.dataloaders(path/"images", path=path, bs=8)

# Show a batch of data that'll be served up!
dls.show_batch(max_n=6)
```

##### Points
- Let's look at a task where we want to predict points in a picture!
- For this, we'll use the Biwi Kinect Head Pose Dataset
	- (This is an image of something like a mesh that looks like someone's upper torso with their head turned in various ways)

```python
# Download and uncompress the data, eturning hte path
path = untar_data(URLS.BIWI_HEAD_POSE)

# Let's see what we've got!
path.ls()
# (#50) [Path('/home/sgugger/.fastai/data/biwi_head_pose/01.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/18.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/04'),Path('/home/sgugger/.fastai/data/biwi_head_pose/10.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/24'),Path('/home/sgugger/.fastai/data/biwi_head_pose/14.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/20.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/11.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/02.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/07')...]

# Seems like we've got a punch of head pose objects... There are 24 directories number 01 to 24 that correspond to different people photographed, and a corresponding .obj file (we don't need them here).
# Look inside a directory
(path/'/01').ls()
# (#1000) [Path('01/frame_00087_pose.txt'),Path('01/frame_00079_pose.txt'),Path('01/frame_00114_pose.txt'),Path('01/frame_00084_rgb.jpg'),Path('01/frame_00433_pose.txt'),Path('01/frame_00323_rgb.jpg'),Path('01/frame_00428_rgb.jpg'),Path('01/frame_00373_pose.txt'),Path('01/frame_00188_rgb.jpg'),Path('01/frame_00354_rgb.jpg')...]

# Inside the subdirectories, we have different frames, each of them coming with an image (\_rgb.jpg) and a pose file (\_post.txt)
# We can easily get all hte files recursively with get_image_files, then write a function that converts an image filename eto its associated pose file.
img_files = get_image_files(path)

# Given an image filename, convert to associated pose file
def img2pose(x): return Path(f'{str(x)[:-7]pose.txt})

# Let's now have a look at our image:
im = PILImage.create(img_files[0])
im.shape
# 480,640

im.to_thumb(160)
# a picture of someone looking off to the right
```

```python
# The post text file associated with each image shows the location of the center of the head. The details of this aren't important for our purposes, so we'll just show the function we use to extract the head center point:
cal = np.genfromtext(path/'01'/'rgb.cal', skip_footer=6)
def get_ctr(f):
	ctr = np.genfromtxt(img2pose(f), skip_header=3)
	c1 = ctr[0] * cal[0][0]/ctr[2] + cal[0][2]
	c2 = ctr[1] * cal[1][1]/ctr[2] + cal[1][2]

# Above: Not important; it returns the coordinates as a tensor of two items:
get_ctr(img_files[0])

# We can pass this function to DataBlock as get_y, since it's responsible for labeling each item. We'll resize the images to half their input size, just to speed up training a bit.

# An important point is that we DON'T use a random splitter. this is becaues the same person appears in multiple images in this dataset, but we want to ensure htat our model gan generalize to people that it HASN'T seen yet; Each folder contains the images for one personl Therefore, we can create a splitter functino which returns True for just ONE person, resulting in a validation set containing just that person's images.

# the only other difference to previous DataBlock examples is that the second block is known as a PointBlock; this is necessary so that fastai knows that the labels represent coordinates; that way, it knows that when doing data augmentation, it should do the same augmentation to these coordinates that it does to the images!

biwi = DataBlocm(
	blocks=(ImageBlock, PointBlock),
	get_items=get_image_files,
	get_y=get_ctr,
	splitter=FuncSplitter(lambda o: o.parent.name=="13"),  # A custom splitter. I think this one is just saying "Use the folder with 13"
	batch_tfms[*aug_transforms(size=240,320)), Normalize.from_stats(*imagenet_stats)]  # Not sure what this does yet :)
)

# Create our DataLoaders by combinign our DataBlock with the datset, plus some additional information
# really, it's not hte dataset. It's the entry point to the dataset, and then teh data block's get_items/get_y fetch the X and y from the entrypoint (Path) to the dataset
dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8,6))

# Now that we've assembled our data, we can use hte reset of the fastai API as usual!
# Create a learner by providing a model architecture, our data loader, and some information on the y range (determining the head of the architecture)
learner = vision_learner(dls, resnet18, y_range=(-1,1))

# Find the optimal learning rate
learn.lr_find()
# Interpret results
learn.fine_tne(1, 5e-3)
# training output

learn.show_results()

```




----
### Text Transfer Learning
- Let's now see how to fine-tune a language model and train a classifier, using the approach from the ULMFit paper using the IMDb dataset.

```python
from fastai.text.all import *

# Download and uncompress our data
path = untar_data(URLs.IMDB)
path.ls()
# (#5) [Path('/home/sgugger/.fastai/data/imdb/unsup'),Path('/home/sgugger/.fastai/data/imdb/models'),Path('/home/sgugger/.fastai/data/imdb/train'),Path('/home/sgugger/.fastai/data/imdb/test'),Path('/home/sgugger/.fastai/data/imdb/README')]

# The dataset follows "IMageNet-style organization"
# Looks like we're got a train folder with two subfolders, pos and neg (for positive and negative reviews).
# We can gather it using the TextDataLoaders.from_folder method -- we just need to specify the name of the validation folder, which is "test" (and not the default "valid")

dls = TextDataLoaders.from_folder(untrar_data(URLSs.IMDB), valid="test")

dls.show_batch()
#...

```



-----
### Tabular Training
...

----
### Collaborative Filtering
...







