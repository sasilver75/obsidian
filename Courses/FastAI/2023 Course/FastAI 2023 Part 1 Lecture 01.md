
Version 5 of the course (2022), the first one they've done in 2 years. A lot has changed!

PaLM (Google AI Pathways Language Model): It's a text model that can not only answer a question, but can "explain its thinking."

Unlike in traditional machine learnign where we do a lot of feature engineering before throwing those features at the model, we instead (can) just give the raw data at the model, and then we can look at it{s early layers} and see what it discovered, in terms of features. Layer 2 {eg} features will be even more advanced.

With a Neural Network, the basic function used in Deep Learning, we don't have to hand-cdoe any of these features -- instead, we just start with some RANDOM neural network, feed it examples, and we have it learn to recognize things! It turns out that these are things that it learns for itself. When we combine those features, it creates feature detectors that detect {as layers progress}, lines, curves, letters, words, etc. Trying to code these features by hands would be insanely difficult even if you *knew* how to encode them by hand!

The key difference above is why we can now do things that we didn't conceive of as possible -- we don't have to hand-code the things we want to look for (features) -- instead, they can be **LEARNED**.

You can use these to classify sounds too! You can take sounds, turn them into images (spectrographs) and then run image classifiers against them!
You could even take a time-series data and turn it into an image, and then run image classifiers on the IMAGE (graph) of the time series data and get results!


| MYTH | TRUTH|
|-------|---------|
| Lots of math | High school math is sufficient!
| Lots of data | We've seen record-breaking results with < 50 items of data!
|Lots of expensive computers | You can get what you need for state of the art work for free!

One of reasons for this is [transfer learning], which we'll learn more about later.

PyTorch vs TensorFlow in 2022:
* The majority of Researchers in 2022 are now doing Pytorch; the papers will be written in this framework. Tensorflow is dying.

PyTorch still requires a lot of hairy code for relatively trivial things!
![[Pic - Adam Optimizer in PyTorch.png]]
As you can see, there's a lot of work to do.

In `fastai`, a library on top of `PyTorch`, it's much easier! Less code is better, particularly if it's not the thrust of your research/model.
![[Pasted image 20231120173601.png]]
If you use the code that fastai provides, you'll generally find that you have better results.

We'll be seeing more and more pure PyTorch as we go on, just to see how things work.

Jupyter Notebooks are a powerful way to explore and build. They're widely used in industry and in research.
Most students don't even run Jupyter notebooks on their own computers; instead, on cloud servers, of which there are quite a few.

If you go to course.fast.ai, you can see how to use various different **cloud notebooks** (`Colab`,` Gradient`, `Sagemaker`, `Kaggle`).

-----
Kaggle Notebooks
- At the top right, you'll see a button called Copy and Edit on someone else's notebook. You can then use that notebook!
- Type in any arbitrary expression in Python and click Run next to the cell. When it says "session is starting," it's basically launching a virtual computer to run your code.
	- You can also press `Shift + Enter` to run a cell.
	- It's just a calculator where you have all of the capabilities of the world's most popular programming language.
- You can also just write prose (meaning non-code) text by just typing into a cell! Treat them as documentation, introductions, etc.
- When you create a new cell, you can create a code cell that lets you type calculators or a Markdown cell that lets you create prose.
- In the code cells, you can start with a `!` to do a terminal command, like `!ls` -- cool!
- In 2017, Jupyter Notebooks won the 2017 ACM Software System Award, which is a big deal! It's good software 😄.

------
📝 FastAI has a lot of libraries that you can download!
Including `fastai`, `fastdownload`, ... They'll all have something to do with the `fast...` prefix, of course!
- For example, fastdownload has a download_images function that downloads a bunch of urls in parallel, efficiently! Cool!
	- Aside: There's probably something that I can learn by reading into how this stuff works.

The `DataBlock` command is a key thing that you'll want to get familiar with at this point
* How do I get this data into my model?
	* This is surprising that this is the important question! Why not something about gradients or optimizers? 
	* The truth is that at this point, the DL community has found a reasonably small number of types of model that work for nearly all of the main applications that you'll need; **FastAI will create the right type model for you** the vast majority of the time! 
	* So the stuff about subtly tweaking NN architectures will come up later in the course, but... it's not as important as it once was. This course is called Practical Deep Learning, so we'll focus on the things that are actually important.

Jeremy Howard tends to use a funcitonal style in a lot of his programs; a lot of people in Python are less familiar with that. That's why you'll see him using things like `map` a lot.


We asked: "Across all of the projects that we've worked on, what are the things that CHANGE? Those are the things that we parameteize"

```python
# Create our DataLoader via DataBlock
dls = DataBlock(
	blocks=(ImageBlock, CategoryBlock),
	get_items=get_image_files,
	splitter=RandomSplitter(valid_pct=.2, seed=42,
	get_y=parent_label,
	item_tfms=[Resize(192, method='squish')]
).dataloaders(path) # Creates a PyTorch DataLoaders object

# Show me an example of a batch of data that you'd be passing into the model!
dls.show_batch(max_n=6)
```

A `DataBlock` asks for you to provide (among others):
- blocks
	- What kind of input do we have? The input is an image, and the output is a category (one of a number of possibilities). That's enough for `fastai`  to know what kind of model to build for you!
- get_items
	- A function which returns a list of all of the image files in a path, based on extension. It uses this to find the things to train from.
- splitter
	- It's critical that you put aside some data for testing hte accuracy of your modle. This is called a `Validation Set`! It's SO CRITICAL that `fastai` won't even let you train a model without one. This one says to randomly set aside 20% of the data.
- get_y
	- A function telling `fastai` how to get the correct label (`y`) of the photo (is it a forest or a bird?)
		- In our case, it simply returns the parent photo
- item_tfms
	- Most computer vision architectures need all the inputs as you train to be the same size
	- This "item transforms" are all of the bits of code that we want to run on every image.
		- "Okay, resize each of htem to be 192x192 pixels".
		- There are two ways to resize:
			- Crop out bits to get down to 192x192
			- Squish the whole thing to 192x192


that's the DataBlock! From there we create a `DataLoaders` class. These are the things that PyTorch iterates through to grab a bunch of data at a time. It does this by using a `GPU`. A dataloader will field a training algorithm with a bunch (`batch`, `minibatch`) of your images at once!

You'll be curious to ask as you develop:
- What kind of splitters are there?
- What kind of optimizers are there?
- What kind of Blocks are there?
The answers to these can be found in the [fast.ai docs](https://docs.fast.ai/)!


Now we're reading to train our model!
The critical concept in fastapi is a `Learner`, which combines a `model` and `data` -- so you pass in those two things! 
- The data is in the form of our DataLoaders object
- The model is the neural network function you want to pass in
	- We're using `resnet18` here.
	- We integrate a library called `timm` (Pytorch image models), which is the largest collection of state-of-the-art deep learning Computer Vision (CV) models in the world! 
	- fastai is the only framework at this point to integrate this. Awesome!
	- More here: [timm docs](https://timm.fast.ai/)
		- We can get a lot of information about all the different models that Russ has provided. Cool! 
		- The model family called `resnet` are probably going to be fine for all of the things that you want to do!

The reason that we can do this so fast is that someone else has already trained the resnet model to recognize over 1,000,000 images of many types (the IMAGENet dataset) -- they then made their weights/parameters available on the internet for anyone to download!

```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

```
Something that fastai has that's unique is then this fine tune method that takes the pretrained weight we downloaded and adjusts them in a very careful way to teach the model the differences between your dataset and what it was orginally trained on -- this is called `fine tuning`

After a few seconds, it turns out that our model is 100% accurate on our `hold-out validation set` of data.

So now we've got a model that's able to recognize bird pictures versus forest pictures!

We can call .predict on our learner to predict on (eg) out-of-sample images!

```python
is_bird, _, probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability of bird: {probs[0]:.4f}")

# The return tuple of learn.predict(...) seems to be (Whether it's a bird or not as a string, Whether it's a bird or not as an integer, Probabilities of outcomes)
```

In the code you have, whatever it needs to do... in this case it's feeding in a bird picture from the funny xkcd comic... we can just call the one line of code, 
```python
learn.predict(...)
```
Powerful stuff!

----
There's more in the world than just `image classification` (even inside computer vision!)
* For example, there's `segmentation`!
	* eg... Taking photos of a road scene and classifying every single pixel as to "what it is" -- Is this a road? A car? A tree? 
![[Pasted image 20231120182040.png]]

```python
path = untar_data(URLs.CAMVID_TINY)

# Create our DataLoader
dls = SegmentationDataLoaders.from_label_func(
	path
	bs=8,
	fnames=get_image_files(path/"images"),
	label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
	codes = np.loadtxt(path/'codes.txt
	, dtype=str')
)

# Create a Learner by combinging out DataLoader with a model
learn = unet_learner(dls, resnet34)
# Fine tune this pre-trained model using our unique dataset `dls`
learn.fine_tune(8)

```

Datablocks we used before are an intermediate level, very flexible approach to handle almost any kind of data. But for the type of data that occurs a lot, we can use hte special `DataLoaders` classes like above that require us to use even less code!
* See above that it has pretty similar thigns that we pass in for the DataBlocks call from earlier.
	* label_function again grabs the path
	* codes ... are telling us what each code means (it's in a text file)
* The basic information that we're providing is very similar, regardless of whether we were doing segmentation or image classification.

What about stepping away from computer vision?
The most wide-spread model might be Tabular Analysis (tables, spreadsheets with columns, and trying to predict things off there).
That looks similar too!

```python

from fastai.tabular.all import *

# This downloads some data and decompresses it for you; there are a whole lot of URLs provided by fastapi for many common datasets
path = untar_data(URLs.ADULT_SAMPLE)

# Create our (tabular) DataLoader
dls = TabularDataLoaders.from_csv(
	path/'adult.csv',
	path=path,
	y_names="salary",
	cat_names=["workclass", "education", "marital-status", "occupation", "reslationship"]
	cont_names = ["age", "fnlwgt", "education-num"],
	procs = [Categorify, FillMissing, Normalize]
)

# Let's use .show_batch(...) to see an example of some of the data that would be processed togheter:
dls.show_batch()
# Image of the data in tabular format, cool!
```

If you call show_batch on something, you should get back something useful regardless of the information that you provide it.

What if you don't really need to pretrain your model?

```python
# We pass the DataLoader
learn = tabular_learner(dls, metrics=accuracy)

# We don't say Fine Tune, we say Fit
# This is becausef for tabular learners, there probably aren't going to be off-the-shelf models that do exactly what you want, because your data is probably pretty unique (vs image classifications, which all involve... similar shapes and stuff)
learn.fit_one_cycle(2)
```

`Collaborative Filtering` is when we take a dataset and say "Which users liked which prodcuts?" and then we use that to guess what other products the user might like given what *similar users* liked.
* where *similar* is similar in the sense of "people who liked the same kind of products," not similar in the sense of demographics.
	* "People who listen to Kanye also like..."
		* NOT "People who are African American like ..."


```python

from fastai.collab import *

# Download/decompress data
path = untar_data(URLs.ML_SAMPLE)

# Create our DataLoader
dls = CollabDataLoaders,from_csv(path/'ratings.csv')

# Let's peek at one of our batches
dls.show_batch()
# Has pkId, userId, movieId, rating

# create a Learner, again
# Because we're not predicting a category, we're predicting a real number, we tell it the possible range. The actual range is 1-5, but for reasons we'll learn about later, it's actually good to GO A LITTLE BIT LOWER THAN YOUR MINIMUM TO A LITTLE BIT HIGHER!
learner = collab_learner(dls, y_range(0.5, 5.5))

# We don't need to fine tune because there's no such thing as a pretrained collaborative filtering model, but it turns out that fine_tune works out fine as well (though we could also run fit_on_cycle(...) )
learner.fine_tune(10)
# Learning output is displayed
```

* The output for fine_tune has "valid_loss" which is [Mean Squared Error (MSE)]. 

A lot of people on forum are asking how people turn the notebook into a presentation:
* He uses a tool called RISE, which is a Jupyter notebook EXTENSION. In the Notebook, it gives you a little thing on the side of a cell where you can mark whether things are slides or fragments.

So what does he make with Jupyter notebooks?
* The entire O'Reilly book was written in Jupyter notebooks! They're in the FastAPI github repo.

----
So what can Deep Learning do at present?
- We're still scratching the surface, even with how hyped it is!
- Every time someone says "I work in Domain X and want to see if DL can help, and then he talks later, they say 'we broke the SOTA in our field'""

Natural Language Processing (NLP)
- Answering questions, speech recognition, summarizing documents, classifying documents, finding names, dates, etc. in documents; searching for articles mentioning a concept.
Computer Vision (CV)
- Satellite and drone imagery interpretation (eg for disaster resilience), face recognition, image captioning, reading traffic signs, locating pedestrians and vehicles in automated vehicles
Medicine
- Finding anomalies in radiology images, including CT, MRI, and X-ray images; counting features in pathology slides; measuring features in ultrasounds; diagnosing diabetic retinopathy
Biology
- Folding proteins; classifying proteins; many genomics tasks, such as tumor-normal sequencing and classifying clinically actionable genetic mutations. Cell classifications
Image generation
- Colorizing images, removing noise from images, converting images to art in the style of famous artists
Recommendation Systems
- Web search, product recommendations, home page layout
Playing Games
-  Chess, Go, Dota
Robotics
- Handling objects that are hard to locate/pick up
Financial and logistical forecasting, text to speech, and much much more

Deep learning is incredibly powerful now, but it's taken decades for us to get to this point. The basic ideas have not changed much at all, but we do have things like GPUs now, and SSDs and stuff like that, and much more data is just ambiently available.

-----

![[Pasted image 20231120190849.png]]

A machine learning model doesn't look that different to a normal program, but...
* The program has been replaced with something called a model; and we don't just have `inputs`, we also have `weights` (also called parameters)... The model is a mathematical function that (in the case of an NN) multiplies them by a set of weights, add them up, and does this for a second sets of weights, and so forth...

Let's take oru inputs and weights and stick them through the model, and get the results.
Let's then decide HOW GOOD those results are (by calculating the `loss`, and feed that information in some way back to the weights of the model! 

![[Pasted image 20231120191137.png]]

The critical step is that "update" portion; We need a way to update the weights such that they're "a bit better" than the previous set. And by "a bit better", we *specifically* mean a set of weights that make the `loss` go down a bit.

We need some mechanism of making it a little better... If we can just do that once.. then we just need to iterate a few times!
* Put in some data, get results, compute loss, make it better by adjusting weights
*  Put in some more data, get more results, compute loss, make it better by adjusting weights
* ...
* Eventually, it's good!

## "These neural networks are actually *infinitely flexible functions*. This incredibly simple series of steps... can actually solve any computable function."

Something like "Generate an artwork based off of someone's twitter bio" or "translate english to chinese" are both actually computable functions! They might be strange ones, but they are!

So the key is to just create the "update" step

Once we're ran through the training procedure and we're happy with our model and want to deploy it, we don't actually need the loss anymore, and we can sort of "integrate" the weights *into* the model, since at that point they're static.

So we have something that looks like this:
![[Pasted image 20231120191529.png]]
- Takes inputs, puts them through the models, and gets results.
- This looks very similar to our original program!


-----------
Notes from the book:
- We define some `Transforms` that we need when we create our `DataLoader`. A `Transform` contains code that's applied automatically during training.
	- A `item_tfms` argument for our DataLoader applies transforms to each item (eg Resizing them to a 224-pixel square)
	- A `batch_tfms` argument for our DataLoader ia applied to a *batch of items at a time* using the **GPU**, so they're particularly fast!

- validation / training set
- random seed
- ==Overfitting== : When you're training your model, the longer you train for, the better your accuracy will get on the training set; validation set accuracy will also begin to improve for a while, but eventually it starts getting *worse* as the model actually begins to memorize the TRAINING set, rather than finding generalizable underlying patterns in the data.
	- Overfitting is the SINGLE MOST important and challenging issue when training for all machine learning practitioners, and for all algorithms! It's easy to make a very flexible function; it's hard not to fit it too closely to your data!
	- We'll learn many methods to avoid overfitting in our book, but we should *only use these methods after we're confirmed that overfitting is actually occurring!* 
		- Lots of deep learning practitioners will erroneously use these techniques even when they have ENOUGH data such that they didn't need to do so, ending up with a model that's less accurate than it could have been.

==Validation Set==
- When training your model, you should always have both a *training* and a *validation* set. You must measure the accuracy on your model *ONLY* on the validation set! IF you train for too long, with not enough data, you'll start to overfit, and your validation set accuracy will decrease.

==Metric==
- A *metric* is a function that *measures the quality of the model's predictions using the validation set*, and will be printed at the end of each epoch, using `fastai`.
- `error_rate` is a metric provided by fastai that tells you what percentage of images in the validation set are being classified incorrectly.
- `accuracy` is a common one too, which is just 1 - error_rate
- The concept of a metric may remind you of ==loss==, but there's an important distinction: loss is what's used to update the model weights automatically during training, and metrics are just used to evaluate performance, and are designed for human consumption.

When we use a pretrained model off the shell, we often remove the last layer, which is the one that's always specially customized tothe original training task (ie ImageNet classification) and replace it with one or more *new* layers with randomized weights of an appropriate size for the dataset we're working with. This last part of the model is known is known as the ==head== of the model.

Using pre-trained models in this way and performing ==transfer learning== is the quickest path to getting more accurate models, more quickly, will less data, less time, and less money.

The importance of using pretrained models is generally not recognized or discussed in most courses, books, or software library features, and is rarely considered in academic papers. 

An ==epoch== is one complete pass through the dataset -- after calling `fit` , the results after each epoch are printed, showing hte epoch number, the training and validation set losses (used to tune the model) and any *metrics* you've requested. 




