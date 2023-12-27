
Jupyter Notebook NB Extensions
- Table of Context (2) gives you a handy navigation bar on the top ("Navigate")
- Collapsible headings lets you use arrows to navigate around (left, right) within a section, and collapse headings.

Jupyter tips:
?verify_images will pull up parameter information
doc(verify_images) will pull up a link to documentation/source code

Note:
- There's additional links and context for each lecture on the lecture page - for example to papers/resources talked about during the class, or more!

### Data Cleaning
- Before cleaning your data, we're goin going to train our model! 
	- This might sound CONTRARY to what you've learned about in the past!
- To train our model, like before, we use a DataBlock to grab our DataLoaders... we do some image resizing
	- ResizeMethod.Squish (Can end up with some very thin bear pictures)
		- Our bears are thin, but we can see all of the bear's little cubs!
	- ResizeMethod.Pad (Padding with zeroes give "black bars" on sides of the images to get them to the right aspect ratio)

An interesting `item_tfms` is `RandomResizedCrop`
- This gets a different bit of an image every time 🤔
	- Someone on the forum wanted to recognize pictures of french and german texts -- (this isn't the normal way we'd do that, but). They had big scans of images that were too big -- what should they do! If the use RandomResizedCrop, they'll add different bits of the images.
	- This idea of getting different pictures of the same image is called ==Data Augmentation==.
		- If you use a batch_tfms called `aug_transforms(...)` , you can see that your teddies get turned, squished, warped, recolored, saturated, etc.
		- Generally speaking, if you're training for more than 5-10 epochs, you're probably going to want to use random_resize_crop and aug_transforms.

---

Interesting question from the audience:
- Is this "copying" the image multiples? The answer is no, we're not "copying" the image; each epoch, every image gets {warped}... in memory, in RAM, the image is warped? There's no copies being stored on your computer, but effectively there are infinitely/slightly different copies, because that's what the model is seeing.

----

So then we created a `Learner` from our `DataLoader` and `resnet18`, and then fine_tune it on our dataset.

Remember when we said that we were going to *train* the model before *cleaning* the data? 
We can create a confusion matrix of our predictions by creating an Interporator from our Learner:

```python
interp = ClassificationInterporator.from_learner(learn)
interp.plot_confusion_matrix()
```
![[Pasted image 20231122145927.png]]
Above: We can see that  we mixed up some classifications

This might be worthwhile to show you which types of bears (or, eg. cats, dogs, cancers) are difficult to classify. 

Now that we have this ClassificationInterpretation object, we can do things like plot the top losses from our training set!
```python
interp.plot_top_losses(5, nrows=1, figsize(17,4))
```
![[Pasted image 20231122150115.png]]
Above: This tells the x's in X that had the greatest loss, along with the (PredictedClass, ActualClass, Loss, and Probability {confidence in prediction}).
Recall that the ==Loss== is the measurement of how good our guess is as we run through the data.

Looking at the pictures:
- It's hard to know for some of these if the model made a mistake, or if the labels were incorrect for some of these.
- Note that some of these are actually *correct*! So why is the loss bad for these, when it's still correct? The answer is that the model still wasn't very confident in those choices.
- You can have a bad loss in this scenario by being 
	- wrong and confident
	- right and unconfident

The reason that this is helpful... is that we can use the `fastai` `ImageClassifierCleaner` to *clean up* the records that are wrongly labeled in our dataset!

```python
cleaner = ImageClassifierCleaner(learn)
# This actually RUNS THE MODEL and then shows all of the itiems that were (eg) marked as teddy bears, and orders them by confidence. We can then scroll through all of the images in that category to check if they're correct or not. The ones that are incorrect can be corrected or even deleted from the dataset! Wow.
```
![[Pasted image 20231122150512.png]]
Above:
- I'm really curious how this even works --  Ah, it seems that the ones that you update are actually recorded in some way *inside* the cleaner instance (see the next cell in pic above)
- Note that you can also go through the "Validation" set of data


Now we can take advantage of the choices that we made (which resulted in changes to the state of the `cleaner` object):
```python
for idx in cleanrer.delete():
	# fns here is "filenames"
	cleaner.fns[idx].unlink()

for idx, cat in cleaner.change():
	# cat = "category"
	shutil.move(str(cleaner.fns[idx]), path/cat)
```

----
TLDR:
- ==Before you start data cleaning, always build a model== to find out what things are difficult to recognize in the data, and then use the model to help you find data problems. As you see these problems, you might:
	- Have a way to automate some of the cleaning
	- Make changes to the way that you're collecting data
	- etc.

----

Jeremy Note: GPUs can't Swap like CPU RAM can; when they run out of memory, they're done; so always close out other notebooks and just do one thing at a time.

Jeremy Tip: Watch the entire video without touching the keyboard to get a sense of what the video is about, and then go back to the start and watch it again, then follow along. Then you know what you're doing, what's going to happen next, etc. It's a bit unusual, because with real-life lectures you can't do that 😄.

------------

Now that we've made our model, how are we going to put it into production?
- In the book we use something called `Voila` , and it's pretty good, but in the class we're using something called `Gradio` , hosted on `HuggingFace Spaces` .

There isn't a chapter on this in the book, but it doesn't matter, because Tanishq Abraham has written a fantastic blogpost about it (link to this from the forum/course page).
- Tanishq is one of the most helpful person in the `fastai` community -- incredibly tenacious and patient.

-----

Hugging Face -> Create a Space at https://huggingface.co/new-space
- Select Gradio
...

```python
# This create a file using the trained model.
# It "pickles" the model, which is a Python-specific serialization format
learn.export('model.pkl')
```

If we ran that in Kaggle
You can go to the data tab on your Kaggle notebook, download it, and that will then be downloaded to our downloads folder...
We can then go to our hugging face app on our local machine (a git repository), and move that file to this hugging face directory. 

So how do we do predictions using a saved model like this?
How do we use a serialized model like this to make predictions?

```python
frmo fastai.vision.all import *
import gradio as gr

# Any external functions that you used in your labeling need to be included here as well, because that learner refers to those functions... but it doesn't have the source code for those functions, so you have to keep it with you.
def is_cat(x): 
	return x[0].isupper()

# Create a Python Image Library (PIL) image of a picture of a dog
im = PILImage.create('dog.jpg')

# Isntead of creating a learner, we LOAD in a learner using our serialized (pickled) model from earlier 
learn = load_learner('model.pkl')

# Now we can use it to predict!
learn.predict(im)
# ('False', TensorBase(0), TensorBase([9,9999e-1, 9.543e-06]))
# It's a dog.

# Let's create a gradio interface that has this information! Gradio requires that we give it a funciton to call:

categories = "Dog", "Cat"

def classify_image(img):
	pred, idx, probs = learn.predict(img)
	# Gradio wants to get back a dictionary containing each of the possible categories [dog, cat] and the probabilty of each one
	# Check out this zip idiom of taking two same-length (?) iterables and turning them into key:value pairs in a dict -- cool!
	return dict(zip(categories, map(float, probs)))

# It works!
classify_image(im)
# {"Dog": .99999, "Cat":. 9.34e-06}

# Let's now register it with Gradio (gr)
image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()  # output will just be a label
examples = ["dog.jpg" ,"cat.jpg", "dunno.jpg"]  # Filenames

# Create a gr "interface"
intf = gr.Interface(
	fn=classify_image,
	inputs=image,
	outputs=label,
	examples=examples
)

# This acutally runs it locally @ :7860; to create a public link, use share=True in launch(...)
intf.launch(inline=False)
# If you open it up, you have your own running-on-your-own-box classifier that has some examples that you can stick in, or upload your own! It's running on your own laptop, basically instantly. It's awesome!


# Now we want to make this into a Pytohn script; One way would be to copy and paste all of the things that you need into a script... You write #|export at each cell that contains information you'll need in your last script
# Then, you can import somethign called notebook2script from nbdev
from nbdev.export import notebook2script

notebook2script('app.ipynb')  # The name of the current notebook
# It then creates a file for you called app.py containing that script! Interesting! This is a nice easy way to work with stuff that's expecting a script rather than a notebook. This is a nice way of doing it because you can do experimentation in the notebook, have a cell at the bottom, and then run it.
# How does it know to call it app.py? Because there was a cell at the top of the notebook with #|default_exp app in it :) 
# That's just a little trick that we use :D 
```

-----
Aside: `Gradio`
- `Gradio` is the fastest way to demo your machine learning model with a friendly web interface so that anyone can use it, anywhere!
- Can be embedded in Python notebooks or presented as a webpage. A Gradio interface can automatically generate ae public link you can share with colleagues that lets them interact with the model on your computer remotely from their own devices.
- Once you've created an interface, you can permanently host it on Hugging Face Spaces! Cool! :)


----

Now that we've got an app.py, we need to upload it to gradio -- we just do this by committing and pushing in our hugging face spaces repository. Once we've done that... 


----
Questions:
- What's the difference between a Learner and a PyTorch model?
	- More on this later
- How many epochs do we train for?
	- As you train a model, your error rate improves. The question is then: "Should I just run more, increasing the number of epochs?". It's up to you! If you train for long enough, your error rate actually starts getting worse! (We'll learn more about this later).
---

`Streamlit` is more flexible than Gradio but not quite as easy to get started with (in the domain of building prototypes).
- At some point you'll want to build more than a prototype; you'll want to build an app.
- There's a button at the bottom on your Gradio UI that says" View the API"; We can actually build any app we want, but the thing that actually does the model predictions for us is going to be handled by hugging spaces / Gradio, and then we can write a javascript application that talks to that!
	- Anyone who's done some FE engineering will say: "Oh, I can create anything in the world!"
	- Data scientists may say: "Uh oh, I have no idea how to use JavaScript!"

Clicking on that "View the API" button on your Gradio UI shows the endpoint you can send requests to. You can then build around that.
![[Pasted image 20231122163330.png]]







