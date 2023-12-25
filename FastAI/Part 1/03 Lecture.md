
How do do a fast.ai lesson:
- Watch lecture
- Run notebook and experiment
- Reproduce results
	-  There's a special folder in the fastbook repo called "clean", but with all of the text removed, and all of the outputs removed. This is a great way to test your understanding of the chapter; Before each cell, say: "What's this for, and what's it going to output?"
- Repeat with different dataset

-----

He's going to do some work in JupyterLab; you can do pretty much everything graphically; there's a file browser, a Git UI, etc.

He tends to go into fullscreen with Jupyter (like its own IDE), and you can open up a terminal window, etc.

----

The key thing that we wanted to understand in Lesson 2 was that there's a *training* piece, and then at the end of the training piece, you can create a pickled model.pkl file; You can feed that thing inputs, and it will feed you outputs; you generally don't need a GPU to run inference. And then there's a second step, *deployment* of the model (eg to Gradio, Streamlit).

- Dataloders
- Check that image looks okay with show_batch
- Create a learner with some resnet architecture
- Fine tune it
- Get some accuracy

But which image models are best?
- There are over 500 image architectures in the fastai library; Which should you use?
	- They're all juts functions! Matmuls with some nonlinearities (activations; relus, etc). We most care about:
		- How fast are they to do inference
		- How accurate are they

![[Pasted image 20231123130933.png]]
- X axis: Seconds per sample (how fast is it)
- Y axis: How accurate is it (on ImageNet in particular)
Generally we want things that are up to to the top (accurate) and left (fast) 😄

The ones that seem to be the most accurate and fast are these "levit" models.
The convnext model are also interesting. How do we create this model?

```python
import timm
timm.list_models('convnext')
# ['convnext_base', 'convext_base_384_in22ft1k', ...]

# When we create our vision learner, we can just put the name of the model in as a string, rather than using a model that fastAI provides as a library; fastai only provides a small number. If you install timm, you'll get hundreds more.
learn = vision_learner(
	dls,  # DataLoaders
	'convnext_tiny_in22k', # str name of model from timm package
	metrics=error_rate
).to_fp16()
learn.fine_tune(3)
# Training output omitted

# Export that bitch as a Python-specific pickle serialization
learn.export('model.pkl')
```

If you're not sure what to use, try these convnext architectures
- The names are like convnext_base_..., convenxt_large_..., convnext_xlarge_...
	- This is how much memory it uses
- convnext_large vs convnext_384_in22ft1k, convnext_large_in22ft1k
	- These are been trained on a version of imageNet with 22k categories of images

We can then add image augmentation, do more epochs, etc.

fastai always store the information about categories... in something called the vocab object, which is inside the dataloaders object.
![[Pasted image 20231123131957.png]]
If we now zip together the categorie and the probabilities, we'll get a dictionary 
![[Pasted image 20231123132013.png]]
Of all of the categories and the probabilities of each one. Just like last week, we can go them and create our Gradio interface and launch it. Cool! 

This pickle file is an object called a learner, which has:
- List of preprocessing steps that it did to turn your images into inputs for the model
	- This is basically the datablocks/dataloaders, transforms, etc.
- The trained model
	- Can grab this with `learn.model`
		- This is the actual model! Lots of stuff going on here...


![[Pasted image 20231123132200.png]]
- It seems to contain lots of layers, and it has a sort of tree-like structure... that because lots of layers themselves consist of layers.

Take a look at one of these:
- There's a convenientm method in Pytorch called get_submodule:
	- `l = m.get_submodule(0.model.stem.1)`
		- This returns the LayerNorm2d thing from the picture above

```python
l = m .get_submodule('0.model.stem.1')
list(l.parameters())
# [Parameter containing:
#    tensor([1.234, 1.134134,1.23431432, ...])]
# ^ A lot of parameters
```
So what are these numbers, where on earth did they come from, and how come these numbers can figure out whether something is a Basset hound or not?


How do we fit a function to data?
Say we have 

```
3x^2 + 2x + 1
```
![[Pasted image 20231123134347.png]]
Imagine that we don't know that this is the true mathematical function that we're trying to find.
Let's try to recreate the function

```python
def quad(a,b,c,x): 
	a*x**2 + b*x + c

quad(3,2,1,1.5)
10.75

# We're going to want to create a bunch of different quadratics to test them out and figure out what's best
# functools has a partial tool that lets us basically "fix"/"bind" certain values of a/b/c. The way we do that is like this:

from functools import partial
def mk_quad(a,b,c):
	# This returns a new function that just takes x
	return partial(quad, a,b,c,)

f = mk_quad(3,2,1)
f(1.5)
10.75

# So now we can create any quadratic equation we want. Then we can call this function like any normal function using just an X!

# So our challenge is to determine what a,b,c should be
# Let's generate some data that matches the shape of this function

from numpy.random import normal, seed, uniform
np.random.seed(42)

def noise(x, scale):
	# Creaates some normally distributed random numbers
	return normal(scale=scale, size=x.shape)

def add_noise(x, mult, add):
	return x * (1+noise(x, mult)) + noise(x, add)

# This creates a tensor (in this case a vector) going from -2,2 in equal steps, and there are 20 of them
x = torch.linspace(-2, 2, steps=20)[:, None]
y = add_noise(f(x), .3, 1.5)
plt.scatter(x,y)
# This shows a scatterplot of data that roughly follows the line we plotted earlier

# Let's create a function called plot_quadratic that plots our data as a scatter plot, and then plots our function, which is a quadratic that we pass in
from ipywidgets import interact
@interact(a=1.5, b=1.5, c=1.5)
def plot_quad(a,b,c):
	plt.scatter(x,y)
	plot_function(mk_quad(a,b,c), ylim=(-3,12))

# The @interact gives you these nice little sliders to play around with to vary a,b,c
```
![[Pasted image 20231123135047.png]]

- How would we try to make this fit better?
	- We'd take the first slider, move it to the left, and see if it looks better or worse
		- It looks worse!
		- We move the first slide the other way; it looks better.
	- We take the second slider move it to the right, and see that it looks worse! We move to the left instead
	- We take the third slider, ....
- You can see that we're taking each tunable parameter of our model (the a,b,c in our quadratic function) and moving them in the direction that improves it a little bit. When we're done, we go back to the first one, see if we can make it any better, etc.

This is basically what we're going to do
- But the big fancy models have hundreds of millions of parameters; we don't have time to manually tune 100s of millions of sliders -- we need something etter


We need
1. An idea of "is it getting better or worse" when we move the slider around
	- We need a number that we can measure that tells us "how good is our model"; This is called a ==loss== function; there are many different that you could pick, but the most common is ==mean squared error==


```python
def mse(predictions, actuals):
	# Mean squared error
	# (Sum of squared differences)/(n comparisons)
	return ((prediction-actual)**2).mean()

```

- Let's do the same interactive thing, but also plotting the MSE
	- We do the same wiggling, but instead use the MSE to guide us to the correct number
	- It works!

For each parameter:
- When we move it up, does the loss get better? Or when we move it down, does the loss get better?
- We could manually increase the parameter and see if the loss gets better... but there's a much faster way:
	- To calculate its derivative with respect to the loss funciton!
		- If you increase the input (eg `a`), how much does the loss increase?
			- This is called the `slope`, `derivative`, or `gradient` ! 

Don't worry, Pytorch can do this for you!


```python
def quad_mse(params):
	# This function basically "rates" how good a specific combination of parameters is, via its MSE

	# Create a quadratic passing in our paramters [a,b,c] (spreading)
	f = mk_quad(*params)
	# Return the MSE of our predictions vs the real labels
	return mse(f(x), y)

quad_mse([1.5, 1.5, 1.5])
# tensor(11.4648, dtype=torch.float64)
# Above: note that the .mean() we used in mse(...) reutrns a Tensor

# This is a 1st-degree or "rank one" or 1-d tensor
abc = torch.tensor([1.5, 1.5, 1.5])
# Tell torch: Calculate the gradient for these numbers whenever we use them in a calculation
abc.requires_grad_()
> tensor([1.500, 1.500, 1.500], requires_grad=True)

# Let's use these parameters in the calculation!
loss = quad_mse(abc)
> tensor(11.4648, dtypetorch.float64, grad_fn=<MeanBackward0>)
# Above: See that it outputs what we got before, but there's something that's new here! It added this grad_fn thing. It says that PyTorch knows how to calculate the gradients for these inputs

# We call backward() on the result to calculate the gradient of our loss function
loss.backward()
# No output. It added an attribute to our INPUT (??)

abc.grad
> tensor([-10.9886, -2.1225, -4.0015])
# If I increase a, the loss will go down
# If I increase b, the loss will go down
# If I increase c, the loss will go down

# So I should increase a,b,c -- but by how much?
# It seems intuitively that perhaps we should increase a by a lot, b by a little, and c by a medium amount.
# In otherwords, we want to adjust a,b,c by the NEGATIVE of these

with torch.no_grad():
	# We don't want to jump too far, so just go a small distance
	# Somewhat arbitrarily pick 0.01
	abc -= abc.grad*0.01
	# Calculate the loss with these new parameters
	loss = quad_mse(abc)

print(f'loss={loss:.2f}')
> loss=10.11
# This is a little beter than before!

# What's up with that torch.no_grad thing, though?
# We said earlier that the paraemter abc requires grad, meaning that pytorch will automatically calcualte its derivative WHENEVER it's being used in a function. But the line of abc -= abc.grad*0.01 is actually interpreted as being a "function" in this sense, and we DON'T want to calculate a gradient here because this gradient isn't one that's with respect to our loss function.
# This is basically the standard inner-part of a PyTorch loop in every NN/DL model, at least of this style, that we'll build. If you look deep in fastai source code, you'll see something that looks like this.

# We can automate these steps of:
# Given some abc
# Inference on some data
# Calculate the loss on the inference
# Get the gradient of the params wrt to the loss function
# Adjust the parameters given the gradient (stepping in that direction by some amount)
# Retest the new abc

for i in range(5):
	loss = quad_mse(abc)  # run inference, get the loss
	loss.backward()  # calculate the gradients
	
	with torch.no_grad():
		# Take a small step in the direction of the gradients
		# Recall that the gradients say how the error would change if you increase that parameter... So if the gradient is negative, we want to INCREASE our parameter, meaning we take a step in the direction "opposite" the gradients, so we use -=
		abc -= (abc.grad*0.01)

	print(f"Step={i}; Loss={loss:.2f})


step=0; loss=10.11
step=1; loss=7.88
step=2; loss=5.53
step=3; loss=3.86
step=4; loss=3.42
```

So we now should have some coefficients in `abc` that should be very close to the ground truth function that we know the data was based on

This is just called ==optimization==
- This was the most basic type of optimizer, but they're all built on this principle of ==gradient descent==
	- We calculate the gradients, and then do a "descent" (by stepping in the direction opposite the upwards-slope) and try to reduce the loss!

We need one more piece:
- What's the mathematical function that we're finding the optimal parameters for? Here, it was a quadratic function
	- But it's unlikely that we can just use a quadratic function to do something like "Is the dog in this image? Is it a basset hound?" Because we only have three parameters to tune which means that we don't have a LOT of flexibility in our model.
	- This is the choice of model architecture!

It turns out that we can create an **infinitely flexible function** form this one tiny thing, a ==rectified linear unit (ReLU)== :

```python

def rectified_linear(m,b,x):
	# This is just a linear function with an intercept; it's a line
	y = m*x + b
	# torch.clip (alias for torch.clamp): torch.clamp(input, min, max, ...): Clamps all elements in input into the range [min, max]
	# Basically, this is setting a "floor" for our values; if y was negative, it's now 0
	return torch.clip(y, 0.)

# We use partial to make the function y=1x+1
plot_function(partial(rectified_linear, 1, 1))

```
![[Pasted image 20231123141756.png]]

We can now make the function interactive by using the `@interact` decorator from jupyter widgets.

These aren't interesting of themselves, but...
By introducing this simple nonlinearity, we can model basically any function. 

We can take this relu and create a double_relu!
```python

def double_relu(m1,b1,m2,b2,x):
	return rectified_linear(m1,b1,x)  rectified_linear(m2,b2,x)

def plot_double_relu(m1, b2, m2, b2):
	plot_function(partial(double_relu(m1,b1,m2,b2), ylim=(-1,6))
```

![[Pasted image 20231123141959.png]]
Playing with this, we start to see that we can create

We can add as many ReLUs together as we want! If we can add all of these ReLUs together, we can have an arbitrarily wiggily function, and with enough ReLUs, we can do things like match audio waveforms (very squiggily) 

(And you can have relus in multiple dimensions)
With this incredibly simple foundatino, you can construct an arbitrarily acurtate model

The problem is that you have to have some paramters for this relu
- But we know how to get parameters
	- Gradient descent

We've derived deep learning!
- Everything from now on is tweaks to make it faster, make it take less data, etc. This is it!

This reminds me of the thing of how to draw an owl:
- Draw two circles
- Draw the rest of the owl

The thing that he has a lot of trouble explaining is that there's nothing between these steps

Using gradient descent to set some parameters to make the wiggly function (which is basically the addition of ReLUs, or something similar to that) match your data!

-------
Questions:
- Is there perhaps a way to try out all the different models and automatically find the best performing one?
	- Absolutely, you can do that! Remember that timm.list_models('convenext') that returned a list of strings of models? You could add a for-loop around thsi that basically goes
```python
for architecture in timm.list_models():
	...
```

- At the start of a new project, Jeremy basically only uses resnet18 or resnet34 (a small model) because he wants to try things out; different ways of cleaning the data, different data augmentation, thinking about external data he can bring in, etc. He wants to try things out as fast as possible
	- Trying out bigger architectures is the LAST thing he does.
		- Do I even need it more accurate for what I'm doing? Or do I need it to be *faster* for inference rather than more accurate?
	- Here's the chart link: https://www.kaggle.com/code/jhoward/which-image-models-are-best 


- Q: How do I know if I don't have enough data?
	- A: Same to the architecture question; you've got some amount of data; Presumably you've used all the data you have access to. You've built your model, and it's done its best -- is it good enough?
	- You can't know until you've trained the model, but as you've seen it only takes a few minutes to train a quick model; his opinion is that the vast majority of projects in industry wait far too long before they train their first model;
	- Train your first model on day one with whatever CSV files you might have together!
	- You might find that you're basically getting no accuracy at all -- that it's impossible! These are things that you'd want to know at the start, not at the end.

You'll learn ways about getting the most out of your data. In particular, there's a recentish technique called semi-supervised learning that lets you get more out of your data. How expensive will it be to get more data? And what do you mean by getting more data? More labeled data? It's often easy to get lots of inputs, but hard to get lots of outputs. It's generally easy to jump into radiology archive to get more CT scans, but might be more expensive to draw segmentation masks and pixel boundaries on them. It might be easy to get images, but hard to get labels. You might be able to use semi-supervised learning to take advantage of unlabeled data  as well.

- Q: In the quadratic example, where we calculate the derivatives for a,b,c,. What units are these in? Why do we not move the params by that amount, instead of by (eg) 0.01 * gradient?
	- Units:
		- For each increase of A (1.5 -> 2.5), what would happen to the loss? It would go down by 10.987.
			- This isn't exactly right, because the gradient is actually sort of like the secant on a curved line. But at the point we're at, if we took a gradient, that's the slope of it.
		- Why use that 0.01 thing?
			- If we jump super far, we might overshoot - particular as you approach the optimal, the slope will decrease, which is why multiply the gradient by a small number.
			- This small number is called the ==learning rate==. It's an example of a ==hyperparameter==  -- a parameter that you use to guide training; a parameter that you pick to help you "back out" the actual model parameters! It takes longer to train with a smaller learning rate (and you might get stuck in bad (or good) local minima). (Often times we start with a higher learning rate and decrease it.)
			- `fastai` will pick reasonable defaults for many of these things.
-------

Let's look at an important mathematical/computational trick:
- We want to do a whole lot of ReLU's -- a lot of mx+b's.
	- And we want to have lots of variables -- if we have every single one of the pixeles of an image, we'll have mx + nx + ox + px + b or something
	- It's going to be kind of hard to write this out for 5000000 pixels.

==Matrix Multiplication== to the rescue!
- When people talk about linear algebra in deep learning, they give the impression that you need years of study -- you don't. Almost all you need is matrix multiplication.
	- [matrixmultiplication.xyz](http://matrixmultiplication.xyz/)
	- Matrix mutliplilcation is the critical mathematical operation in basically all of deep learning; The GPUs that we use... the thing that they're good at is THIS. They have special cores called ==tensor cores== that can basically only do one thing -- multiply together two 4x4 matrices (and they do this lots of times for bigger matrices.)

We're actually going to build a complete ML model on real data in a spreadsheet! :) 
- fastai has become famous for a number of things, and one of them is using spreadsheets to create deeplearning models, lol

----

NLP
- Rather than taking image data and making predictions, we take text data (often prose) and make predictions!
	- If you're a non-english speaker, you'll find that for many languages, there are fewer resources in non-english languages, and there's a great opportunity there.
- Classification
	- Take a *document* (words, wikipedia page, book, etc.), and we try to predict a category for it
		- genre
		- sentiment (positive/negative)
		- ...

Hugging Face Transformers doesn't have the same high-level api that fastai has, but ... at this point in the course, we're going to intentionally use a library that's a little less user friendly just to see what other steps you have to go through to use other libraries.

The reason that we picked this other library (hugging face transformers) is that it's really good! Has a lot of good models and techniques in it. They're hired lots of fastai alumni, which isn't that unsurprising.















