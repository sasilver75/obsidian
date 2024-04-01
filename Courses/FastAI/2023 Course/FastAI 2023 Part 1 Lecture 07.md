*What's Inside a Neural Net?*

There are a lot of tweaks that we can do to Neural Nets
- Most are about tweaking the very first layer or the very last layer -- we'll focus here!
- Over the next couple weeks, we'll slowly get into the stuff in the middle/guts of the models.


We're going to be continuing on from our Rice Disease Detection example, where last time we:
- ConvNext model
- Preprocessing/augmentation
- Test Time Augmentation
- Scaled it up to larger images, and even rectangular images

That got us into the top 25% of the competition!

In this part, we're going to look and something that's pretty interesting -- a simple trick for scaling these models up further... these larger models have more parameters, which mean that they can find trickly little features! The problem is that those activations, or more specifically the gradients that have to be calculated, chew up memory on your GPU!
- Your GPU isn't capable doing doing swap memory like your CPU; when it runs out of memory, it runs out of memory! It just allocates blocks of memory and it stays allocated until you remove them!
- If you try to naively just select the next larger ConvNext model, you'll just run out of memory and get some sort of "Cuda out of memory" error! :D 

We'll talk about a trick to use the largest model that you like on Kaggle! :D 
- When you run something on Kaggle (actually on Kaggle) you'll be on a 16GB GPU, generally. 

He wanted to find out really quickly how much memory a model will use. Here's a hacky way of doing it:

```python
# Read in our training dataset of rice stuff
df = pd.read_csv(path/'train.csv')
# For the label column, what are the counts of each label
df.label.value_counts() 
```
![[Pasted image 20231130190110.png]]
```python
# Okay, well let's not train a model that looks at all of the diseases.... let's insted look at just ONE of the diseases -- the smallest one! And let's make this bacteriail_panicle_blight diseaes.
# It's not that we care about this model, but we can see how much image it uses! 

# (Note that we've added an accum argument to implement "gradient accumulation" -- As we'll see in the code, this does two things): (1) it divides hte batch size by accum and (2) it adds the GradientAccumulation callback, passing in `accum`
def train(arch, size, item=Resize(480, method="squish"), accum=1, fine_tuneTrue, epochs=12):
	dls = ImageDataLoaders.from_folder(
		trn_path,
		valid_pct = .2,
		item_tfms = item,
		batch_tfms = aug_transforms(size=size, min_scale=.75),
		bs=64//accum
	)
	
	cbs = GradientAccumulation(64) if accum else []

	learn = vision_learner(
		dls,
		arch,
		metrics=error_rate,
		cbs=cbs  # Pass the gradient accumulation callback
	).to_fp16()

	if finetune:
		# If this is just a pretrainedm model, finetune it
		learn.fine_tune(epochs, .01) # Finetune
		return learn.tta(dl=dls.test_dl(tst_files))  # Run TTA on test ata
	else:
		# If this is a non-pretrained model, unfreeze it and train it
		learn.unfreeze()
		learn.fit_one_cycle(epochs,0.01)

```

We can then see how much memory we use
Realize
- Each image is the same size
- Each batch is the same size

So training for longer won't use more memory


Above: 
==Gradient Accumulation== refers to a simple trick:
- Rather than updating the model weights after every batch based on that specific batch's gradients, instead we keep *accumulating* (adding up) the gradients for a few batches, and then update the model weights using those accumulated gradients
	- Sam Question: This is basically like having a larger batch size, no? This doesn't seem obviously better, it's just moving us more towards Batch GD on the SGD -> BatchGD spectrum, no?
	- In fastai, the parameter you pass to `GradientAccumulation` defines how many batches of gradients are going to be accumulated. Since we're adding up the gradients over `accum` batches, we therefore need to divide the batch size by that same number.
		- Sam: Oh... I didn't realize that they were doing this.
	- The resulting training loop is mathematically identical to using the original batch size, but the amount of memory used is the same as using a batch size `accum` times smaller!
		- Sam: Okay, let me draw an image to explain to myself
![[Pasted image 20231130191313.png]]

Here's an example in Code of what that kind of looks like:

For without gradient accumulation:
```python
# Wher every x is a batch tensor
for x, y in dl:
	calc_loss(coeffs, x, y).backward()  # Get gradient; set it on coeff
	coeffs.data.sub_(coeffs.grad * lr)  # Update params using grad*lr
	coeffs.grad.zero_()  # Reset coeff grads
```
To with gradient accumulation (assuming bs=64)
```python
count = 0
for x, y in dl:
	count += 1
	calc_loss(coeffs,x,y).backward()  # This sets on coeff the .grad
	# Only once we've calculated grad 64 times do we then update coeffs
	if count > 64:  # Should this be >=
		coeffs.data.sub_(coeffs.grad * lr)
		coeffs.data.zero_()
		count = 0  # Reset count
		
# Recall that if you call .backward() again without zeroing out the gradients that it ADDS the set of gradients to the old gradients; By doing these two (eg) half-size batches without zeroing them, we end up with the total gradient of the 64-image batch size, but just by passing actual batches of 32 images to the GPU.
```

Let's now train our small convnext  model using just the single rice disease label, of the many rice diseases labeled!

```python
# Using our train function from before 
# 128 is the image size
train('convnexxt_small_in22k', 128, epochs=1, accum=1, finetune=False)

# What happens if we train a model 

# Here's somethign that reports hte amount of gpu memory:
import gc
def report_gpu():
	print(torch.cuda.list_gpu_processes())
	gc.collect()
	torch.cuda.empty_cache()

report_gpu()
# GPU: 0
```

So what's Gradient accumulation?
- The key step is
	- We set our batch size (the number of images that we pass through to the GPU at once) to be 64 // accum. So if we pass 2, it will use an actual batch size of 32.
	- So this should cure our memory problems, but the problem is that the dynamics of our training is different!
		- The smaller the batch size, the more volatility from batch to bass, which messes with your learning rates and stuff.
		- So we want to find a way to run just (say) 32 images at a time through, but update AS IF it were 64 images running through?
		- The solution is to consider our training loop! We use the callback system to (in this case) accumulate the gradient that's calculated for each batch *twice* (because `accum` is 2) before applying the gradient to our params.
		- Basically we have an *effective* batch size of 64, in some sense.


Implications
- People on Twitter,Reddit say that they need to buy a bigger GPU to train models... But they don't! They just need to use gradient accumulation.
- Given the $ between a 3080 and 3090Ti (huge price differential), the performance is not that different; the big difference is the memory --- so what! Just put in a smaller batch size and do gradient accumulation!


----
Questions:
- How do we pick a good batch size?
	- Different architectures have different amounts ofm emory that they take up
	- You'oll endu p with different batch sizes for different architectures
	- This isn't necessarily a bad thing, but then each might need a different learning rate/weight decay... The settings that are working for batch size 64 might not work for 32. Again, you wanna be able to experiment as quickly as possible.
	- Re: picking a batch size, the standard approach is to pick the largest one you can; it's faster, you're getting more parallel processing... but tbh he often uses batch sizes that are smaller than he needs, but ... it doesn't make that much of a difference.
	- For performance reasons it's good to use multiples of 8 (people happen to use powers of 2 for some reason)
	- General rule of thumb is that if you divide the batch size by 2, you divide the learning rate by 2, but that's not always exactly right.
----

We do this Gradient Accumulation via a Callback in `fastai` , which is a function that runs while the model trains


So then Jeremy went through a bunch of larger model architectures, and tried running them with `accum=1`  and got OOM errors on his GPU; it turned out then that all of the models' training worked with `accum=2` ! Cool :) 

So then he created a little dictionary of all of the architectures that he wanted, and for each arch, all of the resizes and final sizes he wanted:

```python
models = {
	"convnext_large_in22k": {(Resize(res), (320,224))},
	"swinv2_large_window12_192_22k": {
		(Resize(400, method='squish'), 224),
		(resize(res), 224),
	},
	"swin_large_patch4_window7_224": {
		(Resize(res), 224)
	}
}
```
- Above: most transformers models have a fixed size of (eg) image that it wants to process, so we have to make sure that we use the size that the architecture asked for.

```python
# Let's use the full training data
trn_path = path/'train_images'

tta_res  []
for arch, details in models.items():
	for item, size in details:
		print(arch,size, item.name)
		# Finetune and run predictions with tta on our test data
		tta_res.append(train(arch, size, item=item, accum=2))
		# Handy things to clean up your CPU/GPU memory between runs
		# Otherwise your GPU memory gets like... fragmented or somethingg and you run out of memory when you thought you wouldnt.
		gc.collect()
		torch.cuda.empty_cache()

 
```

Why didn't we set a random seed earlier? So that our models train on different train/val sets for each of them! We want to ensemble these models, using ==Bagging== -- we want to take the average of their predictions!
- With RFs, we were taking averages of intentionally weak models; we're not doing that here, but they're all different architectures with different data, etc. When we average them out, hopefully we get a good blend of different ideas, which is hopefully what you want to use in bagging!

We can stack up these list of probabilities and take their mean

```python
avg_pr = torch.stack(tta_prs).mean(0)
avg_pr.shape
# (3469, 10) <- For each of the 3469 images, we have a prediction for each of the 10 disease categories

# Now we need to just use argmax to get the index of the highest confidence for each of the pictures
idx = avg_pr.argmax(dim=1)
vocab = np.array(dlds.vocab)
# And then look up the actual labels and attach them to the label column on the submission csv
ss _ pd.read_csv(path/'sample_submission.csv')
ss["label"] = vocab[idxs]
ss.to_csv('subm.csv', ...)
```

end of "Scaling Up: Raod to the Top, Part 3" notebook
- This got us to the top of the leaderboard! 

Questions:
- Would trying out cross validations with k folds make sense?
	- K fold cross validation is something similar to what we've done here; We've trained a bunch of models using different random 80% of the data. 
	- 5-fold cross-val does something similar, but rather than picking (say) 5 samples out with different random subsets, instead... do all except for the first 20%, then all but the second 20%, all but the third 20%... So you have 5 subsets with different validation sets that don't overlap at all...
	- But cross-validation is something that Jeremy uses less than most people (almost never)
- Any drawbacks or potential gotchyas with gradient accumulation?
	- Not really! It doesn't even really slow things down much to go from a batch size of 64 to 32; either way, the idea is you're maxing out your GPU.
	- We should all be buying cheaper graphics cards with less memory in them.
	- He expects you could buy 2x 3080s for the price for a 3090ti!
- Any GPU recommendations?
	- Obviously at the moment NVIDIA is the only game in town. If you're trying to use an Apple M1/M2 or an AMD card, you're in for a world of pain in terms of compatability and stuff... unoptimized libraries...
	- The NVIDIA consumer cards (the ones starting with RTX) are much cheaper, but are just as good as the expensive enterprise cards... the reasoning is that there's a licensing issue where they won't let you use an RTX card in a data center.
	- This is why cloud computer is (and ought to be) more expensive... because they have to pay)
- If you have a well-functioning large model, does it make sense to train a smaller model to get the same activations (?) output as the larger model?
	- Yep! we'll talk more about this in part two (teacher, learner, etc). 
	- There are ways to train smaller models that *inference faster* than larger models.

-----

That's the real end to road-to-the top!
But Part 4 is something that's useful to understand for LEARNING, and it will teach us a lot about how the LAST LAYER of a NN works!
- We'll try to build a model that doesn' just predict the DISEASE, but ALSO predicts the type of rice!

![[Pasted image 20231130195931.png]]
We'll build a dataloader that for each image tells us both the type of rice and the disease

To build a model that can predict two things, we'll need:
- a dataloader that has 2 dependent variables!

This is shockingly easy to do in fastai due to the DataBlock API!

```python
# In pandas, you can set a column to be an index; if you do that, it makes the DF kind of like a dictionary! We can say: "tell me the row for this image id"; you can do that using the .loc[....]
df = pd.read_csv(path/'train.csv', index_col='image_id')
# Using our indexed column to select a row, and then selecting the "variety" column from that row
df.loc['100330.jpg', 'variety']
# 'ADT45'

def get_variety(p: Path):
	# We use the filename to get the variety :) 
	return df.loc[p.name, 'variety']


# Creating a DataLoaders with TWO different categorical targets!
dls = DataBlock(
	blocks=(ImageBlock,CategoryBlock,CategoryBlock), # ooo
	n_imp=1, # It doesn't know if the above is 2:1 or 1:2; tell it!
	get_items=get_image_files,
	get_y = [parent_label,get_variety],  # one for each!
	splitter=RandomSplitter(.2, seed=42),
	item_tfms=Resize(192, method="squish"),  # Squash em boys
	batch_tfms=aug_transforms(size=128, min_scal=.75)  # 
).dataloaders(trn_path)  # Turn it into a dataloader by giving train path!

#Then we've got a dataloader that shows the batches we saw above!


```
Now we need a model that predicts two things -- how do we do hat?
- The key thing to realize is that we never had a model that predicted one thing before -- we had a model that predicted TEN things (the probability of each category).
- Now, we want to have a model that predicts TWENTY THINGS! 

How do we do that?
- Let's first try to replicate the disease model we had before with our new dataloader.
- The key thing to know:
	- We told fastai that there's 1 input, so there's 2 outputs; it will submit THREE things instead of two things:
		- The predictions from the model, the disease, and the variety
		- So we can't just use the built-in `error_rate` as our metric anymore; we have to write our own funcitons:

```python
def disease_err(inp, disease, variety): return error_rate(inp, disease)
def disease_loss(inp, disease, variety): return F.cross_entropy(inp, disease)

arch = 'convnext_small_in22k'

# before, our vision_learner was able to guess what loss function to lose because it saw that we were predicting a single category, and the category cardinality was some specific size, so it figured it out for us!
# Now, we have to also provide a new loss function!
# MAE and MSE don't work when the dependent variable is a category; How do you use MSE or MAE to saw how close are these 10 probability distributions to a single correct answer?
# Instead, we use something called CROSS-ENTROPY-LOSS (this is what Fastai was actually picking for us before, lol)
#
learn = vision_learner(
	dls, 
	arch, 
	loss_function=...
)
```


==Cross-Entropy Loss== 
- The stuff that happens in the middle of the model you rarely have to care about; but the stuff that happens at the beginning and end of the model, you'll have to care about a lot; you'll really have to know about things like cross-entropy loss.
- Let's say you're predicting something like a mini imagenet where you're predicting (cat, dog, fish, building), and you set up some sort of model (ConvNext, some linear layers connected up, whatever). 
- You've got some random weights initialized, and it spits out at the end five predictions.
- It doesn't initially spit out "probabilities," it just spits out five numbers; could be negative, positive.
- The output of the model might look like:
![[Pasted image 20231130204132.png]]
- And we want to convert these into probabilities! In two steps:
	- We go exp (e^thing).
	- 


Softmax
![[Pasted image 20231130204158.png]]
- Numerator:
	- e ^ output of cat
- Denominator: 
	- Go through each caegory j of the K categories... And we're going to go e^output, where z[j] is the output for the j'th category
	- Sum of (e^output of cat, e^output of dog, ...)

If you think about it... Since the denominator adds up all of the e-to-the-power-of.... and we do the numerator for each of the categories... it makes sense that if we add up all of the Softmaxes, we get 1, by definition!

Now we have things that can be treated as probabilities - They're all numbers between 0 and 1; Numbers that were bigger in the output will be bigger in the softmax...
But because we did e^...
That means that bigger numbers will be pushed up closer to 1!
That's because the graph of e^x looks like:

![[Pasted image 20231130204542.png]]

Sometimes you'll see people complaining that their model (is it a teddybear, grizzlybear) is wrong because it predicted that a picture of a cat was a grizzly bear; well there's NO WAY for it to predict that something is a cat!
- Something that you COULD do is have it not add up to 1...
- In that case, you could have more than one thing being true, or zero things being true...


Back to ==Cross Entropy Loss==:
- The first part of what cross-entropy does is calculate the softmax (it's actually the log of the softmax, but don't worry about that too much)
- Now, for each of our five things, we've got a probability.
- The next step is the actual cross-entropy calculation!
- We get our softmax for each of our five things (together, a distribution), and we have our actuals (also a distribution... as a sort of one-hot encoding)
![[Pasted image 20231130204822.png]]
- We would expect to have a smaller loss if the softmax were high when the actual were high
- Here's the formula
![[Pasted image 20231130205105.png]]
- We sum up across the 5 M categories..
	- And for each one we multiply the actual target value by the log of the predicted probability (the softmax).
	- Of course, for four of these, that value is zero! By definition, for all but one of them!
	- For the one that's not.... we've got our actual (1) times our log of softmax (log(.87))

This equation looks slightly frightening, but think about it:
- It just finds the predicted probability for the one that is actually 1... and takes the log of it. (at least for a single result, when we're using softmax)
- It looks scary, but that's just how you express it in math, tbh.

We then take that cross-entropy loss and add it over every row 
![[Pasted image 20231130205157.png]]
- Where N is the number of rows
	- Here's a special case (in the blue) that's called BINARY cross-entropy
	- What happens if we're not predicting which one of the five things it is... and we're just predicting "Probability it's a cat"
![[Pasted image 20231130205233.png]]
- These are actually identical formulas... except for two cases:
	- When you either are a cat
	- Or you're not a cat
- Given that your label y[i] is going to be either 1 or 0 (you're a cat, or not a cat, really only one of these terms on either side of this + are going to be non-zero.
- Here's the special case of binary cross-entropy:
- ![[Pasted image 20231130205421.png]]
- Each of these are rows of data
- Predictions are say, sing softmax
- The prediction that you're not a cat is 1- PredictionYou'reACat
- The two columns on the right are showing that these two equations are the same
- Added up is the binary cross-entropy loss of the dataset of predicting cat or not images.

Basically it turns out that all of the loss functions in PyTorch have two versions:
- torch.nn.CrossEntropyLoss(...)  <- A Class
- A version that's just a function.
If you don't need tweaks, you can just use a function.
- The functions live in something like torch.nn.functional, but everyone just calls it F (sort of like np, pd).

So that's all fine! Remember that we're back having created our vision learner for predicting a single output:

```python
learn = vision_learner(
	dls,
	arch,
	loss_func=disease_loss,
	metrics=disease_err,
	n_out=10  # Doesn't know how many acitvations to create, because there are more than one target! This is just saying "What's the size of the last matrix?"
).to_fp16()

# Once we've done that , we can train it!
lr=.1
learn.fine_tune(5, lr)  # Finetune pretrained model for 5 epochs using learning rate lr

```


## Multi-Target Model

- Now we need to output a tensor of length 20, since there are 10 possible diseases, and 10 possible varieties!
	- 10 of those activations are going to predict the disease, and 10 of the activations are going to predict the variety.
	- Q: How does the model know what it's meant to be predicting?
		- A: With the loss function; you're going ot have to tell it!

```python
learn = vision_learner(dls, arch, n_out=20).to_fp16()

# We can define disease_loss just like we did previously, but with an important change: the INPUT tensor is now length 20, now 10, so it doesn't match the number of possible diseases! 
# We can pick whichever part of the input we want to be used to predict disease; let's use the first 10 values:
def disease_loss(inp, disease, variety):
	# inp now has 20 columns in it
	# We're going to arbitrarily decide that the first 10 columns are the prediction probability of what the disease is
	# Let's pass the first 10 columns
	return F.cross_entropy(
		inp[:,10], # Every row, the first 10 columns
		disease
	)

# And then write ANOTHER loss function for the varietal prediction, where we say arbitrarily that the last 10 avtivations are the predicted probabilities of each variet!
def variety_loss(inp, disease, variety):

	return F.cross_entropy(
		inp[:,10:], # Every row, the first 10 columns
		variety
	)

# And then write a single loss function that COMBINES these two loss functions
def combine_loss(inp, disease, variety):
	return disease_loss(inp,disease,variety) + variety_loss(inp, disease, variety)

# We then use this with our learner
```


We can do the same thing for the error metrics:
![[Pasted image 20231130211910.png]]
But these we don't have to combine (but we could) 
![[Pasted image 20231130211938.png]]
![[Pasted image 20231130211946.png]]

![[Pasted image 20231130211953.png]]
Lovely training!

So how does it "know" that the first 10 are varietal and last 10 are disease? Because our loss function informs the gradients, and that MF will optimize for those gradients!

Interesting fact:
- It turns out that this model that's actually trying to predict both disease and variety might over time get better than a model that *just* predicts disease!
- It turns out that the kind of features that help you predict a variety of rice are also useful for recognizing a certain type of disease! Or maybe some diseases impact different varietals in different ways.
* Example: there was a kaggle competition of predicting fish that were caught
	* The model that predicted both the type of boat as well as the type of fish actually won!
	* There's two reasons to learn multitarget models
		* Maybe you want to predict more than one thing
		* Maybe it will be better at predicting just one thing than a single-target model, funnily enough!
	* Intuitively, this makes sene with the fish example:
		* Human: "Well, this looks like a bass, but I see that it's a trawler boat, so it's probably a goldfish!" Seems reasonable; think about how a human might solve it! :) 

-----

## Notebook: Collaborative Filtering Deep Dive

- This is going to cover the last of our four main application areas (vision, nlp, tabular, recsys)

This is the last time that we're presenting a chapter of the book largely without variation (because he couldn't think of any way to improve this).

Using the MovieLens dataset (100k records of it)

```python

# We only care about the user, movie, rating -- the timestamp doesn't even matter
ratings = pd.read_csv(path/'u.data', delimiter="\t", header=None, names=["user", "movie", "rating", "timestamp"])
```
![[Pasted image 20231130212900.png]]
- The goal is to say: 
	- What would user X (in terms of rating) think about movie Y

![[Pasted image 20231130213020.png]]
- Here's another way to visualize the data; cross-tabulation
	- It's particularly full bc these are the most watched movies and 

Ideally we'd like to know... for each movie...
- What are the features of it? Is it actiony? SFy? dialogue-driven? Critically acclaimed?

Let's say we were looking at the movie the last skywalker
- If we had three categories of 
	- scifi
	- action
	- old movies
- We could represnt the movie as :

```python
last_skywalker = np.array([.98, .9, -.9])
```

maybe we could then say that Nick's taste in moviesa reL
```python
np.array([.9, .8, -.6])
```
And then perhaps we could like... look at Nick's tastes, and look at the movie description, and make some claim about how much he might like the movie?

```python
# Maybe we just multiply the matrices together?
(user1*last_skywalker).sum()  # Just a pairwise sum, max of 3)
2.14  # Seemes he might like it?
```


On the otherh and, the movie Casablanca
```python
casablanca = np.array([-.99, -.3, .8])
(user1*casabalnca).sum() 
-1.611
```

This mulitplication of two vectors is called the ==dot product== of the user's preferences and the type of movie!
- The problem is that we wren't given this information!
	- We know nothing about the user preferences (explicitly)
	- We know nothing about teh movie information

We want to learn these things!
We create things called ==Latent Factors==
- "I don't know what things about movies matter to people, bu tehre's probably something -- lets try using SGD to find them!"
	- We can do them in everyon's favorite optimization software; MS Excel!



For each, movie, let's assume there are 5 latent factors -- we'll figure them out later!
Let's randomly intiiatize them
![[Pasted image 20231130214142.png]]
and do the same thing for each user

![[Pasted image 20231130214159.png]]

The idea is ... that the number of .19 in the top right, if it were 


Embedding
- It turns out that an embedding is just looking something up in an array
- That orange thing is an embedding matrix
![[Pasted image 20231130215058.png]]
- We're going to try to LEARN these latent factors, now!

```python
movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1', usecols=(0,1), names=('movie, title'), header=None)

ratings = ratings.merge(movies)
ratings.head()
```
![[Pasted image 20231130215313.png]]
```python

# Expects a user column and an item column
# because our user is called user, we don't have to pass it i
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
dls.show_batch()


```

```python
n_users = len(dls.classes['user'])
n_movies = len(dls.classes['title'])
n_factors = 5

# Initialize random matrices of length # of users/movies, with width of n_factors
user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(m_movies, n_factors)
```

To calculate the result for a particular movie and user combination, we have to look up the index of the movie in our movie latent factor matrix and the index of our user in our user latent factor matrix

Then we do the dot product between the two latent factor vectors

But "look up in an index" is not an operation that our deeplearning models know how to do; they know how to do matrix products and activation functions

Fortunately, it turns out that we can represent *"look up in an index"* with a matrix product! the trick is to replace our indices with one-hot-encoded vectors; here's an example:

```python
# This is how we create a one-hot encoded vector of length n_users where the third elemnt is set to one, and the rest zero
one_hot_3 = one_hot(3, n_users).float()

# If we multiply that using the @ operator
user_factors.t() @ one_hot_3
# tensor([-1.24, -.3, 1.4, 0.08, .0.41])

# What if we had just accessed index 3?
user_factors[3]
# tensor([-1.24, -.3, 1.4, 0.08, .0.41])
# Same result! So Jeremy wasn't lying :) 
```

Taking the dot product of a one-hot encoded vector and *something* is the same thing as looking up a value in that matrix...

So you can think of an EMBEDDING as computational shortcut for multiplying something by a one-hot-encoded vector...
- It seems like embeddings are a cool math trick for speeding up matrix multiplications using dummy variables :thinking:

```python
...
```

==Weight Decay, or L2 Regularization==
- Consists in adding to your loss function the SUM of all the weights squared! In hopes of reducing overfitting.
- Why do that?
	- Because when we compute the gradient, it will add a contribution to them that will encourage the weights to be as small as possible
	- Why would it prevent overfitting?
	- The idea is that the larger your coefficients are, the sharper canyons we will have in the loss function. If we take the basic example of a parabola
		- y = a * (x\*\*2)
		- the larger a is, the more NARROW the parabola is.
- The idea is that by letting our model learn high parameters it might cause it to fit all the data points in the training set with an overcomplex function with very sharp changes, which is going to lead to overfitting.
- Limiting our weights from growing too much is going to hinder the training of the model, but it will yield a state where it generalizes better!
- Going back to the theory,
	- weight decay (or just `wd` ) is is a parameter that controls that sum of squares we add to our loss (assuming `parameters` of all parameters):

```python
# wd might be something like .01 or .001
loss_with_wd = loss + wd * (parameters**2).sum()
```
In practice though, it would be very inefficient (and maybe numerical unstable) to compute that big sum and add it to our loss
- Recall that the derivative of p\*\*2 with respect to p is 2p; so adding that sum to our big sum to our loss is exactly the same as doing:
```python
parameters.grad += wd * 2 * parameters
```
Recall: The purpose of the loss is to take its gradient. The gradient of parameters \*\* 2 is 2p. So instead of determining the loss, all we need to do is add the wd\*2\*parameters

So when you call `.fit` you can call in a `wd` parameter!

```python
model = DotProductBias(n_users, n_movies, 50)
learn = Learner(cdls, model, loss_func=MSELossFlat())

# In fastai, they try to set this for you appropriately; but for things like tabular and collaborative filtering, we don't know enough about your data to do anything here, so try some things out for yourself!
learn.fit_one_cycle(5, 5e-3, wd=.1)
```

















