
![[Pasted image 20231128180538.png]]
A 4-node random forest

Gini Coefficieint
- The ==Gini Cofficient== is the probability that if you pick two rows from a group, you'll get the same `Survived` (y) result each time .
- If the group is all the same, the probability is 1.0 , or 0.0 i they're all different.

With decision tres, you can usually get away with stuff like not creating dummy variables
And you don't have to do logging of things like $ variables either; Because splitting on log(2.6) is the same as splitting on 2.6 -- all the decision tree cares about is the ORDERING of the data. It doesn't get weirdly affected by outliers, long-tailed distributions, categorical variables, etc. For tabular data, Jeremy would always start by creating a baseline with a decision-tree-based approach. It's hard to mess those up, which is important!

Could we grow the tree further to get more accurate?
We could, but... there's currently only 50 samples in these leaves... if we keep splitting it, the leaf nodes are going to have so little data that it won't make good predictions (?).

So what can we do? 
Here's a trick :) 
There's a trick called ==Bagging== (==Bootstrap Aggregating==)
- Let's say we've got a model tht's not very good -- it's a decision tree that's pretty small, it hasn't used a lot of data...
	- It's producing a lot of errors -- it's not a systematically-biased error (not always too high, not always too low). 
- We could build another decision tree in a slightly different way (with different splits) ...
- ... and we could keep doing this, building lots of slightly different decision trees. All of these models are unbiased, better than nothing, have some errors a bit high, some a bit low, etc. 
- What would happen if we averaged the predictions?
- Assuming that the models (and their errors) aren't correlated with eachother, you'll end up with some errors on one side and some on the other... and the average is likely to be a good prediction!
- If we can generate a bunch of uncorrelated, unbiased models, we can average them and get something better than any individual model, because the average of the errors is going to be zero!
- So we just need a way to be able to build a lot of (unbiased, but different) models!
	- Let's do this by grabbing ==a random subset of the data each time==! Each of those decision trees is going to be not-great; but it will be unbiased; it will be better than nothing; they won't be correlated with eachother, because they're each random subsets... that meets our criteria for baggging!

This is how we create a ==Random Forest!==
Let's create one in four lines of code 😉

```python
def get_tree(prop=.75):
	# Given a proportion of data
	n = len(trn_y)  # Number of samples in the subset
	# At random, choose n*proprtion that we requested, from the sample
	idxs = random.choices(n, int(n, k=int(n*prop))
	# And build a decision tree from that subset of data!
	return DecisionTreeClasifier(min_samples_leaf=5).fit(trn_xs.iloc[idxs], trn_y.iloc[idxs]))


# Now let's build a lot of these trees
trees = [get_tree() for _ in range(100)]

# So our prediction is going to be the average of all of our weak predictor's predictions; let's make those predictions on our validation X data
preds = [tree.predict(val_xs) for t in trees]
# Then stack all the predictions up together and take their mean
# This is basically stacking a bunch of lists of specific-model-predictions on top of one another
# And then taking the average for each column (for each x[i])
# np.stack([[1,2,3], [4,5,6]]) becomes
# [1,2,3]
# [4,5,6]
# And then doing .mean() takes the averages per col
# [2.5, 3.5, 4.5]
avg_probs = np.stack(all_probs).mean(0)

# Now that we have an RF prediction for each x[i], let's get the MAE
mean_absolute_error(val_y, avg_probs)
# .2276
```

One other thing that RFs often do is select a random subset of columns to use in the decision tree.

One particularly nice feature of rando mforests is that they tell us which independent variables were the MOST IMPORTANT in the model
```python
pd.DataFrame(dict(cols=trn_xs.columns, imp))

```

We look at the underlying decision trees they create, and we look to see which columns they split on... And we can keep track, for each column, by how much they improved the GINI for a decision tree, and then add them up per column.
This gives us something called a ==feature importance plot==

![[Pasted image 20231128183243.png]]
It tells you how important each feature was; how often they picked it, and how much it improved the gini when they did!

Because random forests don't care about the distribution of your data, the categorical variables, etc.... That means that you can just go ahead and plot this right away, for your tabular data sets! If you've got a big data set with hundreds of columns, do this first and find the 30 out of your 100 columns that might actually matter!

Jeremy did some work in credit scoring, trying to find out who would default on a loan; given 7,000 columns in the DB, he put it straight into a RF and found that there were about 30 columns which were kinda interesting.
He went to the head of marketing/risk and told them "Here's the columns to focus on" and they were totally flabbergasted.

![[Pasted image 20231128183708.png]]
From the ScikitLearn docs:
- As you increase the number of estimators (# of trees in a RF), the accuracy improves; but there is going to be diminishing returns.

Jeremy doesn't often use more than 100 trees; this is sort of an arbitrary rule of thumb.

There' another interesting feature:

==Out-of-Bag error==
- Recall that in a Random Forest, that each tree is going to be trained on a SUBSET of the training data.
- The ==OOB Error== is a way of measuring the prediction error on the training set by ONLY including (in the calculation of a row's error ) the trees where that row was NOT included in training! This allows us to see whether the model is overfitting without needing a separate validation set.

----
JEREMY Intuition on this
- Since every tree was trained with a different randomly-selected subset of rows, out-of-bag-error is a little like imagining that every tree therefore also has its own *validation set*. The validation set is simply the rows that were not selected for that tree's training!
	- Ahh, makes sense 😺
---

This is particularly beneficial in cases where we only have a *small amount of training data*, as it allows us to see whether our model generalizes without removing items to create a validation set.
	* Don't have enough data to create a legit train/validation set split? You can learn from all of your data if you use an ensemble of weak tree-based classifiers (a random forest) and then use the out-of-bag error as your validation error to make sure you're not overfitting! then you get to learn from every piece of data.

This is called the OOB error! 
- It's built into Sklearn:
```python
r_mse(m.oob_prediction_, y)
```

-----
Question
- Question about bagging: We know that bagging is a poewrful ensemble approach to ML Should we try out bagging first when approaching a tabular task before deep learning? Can we create a bagging model that includes fastai deep learning models?
	- To be clear, bagging isn't a method/model itself; it's a meta-method that you can use with any model. I'd always start with a RF for my tabular data. You can always bag with any model, and other people rarely do -- but WE do! We'll be doing more today!
----

## ==Model Interpretation==
- For tabular data, model interpretation is particularly important! For a given model, the things that we are most likely to be interested are:
1. HOW CONFIDENT are we in our predictions using a particular row of data?
	- How confident are we that someone is going to repay their loan? "We think so, but we're not confident" = Give them a smaller loan
2. For predicting with a ***particular row of data***, what were the MOST IMPORTANT FACTORS, and how did they influence that prediction?
	- Let's say we rejected a loan; why?
3. WHICH COLUMNS are generally the STRONGEST PREDICTORS, and which can we ignore?
	- Generally, what makes someone a good fit for a loan?
4. WHICH COLUMNS ARE REDUNDANT with eachother, for the purposes of this prediction task?
	- What can we "not pay attention to?"
5. How do predictions vary, as we vary these columns?
	- As we vary the information that we put into our models (eg race, or some protected variable), how does that influence our model accuracy?

Let's see how RFs

#### Tree Variance for Prediction Confidence
- We saw how the model averages the individual trees' predictions to get an overall prediction; that is, an estimate of the value. But how can we know the confidence of the estimate?
	- One measure would be the *standard deviation of predictions across the trees*, instead of just the mean prediction (which is our prediction).
	- This tells us the relative confidence of predictions; in generally, we'd like to be cautious of using the results for rows where trees give *very different results* (higher standard deviations) compared to cases where they're more consistent.
In the earlier section on creating a random forest, we saw how to get predictions over the validation set, using a Python list comprehension to do this for each tree in the forest:

```python
preds = np.stack([t.predict(valid_xs) for t in m.estimators_])
preds.shape
# 40, 7988    (Our 40 estimators each make 7988 predictions on X)
```

Now we have a prediction for every tree and every auction (40 trees and 7988 auctions) in the validation set.
Now we can get the standard deviation of the predictions over all the trees, for each auction:
```python
# Get the standard deviation over the 0 axis (over each column, meaning for every x[i], get the standard deviation of predict[j](x[i]) for all j predictors)
preds_std = preds.std(0)
```
Here's the standard deviations for predictions for the first five auctions in the validation set:
```python
preds_std[:5]
# ...
```

#### Feature Importance
- We talked about this already; it's the bit where you look to see which splits were actually used in each of your weak predictors
#### Removing Low-Importance Variables
- It seems likely that we could use just some subset of the columns by removing the variables of low importance and still get good results.
	- Just because Jeremy has access to 7,000 columns, should he use all of them? He tends to remove the low importance variables.

#### Partial Dependence
- What's hte relationship between a column and the dependent variable? This is something called a ==partial dependence plot==, which is not something that's specific to random forests.
![[Pasted image 20231128191601.png]]
- This is from the BlueBooks for Bulldozers kaggle competition where we predict the sale price of a bulldozer.
	- We can see that more recently bulldozers are more expensive
	- We can see that as we go back and back to older bulldozers, they're less expensive, to a point.
	- You might think that you could easily create this plot by basically looking at your data in each year and taking the average sale price, but that doesn't work very well!
		- It turns out that one of the biggest predictosr of sale price for industrial equipment is whether it has air conditioning.
		- Most things didn't have AC in the 60s-70s.
		- If you plot the relationship between year made nad price, you're actually seeing "how popular is air conditioning"
		- What we really want is "What's the impact of the year, ALL ELES BEING EQUAL?"
		- The way we do this is we take our training dataset, we take every row for the year made column, and we set it to 1950; We then predict for every row what the sale price would have been if it were made in 1950
		- We then repeat that for 1951, 1952, and so forth. We then plot the averages.
		- The key words are the "ARE ELSE BEING EQUAL"; that's the data as it actually occurred, only varying the amount.
		- That owrks just as well for deep learning, gradient boosting trees, linear regressions, whatever!
		- And you can even do more than once column at a time!

Q: How did you decide to predict a certain category for a specific value? In other words, what were the most important factors for a specific row of data, and how did they influence the prediction?
A:
- we already looked at how to compute feature importance across the ENTIRE random forsest
	- The basic premise here was to look at hte contribution of each variable to improving the model at each branch of every tree, and then all up all these contributions per variable.
- We can do exactly the same thing, but just for a single row of data!
	- Let's say we're looking at a particular item at auction; Our model might predict that this item is very expensive, and we want to know why.
	- So we take that one row of data, nad put it through teh first decision tree, looking to see what split is used at each point throughout the tree.
	- For each split, we see what the increase or decrease in the addition is, compared to the parent node of the tree. We do this for every tree, and add up the total chagne.
- Then plot them like this!

![[Pasted image 20231128192246.png]]
This is an auction price prediction; this is how much each one impacted the price.
It's basically a feature importance plot, but just for a single row.

---
question:
- Would you ever EXCLUDE a tree from a forest if you had a very bad OOB error?
	- No, you wouldn't! If you start deleting trees, then you're no longer having an unbiased prediction of the dependent variable; you're biasing it by making a choice. Even the 'bad' ones will be improving the quality of the overall average.
- Re: Bagging: We could go on and create ensembles of bagged models. Is it reasonable to continue ensembles of bagged models?
	- At some point, it's just the average of averages... It doesn't buy you much as you add more and more.
- Can you overfit by adding too many trees?
	- Nope! 
	- You just need to make sure that you have *enough* trees :D 
---

#### Conclusion
- Clearly, more complex models are NOT always better!
- Our OneR model, consisting of a single binary split, was neaerly as good as our more complex models! Our random forest wasn't really an improvement on the single-tree model.
- So we should always be careful to BENCHMARK using simple models, to see if they're good enough for our needs!
	- In practice, you'll find that simple models don't have the requisite accuracy for more complex tasks like nLP, RecSys, etc
- Learned: RAndom Forests aren't actually that complicated at all! We're able to interpret hte key features of them in a notebook quite quickly, and they aren't sensitive to issues like normalization, interactions, or non-linear transformations, which make them extremely easy to work with and hard to mess up!


---------

# Gradient Boosting
- Like RFs, but isntead of fitting multiple trees on the data, we fit very small trees adn then ask: "Whats the error?"
	- Say we start with a 1R tree, make predictions, take the residual (different between actual and predicted) and create ANOTHER very small tree that tries to predict the residual of THAT, and so on. 
	- To calculate a prediction, you then take the SUM of all the trees predictions, rather than the AVERAGE.
	- This is called ==Boosting==, rather than ==Bagging==. They're both meta ensembling techniquess
	- When applied to trees, they are called ==Gradient Boosted Machines== and ==Random Forests==, respecitvely.
	- GBMs are "more accurate" than random forests, but you can ABSOLUTELY overfit (though there are ways to avoid overfitting)... It's just...fiddly. There's lots of hyperparameters you have to set, and it's a little more "Dangerous" than a RF, and you won't necessarily always get a meaningfully better model than a RF, so it doesn't seem like Jeremy is too hot on these.

We've been doing these daily walkthroughs with students about getting through the course, setting up machines, and things like that; we've been trying to practice things along the way -- a couple weeks ago, we wanted to pick a Kaggle competition and do the normal, sensible, mechanical steps that you'd do for a CV model. We spent a few hours predicting disease in rice, and then found that we were number one on the leaderboard!
That's interesting; there's all these other things we should be doing as well, though! He thought it'd be fun to take us through the process!

Here it is!

------
Since he's been doing more nad more stuff on Kaggle... he realized that there's some menial steps that he has to do each time, particularly because he likes to run stuffo n his own machine and then upload it to kaggle.
![[Pasted image 20231128194435.png]]
He created this module called fastkaggle that you can download using pip or conda that makes some things easier (eg downloading data, etc).

Kaggle competitions are nice because you can't hide from the truth; at work, you might be able to convince yourself and everyone around you that your model is better than everyone else, etc. But the brutal assessment of the private leaderboard will tell you the truth! Until you've been through that process, you're never going to know! 
- As you'll improve, you'll find that you have much confidence!
- The things you have to do in a Kaggle competition are just a SUBSET of the things that you'll have to do in real life (which include talking to people), but it's an IMPORTANT subset! Building a model that actually predicts things correctly and doesn't overfit is important!
- Structuring your code and analysis in a way that you can gradually keep improving your model over 3 months... that's all stuff that you want to be practicing, ideally well-away from customers or whatever :) 


So let's say we've got our data at `path` 

The overall approach (and not just to a kaggle competition) is something like this:
- The focus should be on two things
	- Creating an effective validation set
		- Sometimes this is easy! The test set in this competition seems to be a random sample, but many times it's not
	- Iterating rapidly to find changes which improve results on the validation set
		- This means NOT saying "What's the coolest big OpenAI model that I can train", but instead "What can I do that's going to train in a minute or so and will give you a sense of "I can do this, that, what's going to work" and then try 80 different things!
		- It also means NOT saying "Ah, there's some bayesian hyperparameter tuning approach, which gives you *one* model artifact after a lot of work"
		- Doing one thing really well will still get you in last place; You have to do pretty much everything well!
			- A team spent an entire 3 months building an amazing fancy thing, but they came in last place because they didn't iterate!

```python
comp = 'paddy-disease-competition'
path = setup_comp(comp, install='fastai timm>=0.6.2.dev0') # using fastkaggle

path
# Path('paddy-disease-classification')

# Let's import our vision shit and set our random seed
from fastai.vision.all import *
# This is just because we want to share this notebook; otherwise we shouldn't set the random seed!
set_seed(42)

path.ls()
#(#6) [Path(...), Path(..)]
# There's some training set images, some test images, ...

trn_path = path/'train_iamges'
files = get_image_files(trn_path) # Gets a list of all the filenames in the train images directory

# Let's look at the first!
img = PILImage.create(files[0])  # A pillwo image
# Recall: In the imaging world, they generally say cols x rows
# Whereas in the matematics world, we say rows x cols
print(img.size)
img.to_thum(128)
# (480, 640)
# Tall picture of grass

# So if you ask Pytorch what size this was, it would asy 640x480 -- he guarantees this will mess us up at some point! :)

# Looks like the images might all be 480x640; are they? Let's Check! We can do this in parallel by using fastcore's `parallel` submodule:
from fastcore.parallel import *

def f(o):
	return PILImage.create(i).size

# Let's get the sizes and then the count of images by size
sizes = parallel(f, files, n_workers=8)
pd.Series(sizes).value_counts()
# (480,640) 10403
# (640, 480) 4
# So indeed the vast majority are this 480 x 640 size! This basically tells Jeremy that we shoudl process them such that they're all 480x640

# The most common way to do things is just to squish or crop every image to be square (480x480 in this case).
dls = ImageDataLoaders.from_folder(
	trn_path,
	valid_pct=.2,  # 20% of the data used in the validation set
	seed=42,
	item_tfms=Resize(480, method='squish'),  # Doing this on the CPU means that we then have things in the same shape; then our minibatch can be operated on by the GPU in the batch transformations step
	batch_tfms=aug_transforms(size=128, min_scale=.75)  # Grab a random subset of the image and make it a 128x128 pixel.
)

# Let's see what some of that data would look like!
# We see some pictures of rice with various diseases
dls.show_batch(max_n=6)

```

#### Our first model
By creating a quick model, we can begin to understand our data!

We want a model that can iterate quickly, which means we want a model that can TRAIN quickly!
(Link: [The Best Vision Models for Fine Tuning](https://www.kaggle.com/code/jhoward/the-best-vision-models-for-fine-tuning): They looked at nearly 100 architectures from the timm model library (Pytroch image model) and looked to see which we could best fine tune (which had the best finetuning results). They did it on a pets dataset and a satellite imagery dataset)
The main things we measured were:
- how much memory did it use
- how accurate was it
- how long did it take to fit
And then combined those into a score

That ^ table up there is a useful score for picking a model! let's pick one that's really fast -- `restnet26d` - It's pretty good and very fast to train. A lot of people think that when they do deep learning, they have to learn about exactly how `resnet26d` is made... and we *will* cover that stuff, but it frankly almost never matters. It's just a function, and what matters are the inputs to it, and the outputs to it, and how fast and accurate it is.

Let's make a model and run LR Find!

```python
# Make our damn learner frmo our dataloader and a pretrained model architecture
learn = vision_learner(dls, 'resnet26d', metrics=error_rate, path.'.').to_fp16()

# Let's find a learning rate
learn.lr_find(suggest_funcs=(valley,slide))
# Looks like 10^-2 is good or so (.01)

# Let's just tune up our shit real quick, just for 3 epochs (Get some water for a minute)
learn.fine_tune(3, 0.01)

# Let's immediately submit our results to get something on the board!
ss = pd.read_csv(path/'sample_cusbmission.csv')
ss
# Looks like we have to have a column of the image filenames, and a column of labels

# Let's get all the image files in the test folder
tst_files = get_image_files(path/'test_images').sorted()
# And we want to be able to feed these to our model using the same ITEM transformations that our other dataloader used...
# We'll create a test dataloader! It just points at the test X
# 
tst_dl = dls.test_dl(tst_files)
# Let's now use this to do some inference!

# probabilities, targets (empty since it's a test set, which doesn't have labels), decoded indexes
probs, _, idxs = learn.get_preds(
	dl=tst_dl
	with_dedcoded=True # For a classification problem, we can set this; rather than get decoded the prorbability of each rice disease, it will also tell you the index of the MOST likely rice disease
)

# Our idxs are things like 0-9, which represnt the labels. let's look up our vocab ot get a list
dls.vocab
# ['bacterial_leaf_blight', 'bacterial_leaf_streak', ...]

# Now let's map our idx to stirngs
mapping = dict(enumerate(dls.vocab))
results = pd.Series(idxs.numpy(), name="idxs").map(mapping)
results
# A pd Series of our diseases
# So we've got our sample submission file ss, let's replace the label column with our submissions!
ss['label'] = results
ss.to_csv('subm.csv', index=False)
!head subm.csv
# File displays, looks reasonable

# Let's submit to Kaggle!
if not iskaggle:
	from kaggle import api
	api.competition_submit_cli('subm.csv', 'initial rn2d 128px', comp)

# Submitted! :)
# It was a bad submission -- top 80% of teams [also known as bottom 20%]
# This isn't that surprising! It's something to stasrt with, though! Tomorrow, we can try to make a slightly better model!
 
```


Addendum
- `fastkaggle` also provides a function that pushes a notebook to Kaggle Notebooks; 
```python
if not is_kaggle:
	push_notebook('jhoward', 'first-steps-road-to-the-top-part-1',
	title='First Steps: Rorad to the Top, Part 1',
	file='first-steps-road-to-the-top-part-1.ipynb',
	competition=comp, private=False, gpu=True)
```
If you can create models that predict things well, and communicate those results in a clear way, then you're a good data scientist! :) 

-----
Questoin:
- Do you create different Kaggle Notebooks for each model you try? Or do you append to the bottom of the notebook?
	- A: In the 6 hours of going through the daily videos, you'll see him create all the notebooks!
		- He usually duplicates his notebook to create another one, and then just delete all the stuff he doesn't need.
- How do you go about testing different models (SVM, RL, DL, etc)
	- A: I use AutoML less than anyone that I know, which is to say never. He never does hyperparameter optimization either
	- He likes being highly intentional -- he likes to think like a scientist and have hypotheses and test them carefully, and come up with conclusions, which he implements!
	- In this example, he didn't try a huge grid search of every possible model, learning rate, preprocessing approach, etc.
	- Step 1 was just to find out: Which things matter?
		- Does whether we squish or crop make a difference?
		- Are some models better with squish or crop?
		- A: In every case, the same thing turned out to be better.
		- Learning Rates
			- Most people do a grid search over learning rates a few years ago; Leslie Smith invented this learning rate finder a few years ago, and that's what Jeremy's used ever since!
		- Otherwise, rule of thumb for tabular:
			- RF is going to be the fastest-easiest to get a good result
			- GBMs can give a slightly better result but take some fussing. Honestly he probably would run a hyperparameter sweep for this, because it's fiddly and fast.


---

Had a couple thoughts about this:
- That thing trained in a minute on his home computer, and then on Kaggle it took about 4 minutes per epoch
	- Kaggles GPUs aren't amazing, but they aren't THAT bad!
		- It turns out they only have two virtual CPUs! you generally want about 8 CPUs per GPU
		- It spent all its time reading the damn data! 
		- The data was 640x480 and we were ending up with 128px for speed
		- There's no point in doing that every epoch!
		- So let's make our Kaggle iteration faster as well

```python
# Let's 
trn_path = Path('sml')
resize_images(path/'train_images', dest=trn_path, max_size=256, recurse=True)  # Recursively here will recreate the same folder structure at your trn_path; This is now our training data!

# When we trained on THIS on Kaggle, it went down to 4x faster with no loss of accuracy :) 
```

----
Still,.... we noticed that the GPU usage bar in Kaggle was still nearly empty, so we're still CPU bound! This means we should be able to use a more capable model with no negative impact on speed!

We look at the "best vision models for fine tuning" notebook from Jeremey again.

We're going to look at the ConvNext family of models
- We were looking at resnet26d which took 69.39; convnext_tiny_in22k is nearly the best on the dataset, but still quite fast at around 93.xx

```python
arch = 'convenext_small_in22k'
learn = ...
```

We end up getting a 4.5% error rate, more than twice as good as our previous model

A lot of people haven't heard of ConvNext
- ResNets are still probably the fastest, but for the mix of speed and performance, probably not. 
- If you're not sure what to do, use ConvNext!
	- Like most things, there's different sizes (tiny, small, large, xl, etc)


Instead of Cropping, we can Pad
- Padding is interesting because it's the only way of preprocessing images that DOES NOT DISTORT and DOES NOT LOSE INFORMATION
- The downside is that there are pictures that are literally "pointless"
	- So there are compromises
- This is frankly not used enough, and it can work quite well.

What else can we do?
- We can do ==test-time augmentation}== !
	- During inference or validation, we create multiple versions of each image using data augmentation, and then take the average or maximum of the predictions for each augmented version of the image.
 
- This is all "the same picture" that's gone through data augmentation (darker, flipped, rotated, warped, zoomed)
- Maybe our model would like some of these versions better than others
- We can pass ALL OF THESE to our model, get predictions for all of them, and take the average
	- It's our own kind of ... mini bagging approach

```python
# Will pass multiple augmented versions of the image and average them for you!
tta_preds, _ = learn.tta(dl=valid)
```


Scaling Up
- now that we've got a good model and preprocessing approach, we can scale it up to larger images and more epoches; we'll switch back our path to the original un-resize images
- Nearly all of our images are 640x480 (some were 480 x 640)
- We can just use that aspect ratio (256, 192) and it will resize everything to the asme aspect ratio rectangular

```python
learn = train(
	arch,
	epochs=12,
	item=Resize((480,360), method=ResizeMethod.pad, pad_mode=PadMode.Zeros),
	batch=aug_transforms(size=(256,192), minscale=.75)  # Just for training...Same 1.3333 aspect ratio
)

# Error rate is now down to 2.2%
# Addindg back in TTA, it's now down to 1.97%
```


-----
Jeremy finds most of the time that datasets have a wide range of input sizes and aspect ratios; If there are just as many tall skinny ones as wide, short ones, it doesn't make sense to create a rectangle, because some will be destroyed; so a square is the best compromise.
There are better things we can do.... but we don't have better off-the-shelf support for this in our library
We could batch things in a similar aspect ratio together...

Q: The issue with padding as you say... the black pixels
- Those aren't NaNs, those are BLACK PIXELS -- there's something wrong about that, conceptually! When you see 4:3 aspect ratio footage on 16:9, you get blurred stuff on the top/bottom. 
- A: We've played with that! fastai by default sues *reflection padding*, which looks pretty good -- another one is "copy," which simply takes the outside pixel and copies it out.
	- Much to Jeremy's chagrin, it turns out that none of these really help that much
	- in the end, the computer wants to know "this is end of the image, there's nothing here!"



