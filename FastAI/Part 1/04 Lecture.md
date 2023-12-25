This is going to be a lectuer on NLP, and we're going to not even be using the `fastai` library; instead, we're going to be using the hugging face transformers library, and we're going to be using the transformers architecture too! It's the state of art for NLP.

At the end, we'll know how to do NLP in a very fantastic library. `HuggingFace Transformers` doesn't have the same architecture as `fastai` does. For beginners, the high-level API that you'll be using most of the time isn't quite as ready-to-go for beginners as you'll be used to, for `fastai` .

But it's good to get some experience using a *little bit* of a lower-level library! This is how the rest of the course is generally going to be, on the whole.

Let's talk about what we're going to be doing with respect to fine-tuning a pre-trained model:
- We played with some sliders last week

![[Pasted image 20231125121535.png]]
- Where we were approximating a quadratic function. The job was to move the sliders to get this as nice as possible. The person who gave you these sliders said ("Oh, slier A should be on 2.0, and slider B should be around 2.5, and slider C we have no idea."). That'd be pretty helpful, wouldn't it! We could just sort of start with A and B pretty much where they think they should be, and then just wiggle them a bit to get the line best-fit. That's what a pretrained model is! A bunch of parameters have already been fit, where some of the parameters are already about where they should be, and others need to be moved quite a lot.

The idea of fine-tuning a pre-trained NLP model in this way was pioneered by an algorithm called `ULMFit`, which was first presented in the first `fastai` course! It was later turned into an academic paper! It went on to inspire a huge change in NLP capabilities around the world, along with a number of important innovations.

The basic process that ULMFit describes is:
1. Build something called a ==Language Model== using all of WIkipedia
	- This model tries to *predict the next word* of every Wikipedia article.
	- Doing this is difficult! Some articles said things like "The 17th prime number is..." or "The 40th president of the US blah said, at his residence blah, ...""
	- To be good at being a language model, a Neural Net needs to be pretty good! It needs to understand language at a good level, and also have knowledge of specific things about (eg) Rutherford B Hayes.
	- We started with random weights, and at the end, had a model that could predict ~30% of the time what the next token was
2. Then, we tried to figure out whether IMDB movie reviews were positive or negative sentiment. We created a *second* language model (predicts next word of sentence), but started with the pretrained model on Wikipedia, and ran a few extra epochs using IMDb movie reviews! It got good a
3. Then we took those weights, and fine-tuned them on predicting whether a movie on IMDb was positive or negative.


![[Pasted image 20231125122033.png]]
- It's interesting because steps 1 and 2 there don't require us to collect a bunch of labeled data -- because the "labels" were just... "What's the next word?"

Since we built ULMFit, and we used RNNs (recurrent neural networks) for this... at about the same time we released this, a new type of architecture specifically useful for NLP called ==Transformers== was released!
- These models can take great advantage of modern accelerators (like Google's TPUs (Tensor Processing Units)).
- They didn't really *allow* you to predict the next token -- that's just not how they're structured. Instead, they did something just as good, and pretty clever, which is take chunks of Wikipedia, and deleted at random a few words, and asked the model to predict which words were deleted, essentially. It's a pretty similar idea (I think this is called a ==Masked Language Model==.

Otherwise, it's a pretty similar idea!

You might remember from lesson 1 that we looked at a Zeiler+Fergus paper where we looked at the visualizations of the first layer of an ImageNet classification model:
- It seemed like layer 1 was capturing edges, color gradients, etc.
- It seemed like layer 2 combined those into circles
- It seemed like layer 5 had bird and lizard eyeball detectors, and dog face detectors, and flower detectors, and so forth. 

Something like a resnet-50 has *fifty layers*! These later layers do something that's very specific to a training task (ie what we're looking at), but it's pretty unlikely that we're going to need to change those too much (you're going to need edge detectors and curve detectors)

(Handwavey): In the finetuning process, there's actually a layer at the end that says "What is this?" (a dog or a cat) -- we delete this/throw it away. The model now is spitting out some matrix of a few hundred activations. What we do is just stick a new random matrix on the end of that, and that's what we initially train. It learns to use the already-learned features to predict whatever it is that you want to predict! :) ✨

Let's jump into the notebook: [Getting started with NLP for absolute beginners](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners) 

Recall: Classification refers to taking some object (a picture) and identifying which category that image belongs to. We've mainly been using this in the context of images, but now we're going to apply it to documents (where document could be anywhere from 3-4 words, or a document could be all of wikipedia as text).

Classifying a document is a surprisingly rich thing to do; there's all sorts of stuff that you could do with it! We've already mentioned sentiment analysis (positive, negative). Author identification would be taking the document and finding the category of author! Legal discovery would be "Is this document germane to my court case?". Emails being put into categories of Spam, sendToCustomerService, sendToSales, etc.

For people interested in trying out NLP in real life, Classification is a good place to start.

The Kaggle competition contains data that looks like this:
![[Pasted image 20231125123254.png]]
Above:
- I think these are from Patent applications, and on the patents there are various things they have to fill in on the patent. One of those things are called anchor, and one is called target. The goal of the competition is to figure out which anchor and target pairs are talking about the same thing.

This is... kind of a classification problem, kind of not. We're basically trying to classify into "the same" or "not the same." There's also a category called context, which is basically the category that the patent was filed in.

So... how do we take this and turn it into something like a classification problem?

Jeremy suggests:
- By representing the question like this:
	- "For the following text...: *"TEXT1: abatement; TEXT2: eliminating process"* .... choose a category of meaning similarity: "Different, Similar, Identical".
- So basically we can concatenate the columns together, and treat it as a document. That's just an example of how we can take this ... "similarity" problem and turn it into something that looks like a "classification" problem -- we do this a lot in deep learning, where we turn problems that don't look like things we recognize, and turn them into things we recognize.

... Some stuff in my notebook here: https://www.kaggle.com/code/samsilver/getting-started-with-nlp-for-absolute-beginners/edit ...

Split the data into ==tokens== (basically, words; often things called ==subwords==). We're going to generate a list of all of the unique words, called the ==vocabulary==, and each will get a number. In general, we don't want the vocabulary to be too big. 
We're going to later turn these tokens in the vocabulary into numbers, in a process called ==numericalization==. 

We don't want ==underfitting== OR ==overfitting==.
- Underfitting is easy to recognize, because we can look at the training data and see that we aren't doing well
- Overfitting is hard to recognize, because it seems like we're doing quite well on the training data.

So how do we tell if we're overfitting?
- We take our original dataset, and remove say 20% of the data.
- We then fit our model, using only those points that we haven't removed. Then we measure how good the model is by looking at ONLY the points that we removed!
- The data that we take away and don't let the model see when it's training is called the ==validation set==. `fastai` won't even let you train a model without a validation set. It shows you your metrics on a validation set.

HuggingFace transformers is also good about this; they make sure to show you your metrics on a validation set.
Creating a good validation set isn't as easy as just randomly pulling some data from your initial dataset.
- Imagine that you were trying to do some time-series data. In real life, you're going to want to predict *future* dates. If you just randomly remove data, that's not going to be a good example of how you're going to use the model. Instead, you should use the last few weeks as the validation set.

Kaggle competitions allow you to only submit a couple times a day. The dataset that you're scored on is a separate subset to the one that you'll be scored to at the beginning of the competition. It's not until you've done it that you'll realize (as a beginner on Kaggle) that you're probably overfitting. In the real world, you probably won't even know that you're overfitting! You'll just destroy value silently 😉. 
- So it's important to know how to create a good validation set, on Kaggle first!
- A good example of this is a distracted driver competition on Kaggle. The idea was that you have to predict if someone was driving in a distracted way or not.
	- On Kaggle, the test set contained people that didn't exist in the competition data that you trained the model with
	- So if  you wanted to create an effective validation set, you need to be sure that your validation set contains people that *aren't* in the data that you're training the model on
- In a Kaggle fishery competition (what fish is in the picture), there were boats that weren't in the training dataset. Some people overfit because certain boats were used to catch certain fish, in the provided datset.


==Cross-Validation==
- Be very careful! This is explicitly *not* about building a good validation set, so you need to be super careful if you don't do that. 
- Sklearn/HuggingFace/Fastai conveniently offers something called train/test split...
	- It can almost feel like we're encouraging you to use a random validation set, but ***BE CAREFUL***, because that's often NOT what you want for your problem..

So what's a ==Test Set==? It's basically another *validation set, but you don't use it for checking your accuracy while you build your model.
- Say you're trying 3 models a day for 60 days, and you're looking at the validation set accuracy for each ones. For some of those models, you got a good score on the validation set perhaps by chance! And when you submit, you messed up because **you overfit on the validation set**!
- On Kaggle they have two tests sets, even!
	- One gives you feedback on the competition leaderboard during the competition
	- One is secret and is used to crown the ewinner

In real life:
- Don't try so many models that you'll just overfit to the validation set!
- Only if you have a test set that you've held out will you know that

So what happens when:
- You have 3 months of model training, etc. With a hold-out test set
- And you finally run your model against the test set...and it sucks.
- What do you do? You really have to go back to square one.

So what do you want to do with the validation set?
- You want to measure some ==metrics==
	- Something *like* Accuracy; some number that tells you how good is your model
- A Kaggle competition might tell you in the abstract how the models will be evaluated (eg the Pearson Correlation Coefficient)
- Question:
	- Is this the thing that we should use as our loss function?
	- Answer:
		- Maybe, sometimes, but probably not. For example, consider accuracy: If we were using accuracy to calculate our derivative/get the gradient... We could have a model that is doing slightly better at categorizing... but not so much butter that it doesn't change the model accuracy in terms of classifying cat and dog. So the gradient is zero. You don't want bumpy functions like this, because they don't have nice gradients. You want a function that's nice and smooth, like the average absolute error, or something.
	- So be careful! When you're training, your model is trying all its time to improve its *loss*, which often isn't the same thing as what you actually care about, which is the *metric*, often. In real life, you can't be told what metric to use! In real life, the model you choose... there isn't one number that tells you if it's good or bad, and even if there was, you can't find it ahead of time. In real life, the model you use is part of a complex process involving humans (as users, customers) and there are lots of outcomes of decisions that are made. One metric isn't enough to capture all that.
		- Unfortunately, people will roll out models based on a single metric that are easy to measure, rather than those that actually result in the best outcomes in reality.
			- "When a measure becomes a target, it ceases to be a good measure."
			- https://www.fast.ai/posts/2019-09-24-metrics.html

Re: that Pearosn Correlation Coefficient (*r*)
- It's the most widely used measure of "how similar two vairables are"
	- So if your predictions are very similar to the real values, then the *r* will be high; that's what you want.
	- r can be between -1 and 1; -1 means you predicted exactly the wrong answer (which is fine, just reverse your prediction). +1 means you got everything right.
- In textbooks, they'll show you some mathematical function in a book. Jeremy's not going to do that! We don't care about the function, we care about how it behaves!
- Let's look at a bunch of datasets to understand how r behaves. Let's look at real-life data:

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
# How do I graph a dataset with too many points? Just graph fewer points. There's no reason to plot huge amounts of data; grab a random sample
# Below: We join two series into a df
housing = housing['data'].join(housing["target"]).sample(1000, random_state=52)

housing.head()
# Some data

# Numpy has something called np.corrcoef that gets the pairwise r between every variable
np.set_printoptions(precision=2, suppress=True)
# Get hte correlation coefficient of our 
np.corrcoef(housing, rowvar=False)
```
![[Pasted image 20231125145318.png]]


Re: Outliers and removing them 
"Outliers should never just be removed..."
- It seems like there's a separate group of houses with a different kind of behavior. And so Jeremy would be saying "clearly from looking at this dataset, these two groups of houses can't have the same behavior, so I'd split them into two separate groups and do separate analysis."
- "Some of the most useful insights that I've seen have been digging into outliers, and thinking about where they came from. Often in those edge cases you can discover interesting things about where processes went wrong, or about labelling problems, processing problems ,etc. So never just delete outliers without investigating them and having some strategy for understanding where they came from."

NLP, because it's really become effective in the last year or two, is probably the place where there's the biggest wins, both commercially and for research. Re: "Why now," the answer for NLP is simple: "Because until last year, this wasn't possible, unless you spent 10x the time or 10x the money." NLP is a huge opportunity area.

It's worth thinking about both use and misuse of modern NLP.
Let's look at a subreddit:
![[Pasted image 20231125151751.png]]
Question: What subreddit do you think this is frome?
Answer: It's from a subreddit that posts automatically generated conversations between GPT models (old ones). Despite being weird, this is still context-appropriate, vaguely-believable pros!
- Any of us upper-tier `fastai` alumni can create a bot that creates content-appropriate bots that argue for a size of an argument; You can scale this up so that 99$ of twitter are these bots... and no one would know. And that's very worrying to Jeremy. A lot of the kind of the way that people see this world is coming out of these conversations; and at this point that's controllable -- it wouldn't be hard to make something that's optimized to moving the point of view of people over time.
- eg: More than a Million Pro-Repeal Net Neutrality comments were likely faked!















