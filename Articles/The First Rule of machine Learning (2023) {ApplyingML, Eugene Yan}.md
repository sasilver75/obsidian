https://applyingml.com/resources/first-rule-of-ml/

Applying machine learning is difficult! You need
- data
- a robust pipeline to support data flows
- high-quality labels

As a result, most of the time your first iteration might not even involve machine learning at all!

> The first rule of machine learning: Start without machine learning!
> 
> - Eugene Yan (@[eugeneyan](https://twitter.com/eugeneyan?lang=en))

This seems strangle, right?
But if you check out [[Google's 43 Rules of Machine Learning (2023) {Google}]], you'll see:

> Rule #1: Don't be afraid to launch a product without machine learning!
> 
> Machine learning is cool, but it requires data. Theoretically, you can take data from a different problem and then tweak the model for a new product, but this will likely underperform basic heuristics, once you understand the problem.

Several machine learning practitioners that I interviewed as part of the ApplyingML blog also responded similarly to the question:
> "Imagine that you're given a new, unfamiliar problem to solve with machine learning. How would you approach it?"

> "I'd first try really hard to see if I could solve it without ML! I'm all about trying the less glamorous stuff before moving on to any more complicated solutions."
> - Vicki Boykis (ML Engineer @ Tumblr)

> "I think it's important to do it without mL first. Solve the problem manually, or with heuristics. This will force you to become innately familiar with the problem and the data, which is the most important step. Furthermore, arriving at a non-ML baseline is important in keeping yourself honest!"
> - Hamel Hussain (Staff ML Eng @ Github)

> "First, try to solve it without ML. Everyone gives this advice because it's good. You can write some if/else rules or heuristics that make some simple decisions and take actions as a result."
> Adam Laciano, Staff Eng @ Spotify


### What should we start with then, if not ML?
- Regardless of whether you're using simple rules or deep learning, it helps to have a decent understanding of the data! Thus, grab a sample of the data to run some statistics and visualize! 
	- This mainly applies to tabular data, but other data like images, text, audio, etc. can be trickier to run aggregate statistics on.
- **Simple correlations** can help figuring out the relationships between each figure and the target variable! Then, we can select a subset of features, with the strongest relationships, to visualize.
- Not only does this help with understanding the data and problem, it will help us apply machine learning more effectively later on! We also gain better context on the business domain.
	- Note that correlations and aggregate statistics can be misleading! 
	- Note that variables with strong correlative relationships can have zero causal effect, and variables that have strong relationships can appear to have zero correlation!

- Scatter plot are a favorite for visualizing numerical values -- have the features on the X-Axis and the target variable on the Y-Axis, and let the relationship reveal itself to you!
![[Pasted image 20231224230028.png]]
*Above: Looks like people are buying more and more ice cream, up to a certain temperature Celsius, and beyond that, sales dip again.*

- If either variable is categorical, box plots often work well! Imagine you're trying to predict the lifespan of a dog -- how does the size of the dog matter?
![[Pasted image 20231224230120.png]]
*Above: The box plot is basically distributions shown across the categories of a categorical predictor. You could alternatively use one of those violin plots that actually shows the distribution.*


- With an understanding of the data, we can then start by solving the problem with heuristics! Here are some examples of using heuristics to solve common problems:
	- `Recommendations`
		- Recommend top-performing items from the previous period! You can further segment this by categories (eg genres, brands). 
		- If you have customer behavior, you can compute aggregated statistics on co-interactions to calculate item similarity for i2i recommendations (these are item-to-item recommendations; given an item, we recommend other items).
	- `Product classification`
		- Regex-based rules on product titles. If the product contains "ring", "wedding band", "diamond", "\*bridal," etc., classify it in the ring category.
	- `Review spam identification`
		- Rules based on:
			- The count of reviews from the same IP
			- Time the review was made (odd timing like 3am)
			- Similarity (e.g. edit distance) between the current review and other reviews made on the same day

There are a bunch of other techiques like regex, interquartile range for outlier detection, moving average for forecasting, building dictionaries for address matching, etc.

> You might catch spammers because they're using images with the same filenames.
> - Jack Hanlon (@JHanlon)

>I've gotten so much flack for this. One project I did with string comparisons, but the customer was disappointed that I didn't use NNs and hired someone else to do that. Guess which one was cheaper and more accurate?
>- Mitch Haile (@bwahacker)
 
Yeah, you might say that those people training the machine learning models didn't know what they were doing -- perhaps! Nonetheless, the point is that understanding the data and simple heuristics can easily do better than `model.fit()`, and in less than half the time!

These heuristics also help with bootstrapping labels (aka weak supervision)!
- If you're starting from scratch and don't have any labels, weak supervision is a way to quickly get lots of labels efficiently - albeit perhaps at lower quality.
- These heuristics can be formalized as learning functions to generate labels. Other forms of weak supervision include using knowledge base and pre-trained models.


### So when should we use machine learning?
- After you have a non-ML baseline, that performs reasonably well, and the effort of maintaining and improving that baseline outweighs the effort of building a deploying an ML-based system.
	- Once you're at some 195-rule handcrafted system, that becomes hard to update without breaking something.

Google's Rules:
> Rule #3: Choose machine learning over a *complex heuristic*.
> 
>A simple heuristic can get your  product out the door. A complex heuristic is unmaintainable. Once you have some data and a basic idea of what you're trying to accomplish, move on to machine learning! You'll find that the machine-learned model is easier to update and maintain.

Having robust data pipelines and high-quality data labels also suggest that you're ready for machine learning. Before this happens, your existing data might not be good enough for ML. Or you might have the data, but it's in such a bad state that it's unusable! For example maybe the merchants on your e-commerce platform are deliberately misclassifying products to game the system!

Often, *manual labeling* is required to bootstrap a `golden dataset` of high-quality labels!
With it, training and *validating* your ML efforts become much easier!


### But what if I need to use ML, just for the sake of it!?
- Hmmm... that's a tough position.

> ML Strategy tip:
> -  When you have a problem, build *two solutions!* A deep Bayesian transformer running on multicloud Kubernetes, and a SQL query built on a stack of egregiously-oversimplifying assumptions.
> - Put the former on your resume and the latter in production. Everyone goes home happy.
> - .
> Brandon Rohrer (@\_\_brohrer\_\_) 































